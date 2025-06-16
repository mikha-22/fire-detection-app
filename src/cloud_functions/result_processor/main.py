# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path

import pandas as pd
from google.cloud import storage, aiplatform
from PIL import Image
import folium
from src.map_visualizer.visualizer import MapVisualizer
from src.common.config import GCS_PATHS, FILE_NAMES

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

# --- Initializations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

storage_client = storage.Client()

try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logger.info("Vertex AI client initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize Vertex AI client. Error: {e}")

def get_vertex_ai_job_info(job_metadata_path):
    """
    Get Vertex AI job information from our metadata file.
    """
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        metadata_blob = bucket.blob(job_metadata_path)
        metadata = json.loads(metadata_blob.download_as_string())
        return metadata
    except Exception as e:
        logger.error(f"Failed to load job metadata: {e}")
        return None

def wait_for_vertex_ai_output(bucket, raw_output_prefix, max_wait_seconds=300):
    """
    Wait for Vertex AI to create prediction output files.
    Returns list of prediction.results files when found.
    """
    start_time = time.time()
    check_interval = 10
    
    logger.info(f"Waiting for Vertex AI output in: {raw_output_prefix}")
    
    while time.time() - start_time < max_wait_seconds:
        all_blobs = list(bucket.list_blobs(prefix=raw_output_prefix))
        
        # Vertex AI creates a subdirectory with timestamp, then puts prediction.results files in it
        prediction_files = [b for b in all_blobs if 'prediction.results' in b.name]
        
        if prediction_files:
            logger.info(f"Found {len(prediction_files)} prediction files after {time.time() - start_time:.1f} seconds")
            return prediction_files
        
        # Log what we're seeing
        subdirs = set()
        for blob in all_blobs:
            parts = blob.name.split('/')
            if len(parts) > len(raw_output_prefix.split('/')):
                subdir = '/'.join(parts[:len(raw_output_prefix.split('/')) + 1])
                subdirs.add(subdir)
        
        if subdirs:
            logger.info(f"Found subdirectories: {list(subdirs)[:3]}... but no prediction.results yet")
        
        logger.info(f"No prediction files yet, waiting {check_interval} seconds...")
        time.sleep(check_interval)
    
    logger.warning(f"Timeout after {max_wait_seconds} seconds waiting for prediction files")
    return []

def process_vertex_ai_output(prediction_files, bucket, job_id, run_date):
    """
    Process raw Vertex AI output and save cleaned results.
    Returns path to processed predictions file.
    """
    all_predictions = []
    
    for pred_file in prediction_files:
        logger.info(f"Processing prediction file: {pred_file.name}")
        content = pred_file.download_as_string().decode('utf-8')
        
        for line in content.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                # Parse Vertex AI output format
                output_data = json.loads(line)
                
                # Extract instance and prediction
                instance = output_data.get('instance', {})
                predictions = output_data.get('prediction', [])
                
                # Create cleaned prediction record
                for pred in predictions:
                    cleaned_pred = {
                        'instance_id': instance.get('instance_id'),
                        'cluster_id': pred.get('instance_id', instance.get('instance_id')),
                        'detected': pred.get('detected', False),
                        'confidence': pred.get('confidence', 0.0),
                        'detection_details': pred.get('detection_details', ''),
                        'processed_at': datetime.utcnow().isoformat() + 'Z'
                    }
                    all_predictions.append(cleaned_pred)
                    
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                continue
    
    # Save processed predictions
    processed_output_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_processed_output']}/{FILE_NAMES['batch_predictions']}"
    
    jsonl_content = '\n'.join([json.dumps(pred) for pred in all_predictions])
    bucket.blob(processed_output_path).upload_from_string(jsonl_content)
    
    logger.info(f"Saved {len(all_predictions)} processed predictions to: {processed_output_path}")
    
    # Also save a summary
    summary = {
        'total_predictions': len(all_predictions),
        'fire_detections': sum(1 for p in all_predictions if p['detected']),
        'no_fire_detections': sum(1 for p in all_predictions if not p['detected']),
        'average_confidence': sum(p['confidence'] for p in all_predictions) / len(all_predictions) if all_predictions else 0,
        'processed_at': datetime.utcnow().isoformat() + 'Z'
    }
    
    summary_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_processed_output']}/{FILE_NAMES['job_summary']}"
    bucket.blob(summary_path).upload_from_string(json.dumps(summary, indent=2))
    
    return processed_output_path, all_predictions

def result_processor_cloud_function(request=None, context=None):
    """
    Process results from Vertex AI batch prediction job.
    """
    logger.info("Result Processor function triggered.")
    
    # Handle both HTTP and Pub/Sub triggers
    if request and hasattr(request, 'get_json'):
        logger.info("Triggered via HTTP request")
        request_json = request.get_json(silent=True) or {}
        run_date = request_json.get('run_date', datetime.utcnow().strftime('%Y-%m-%d'))
        job_id = request_json.get('job_id')
    else:
        logger.info("Triggered via Pub/Sub")
        run_date = datetime.utcnow().strftime('%Y-%m-%d')
        job_id = None

    if not all([GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION]):
        logger.critical("Missing required environment variables.")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": "Missing environment variables"}, 500
        return

    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # --- Step 1: Determine which job to process ---
    if not job_id:
        manifest_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{FILE_NAMES['job_manifest']}"
        try:
            manifest_blob = bucket.blob(manifest_path)
            manifest = json.loads(manifest_blob.download_as_string())
            
            if manifest.get("jobs"):
                latest_job = sorted(manifest["jobs"], key=lambda x: x["created_at"], reverse=True)[0]
                job_id = latest_job["job_id"]
                logger.info(f"Using latest job_id from manifest: {job_id}")
            else:
                logger.error(f"No jobs found in manifest for date {run_date}")
                if request and hasattr(request, 'get_json'):
                    return {"status": "error", "message": "No jobs found"}, 404
                return
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            if request and hasattr(request, 'get_json'):
                return {"status": "error", "message": str(e)}, 500
            return
    
    logger.info(f"Processing results for run_date: {run_date}, job_id: {job_id}")

    # --- Step 2: Load job metadata and wait for Vertex AI output ---
    job_metadata_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{FILE_NAMES['job_metadata']}"
    job_metadata = get_vertex_ai_job_info(job_metadata_path)
    
    if not job_metadata:
        logger.error("Failed to load job metadata")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": "Job metadata not found"}, 404
        return
    
    # Wait for Vertex AI to complete and create output files
    raw_output_prefix = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_raw_output']}/"
    prediction_files = wait_for_vertex_ai_output(bucket, raw_output_prefix)
    
    if not prediction_files:
        logger.error("No prediction files found after waiting")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": "No prediction files found"}, 404
        return
    
    # --- Step 3: Process the raw output ---
    processed_path, predictions = process_vertex_ai_output(prediction_files, bucket, job_id, run_date)
    
    # --- Step 4: Load additional data needed for visualization ---
    try:
        # Load incidents
        incidents_path = f"{GCS_PATHS['incidents']}/{run_date}/{FILE_NAMES['incident_data']}"
        incidents_content = bucket.blob(incidents_path).download_as_string().decode('utf-8')
        incidents_data = [json.loads(line) for line in incidents_content.strip().split('\n')]
        hotspots_by_cluster = {inc['cluster_id']: inc['hotspots'] for inc in incidents_data}
        
        # Load batch input to get image paths and bboxes
        batch_input_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_input']}/{FILE_NAMES['batch_input']}"
        input_content = bucket.blob(batch_input_path).download_as_string().decode('utf-8')
        
        input_metadata = {}
        for line in input_content.strip().split('\n'):
            instance = json.loads(line)
            for cluster in instance.get('clusters', []):
                input_metadata[cluster['cluster_id']] = cluster
                
    except Exception as e:
        logger.error(f"Failed to load supporting data: {e}")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": str(e)}, 500
        return

    # --- Step 5: Generate visualizations and save them ---
    report_images_path = f"{GCS_PATHS['reports']}/{run_date}/{GCS_PATHS['report_images']}"
    visualization_data = []
    
    visualizer = MapVisualizer()
    
    for pred in predictions:
        cluster_id = pred['cluster_id']
        
        # Get metadata for this cluster
        cluster_metadata = input_metadata.get(cluster_id)
        if not cluster_metadata:
            logger.warning(f"No metadata found for cluster {cluster_id}")
            continue
        
        try:
            # Download satellite image
            image_uri = cluster_metadata['gcs_image_uri']
            img_bucket_name, img_blob_name = image_uri.replace("gs://", "").split("/", 1)
            image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
            
            # Get FIRMS hotspots
            hotspots = hotspots_by_cluster.get(cluster_id, [])
            firms_df = pd.DataFrame()
            if hotspots:
                firms_df = pd.DataFrame.from_records([h['properties'] for h in hotspots])
            
            # Generate visualization
            map_image = visualizer.generate_fire_map(
                base_image_bytes=image_bytes,
                image_bbox=cluster_metadata['image_bbox'],
                ai_detections=[pred],
                firms_hotspots_df=firms_df,
                acquisition_date_str=run_date
            )
            
            # Save image to GCS
            image_filename = f"{cluster_id}.png"
            image_path = f"{report_images_path}/{image_filename}"
            img_byte_arr = BytesIO()
            map_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            bucket.blob(image_path).upload_from_string(
                img_byte_arr.getvalue(),
                content_type='image/png'
            )
            
            # Store visualization data for report
            visualization_data.append({
                'cluster_id': cluster_id,
                'image_path': f"gs://{GCS_BUCKET_NAME}/{image_path}",
                'latitude': (cluster_metadata['image_bbox'][1] + cluster_metadata['image_bbox'][3]) / 2,
                'longitude': (cluster_metadata['image_bbox'][0] + cluster_metadata['image_bbox'][2]) / 2,
                'detected': pred['detected'],
                'confidence': pred['confidence']
            })
            
            logger.info(f"Generated visualization for cluster {cluster_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualization for {cluster_id}: {e}")
            continue

    # --- Step 6: Generate final HTML report ---
    if visualization_data:
        logger.info(f"Generating final report with {len(visualization_data)} visualizations")
        
        # Create Folium map
        m = folium.Map(location=[-2.5, 118], zoom_start=5)
        
        for viz in visualization_data:
            # For the HTML report, we'll embed base64 images
            image_path = viz['image_path'].replace(f"gs://{GCS_BUCKET_NAME}/", "")
            try:
                image_blob = bucket.blob(image_path)
                image_bytes = image_blob.download_as_bytes()
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                
                popup_html = f"""
                <h4>Cluster ID: {viz['cluster_id']}</h4>
                <p>
                  <b>AI Detection:</b> {'FIRE DETECTED' if viz['detected'] else 'No Fire Detected'}<br>
                  <b>Confidence:</b> {viz['confidence']:.2%}
                </p>
                <img src='data:image/png;base64,{encoded_image}' width='400'>
                """
                
                iframe = folium.IFrame(popup_html, width=430, height=450)
                popup = folium.Popup(iframe, max_width=430)
                marker_color = 'red' if viz['detected'] else 'green'
                
                folium.Marker(
                    location=[viz['latitude'], viz['longitude']],
                    popup=popup,
                    tooltip=viz['cluster_id'],
                    icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
                ).add_to(m)
                
            except Exception as e:
                logger.error(f"Failed to add marker for {viz['cluster_id']}: {e}")
        
        # Save report
        report_filename = f"wildfire_report_{run_date}_{job_id}.html"
        report_path = f"{GCS_PATHS['reports']}/{run_date}/{report_filename}"
        
        local_temp_path = f"/tmp/{report_filename}"
        m.save(local_temp_path)
        
        bucket.blob(report_path).upload_from_filename(local_temp_path, content_type='text/html')
        
        logger.info(f"Report saved to: gs://{GCS_BUCKET_NAME}/{report_path}")
        
        # Save report metadata
        report_metadata = {
            'report_path': f"gs://{GCS_BUCKET_NAME}/{report_path}",
            'job_id': job_id,
            'run_date': run_date,
            'visualization_count': len(visualization_data),
            'fire_detections': sum(1 for v in visualization_data if v['detected']),
            'generated_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        metadata_path = f"{GCS_PATHS['reports']}/{run_date}/{FILE_NAMES['report_metadata']}"
        bucket.blob(metadata_path).upload_from_string(json.dumps(report_metadata, indent=2))
        
        # Update job metadata to completed
        job_metadata['status'] = 'completed'
        job_metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
        job_metadata['report_path'] = f"gs://{GCS_BUCKET_NAME}/{report_path}"
        bucket.blob(job_metadata_path).upload_from_string(json.dumps(job_metadata, indent=2))
        
        if request and hasattr(request, 'get_json'):
            return {"status": "success", "report_url": f"gs://{GCS_BUCKET_NAME}/{report_path}"}, 200
    else:
        logger.error("No visualizations generated")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": "No visualizations generated"}, 500

    logger.info("Result Processor completed successfully")
    return
