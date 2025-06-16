# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
import time
from io import BytesIO
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
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        metadata_blob = bucket.blob(job_metadata_path)
        metadata = json.loads(metadata_blob.download_as_string())
        return metadata
    except Exception as e:
        logger.error(f"Failed to load job metadata from {job_metadata_path}: {e}")
        return None

def wait_for_vertex_ai_output(bucket, raw_output_prefix, max_wait_seconds=600): # Increased wait time
    start_time = time.time()
    check_interval = 20 # Check less frequently
    
    logger.info(f"Waiting for Vertex AI output in prefix: gs://{bucket.name}/{raw_output_prefix}")
    
    while time.time() - start_time < max_wait_seconds:
        blobs = list(bucket.list_blobs(prefix=raw_output_prefix))
        
        # Vertex AI often creates a subdirectory. Find files with 'prediction.results' in their name.
        prediction_files = [b for b in blobs if 'prediction.results' in b.name]
        
        if prediction_files:
            logger.info(f"Found {len(prediction_files)} prediction files after {int(time.time() - start_time)} seconds.")
            return prediction_files
        
        logger.info(f"No prediction files found yet. Waiting {check_interval} seconds... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(check_interval)
    
    logger.error(f"Timeout after {max_wait_seconds} seconds waiting for prediction files.")
    return []

def process_vertex_ai_output(prediction_files, bucket, job_id, run_date):
    all_predictions = []
    
    for pred_file in prediction_files:
        logger.info(f"Processing prediction file: {pred_file.name}")
        content = pred_file.download_as_string().decode('utf-8')
        
        for line in content.strip().split('\n'):
            if not line.strip(): continue
            
            try:
                output_data = json.loads(line)
                instance = output_data.get('instance', {})
                predictions = output_data.get('prediction', [])
                
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
                logger.error(f"Error processing line: '{line}'. Error: {e}", exc_info=True)
                continue
    
    processed_output_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_processed_output']}/{FILE_NAMES['batch_predictions']}"
    jsonl_content = '\n'.join([json.dumps(pred) for pred in all_predictions])
    bucket.blob(processed_output_path).upload_from_string(jsonl_content, content_type='application/jsonl')
    logger.info(f"Saved {len(all_predictions)} processed predictions to: {processed_output_path}")

    summary_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_processed_output']}/{FILE_NAMES['job_summary']}"
    summary = {
        'total_predictions': len(all_predictions),
        'fire_detections': sum(1 for p in all_predictions if p['detected']),
        'average_confidence': sum(p['confidence'] for p in all_predictions) / len(all_predictions) if all_predictions else 0,
        'processed_at': datetime.utcnow().isoformat() + 'Z'
    }
    bucket.blob(summary_path).upload_from_string(json.dumps(summary, indent=2))
    
    return processed_output_path, all_predictions

def result_processor_cloud_function(event, context):
    logger.info("Result Processor function triggered.")
    
    run_date = datetime.utcnow().strftime('%Y-%m-%d')
    job_id = None
    
    if event and isinstance(event, dict):
        # Pub/Sub triggers have a 'data' key with a base64 encoded string
        if 'data' in event:
            try:
                message_data = base64.b64decode(event['data']).decode('utf-8')
                message_json = json.loads(message_data)
                job_id = message_json.get('job_id') # Attempt to get job_id from message
                logger.info(f"Triggered via Pub/Sub, extracted job_id: {job_id}")
            except Exception as e:
                logger.warning(f"Could not parse job_id from Pub/Sub message: {e}")

    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    if not job_id:
        manifest_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{FILE_NAMES['job_manifest']}"
        logger.info(f"job_id not in trigger, loading from manifest: {manifest_path}")
        try:
            manifest_blob = bucket.blob(manifest_path)
            manifest = json.loads(manifest_blob.download_as_string())
            if manifest.get("jobs"):
                latest_job = sorted(manifest["jobs"], key=lambda x: x["created_at"], reverse=True)[0]
                job_id = latest_job["job_id"]
                logger.info(f"Using latest job_id from manifest: {job_id}")
            else:
                logger.error(f"No jobs found in manifest for date {run_date}")
                return
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}", exc_info=True)
            return
    
    logger.info(f"Processing results for run_date: {run_date}, job_id: {job_id}")

    job_metadata_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{FILE_NAMES['job_metadata']}"
    job_metadata = get_vertex_ai_job_info(job_metadata_path)
    if not job_metadata:
        logger.error("Could not load job metadata, exiting.")
        return
    
    raw_output_prefix = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_raw_output']}/"
    prediction_files = wait_for_vertex_ai_output(bucket, raw_output_prefix)
    if not prediction_files:
        logger.error("No prediction files found after waiting, exiting.")
        return
    
    _, predictions = process_vertex_ai_output(prediction_files, bucket, job_id, run_date)
    if not predictions:
        logger.error("No predictions were processed from the output files, exiting.")
        return

    try:
        incidents_path = f"{GCS_PATHS['incidents']}/{run_date}/{FILE_NAMES['incident_data']}"
        incidents_content = bucket.blob(incidents_path).download_as_string().decode('utf-8')
        hotspots_by_cluster = {inc['cluster_id']: inc['hotspots'] for inc in [json.loads(line) for line in incidents_content.strip().split('\n')]}
        
        batch_input_path = job_metadata['input']['input_path'].replace(f"gs://{GCS_BUCKET_NAME}/", "")
        input_content = bucket.blob(batch_input_path).download_as_string().decode('utf-8')
        input_metadata = {cluster['cluster_id']: cluster for instance in [json.loads(line) for line in input_content.strip().split('\n')] for cluster in instance.get('clusters', [])}
    except Exception as e:
        logger.error(f"Failed to load supporting data (incidents or inputs): {e}", exc_info=True)
        return

    visualizer = MapVisualizer()
    visualization_data = []
    
    for pred in predictions:
        cluster_id = pred['cluster_id']
        cluster_metadata = input_metadata.get(cluster_id)
        if not cluster_metadata:
            logger.warning(f"No metadata found for cluster {cluster_id}")
            continue
        
        try:
            image_uri = cluster_metadata['gcs_image_uri']
            img_bucket_name, img_blob_name = image_uri.replace("gs://", "").split("/", 1)
            image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
            
            hotspots = hotspots_by_cluster.get(cluster_id, [])
            firms_df = pd.DataFrame([h['properties'] for h in hotspots]) if hotspots else pd.DataFrame()
            
            map_image = visualizer.generate_fire_map(
                base_image_bytes=image_bytes, image_bbox=cluster_metadata['image_bbox'],
                ai_detections=[pred], firms_hotspots_df=firms_df, acquisition_date_str=run_date
            )
            
            report_images_path = f"{GCS_PATHS['reports']}/{run_date}/{GCS_PATHS['report_images']}"
            image_path = f"{report_images_path}/{cluster_id}.png"
            img_byte_arr = BytesIO()
            map_image.save(img_byte_arr, format='PNG')
            bucket.blob(image_path).upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
            
            visualization_data.append({
                'cluster_id': cluster_id, 'image_path': f"gs://{GCS_BUCKET_NAME}/{image_path}",
                'latitude': (cluster_metadata['image_bbox'][1] + cluster_metadata['image_bbox'][3]) / 2,
                'longitude': (cluster_metadata['image_bbox'][0] + cluster_metadata['image_bbox'][2]) / 2,
                'detected': pred['detected'], 'confidence': pred['confidence']
            })
            logger.info(f"Generated visualization for cluster {cluster_id}")
        except Exception as e:
            logger.error(f"Failed to generate visualization for {cluster_id}: {e}", exc_info=True)

    if visualization_data:
        report_filename = f"wildfire_report_{run_date}_{job_id}.html"
        report_path = f"{GCS_PATHS['reports']}/{run_date}/{report_filename}"
        
        m = folium.Map(location=[-2.5, 118], zoom_start=5)
        for viz in visualization_data:
            try:
                image_blob = bucket.blob(viz['image_path'].replace(f"gs://{GCS_BUCKET_NAME}/", ""))
                encoded_image = base64.b64encode(image_blob.download_as_bytes()).decode('utf-8')
                
                popup_html = f"""<h4>Cluster ID: {viz['cluster_id']}</h4>
                                 <p><b>AI Detection:</b> {'FIRE DETECTED' if viz['detected'] else 'No Fire Detected'}<br>
                                 <b>Confidence:</b> {viz['confidence']:.2%}</p>
                                 <img src='data:image/png;base64,{encoded_image}' width='400'>"""
                
                iframe = folium.IFrame(popup_html, width=430, height=450)
                popup = folium.Popup(iframe, max_width=430)
                marker_color = 'red' if viz['detected'] else 'green'
                
                folium.Marker(
                    location=[viz['latitude'], viz['longitude']], popup=popup, tooltip=viz['cluster_id'],
                    icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
                ).add_to(m)
            except Exception as e:
                logger.error(f"Failed to add marker for {viz['cluster_id']}: {e}", exc_info=True)
        
        m.save(f"/tmp/{report_filename}")
        bucket.blob(report_path).upload_from_filename(f"/tmp/{report_filename}", content_type='text/html')
        logger.info(f"Report saved to: gs://{GCS_BUCKET_NAME}/{report_path}")
        
        job_metadata['status'] = 'completed'
        job_metadata['completed_at'] = datetime.utcnow().isoformat() + 'Z'
        job_metadata['report_path'] = f"gs://{GCS_BUCKET_NAME}/{report_path}"
        bucket.blob(job_metadata_path).upload_from_string(json.dumps(job_metadata, indent=2))
        
    else:
        logger.error("No visualizations generated, so no report was created.")

    logger.info("Result Processor function finished successfully.")
    return
