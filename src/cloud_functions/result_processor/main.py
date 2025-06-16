# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
import time
from io import BytesIO
from datetime import datetime

import pandas as pd
from google.cloud import storage, aiplatform
from PIL import Image
import folium
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
FINAL_OUTPUT_GCS_PREFIX = "final_outputs/"

# --- Initializations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

storage_client = storage.Client()

try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logger.info("Vertex AI client initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize Vertex AI client. Error: {e}")

def result_processor_cloud_function(request=None, context=None):
    """
    Process results from Vertex AI batch prediction job.
    Can be triggered by HTTP request or Pub/Sub message.
    """
    logger.info("Result Processor function triggered.")
    
    # Handle both HTTP and Pub/Sub triggers
    if request and hasattr(request, 'get_json'):
        # HTTP trigger - useful for manual testing
        logger.info("Triggered via HTTP request")
        request_json = request.get_json(silent=True) or {}
        run_date = request_json.get('run_date', datetime.utcnow().strftime('%Y-%m-%d'))
        job_id = request_json.get('job_id')  # Can specify a specific job
    else:
        # Pub/Sub trigger
        logger.info("Triggered via Pub/Sub")
        run_date = datetime.utcnow().strftime('%Y-%m-%d')
        job_id = None

    if not all([GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION]):
        logger.critical("Missing required environment variables. Exiting.")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": "Missing environment variables"}, 500
        return

    logger.info(f"Processing results for run_date: {run_date}")

    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # --- Step 1: Determine which job to process ---
    if not job_id:
        # If no job_id specified, find the latest completed job
        manifest_path = f"incident_outputs/{run_date}/manifest.json"
        try:
            manifest_blob = bucket.blob(manifest_path)
            manifest = json.loads(manifest_blob.download_as_string())
            
            # Find the most recent job
            if manifest.get("jobs"):
                # Sort by created_at and get the latest
                latest_job = sorted(manifest["jobs"], key=lambda x: x["created_at"], reverse=True)[0]
                job_id = latest_job["job_id"]
                logger.info(f"Using latest job_id from manifest: {job_id}")
            else:
                logger.error(f"No jobs found in manifest for date {run_date}")
                if request and hasattr(request, 'get_json'):
                    return {"status": "error", "message": "No jobs found for this date"}, 404
                return
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            if request and hasattr(request, 'get_json'):
                return {"status": "error", "message": f"Failed to load manifest: {str(e)}"}, 500
            return
    
    logger.info(f"Processing results for run_date: {run_date}, job_id: {job_id}")

    # --- Step 2: Load original incidents and batch inputs ---
    try:
        incidents_blob_path = f"incidents/{run_date}/detected_incidents.jsonl"
        logger.info(f"Attempting to download original incidents from: {incidents_blob_path}")
        incidents_content = bucket.blob(incidents_blob_path).download_as_string().decode('utf-8')
        incidents_data = [json.loads(line) for line in incidents_content.strip().split('\n')]
        hotspots_by_cluster = {incident['cluster_id']: incident['hotspots'] for incident in incidents_data}

        master_input_blob_path = f"incident_inputs/{run_date}/{job_id}/input.jsonl"
        logger.info(f"Attempting to download batch inputs from: {master_input_blob_path}")
        input_content = bucket.blob(master_input_blob_path).download_as_string().decode('utf-8')
        
        input_metadata = {}
        for line in input_content.strip().split('\n'):
            instance = json.loads(line)
            for cluster in instance.get('clusters', []):
                input_metadata[cluster['cluster_id']] = cluster
                
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load prerequisite data (incidents or inputs). Error: {e}", exc_info=True)
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": f"Failed to load data: {str(e)}"}, 500
        return

    # --- Step 3: Find and load AI prediction results ---
    gcs_output_prefix = f"incident_outputs/{run_date}/{job_id}/"
    
    logger.info(f"Checking for prediction files in '{gcs_output_prefix}'...")
    
    # List ALL files in the output directory
    all_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
    logger.info(f"Found {len(all_blobs)} total files in job output directory")
    
    # Vertex AI might create a subdirectory, so look for prediction.results files
    prediction_blobs = [b for b in all_blobs if 'prediction.results' in b.name or 
                       (b.name.endswith('.jsonl') and 'prediction' in b.name)]
    
    if not prediction_blobs:
        logger.error(f"No prediction result files found for job {job_id}. Files found: {[b.name for b in all_blobs]}")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": f"No prediction files found for job {job_id}"}, 404
        return
    
    logger.info(f"Found {len(prediction_blobs)} prediction result file(s): {[b.name for b in prediction_blobs]}")

    # --- Step 4: Process results and generate visualizations ---
    folium_map_data = []
    for prediction_blob in prediction_blobs:
        logger.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        for line in prediction_content_str.strip().split('\n'):
            if not line.strip(): continue

            try:
                full_output_line = json.loads(line)
                
                # Vertex AI batch prediction format: {"instance": {...}, "prediction": [...]}
                instance_data = full_output_line.get('instance', {})
                prediction_data = full_output_line.get('prediction', [])
                
                # Log the structure to understand it better
                logger.debug(f"Parsed line structure - keys: {list(full_output_line.keys())}")
                logger.debug(f"Instance data: {instance_data}")
                logger.debug(f"Prediction data: {prediction_data}")
                
                # The prediction should be a list from our CPR model
                if not isinstance(prediction_data, list) or not prediction_data:
                    logger.warning(f"Unexpected prediction format. Expected list, got: {type(prediction_data)}")
                    continue
                
                # Extract instance_id from the instance data
                instance_id = instance_data.get('instance_id')
                
                if not instance_id:
                    logger.warning(f"No instance_id found in instance data: {instance_data}")
                    continue
                
                # Our CPR model returns a list of predictions
                # Process each prediction in the list
                for prediction in prediction_data:
                    # The cluster_id might be in the prediction or we use the instance_id
                    cluster_id = prediction.get('instance_id', instance_id)
                    
                    # Get the input metadata for this cluster
                    input_data = input_metadata.get(cluster_id)
                    if not input_data:
                        logger.warning(f"Could not find matching input metadata for cluster_id '{cluster_id}'. Skipping.")
                        continue

                    original_image_uri = input_data['gcs_image_uri']
                    image_bbox = input_data['image_bbox']
                    
                    # Download the original satellite image
                    try:
                        img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                        image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
                    except Exception as e:
                        logger.error(f"Failed to download image for cluster {cluster_id}: {e}")
                        continue

                    # Get FIRMS hotspots for this cluster
                    cluster_hotspots = hotspots_by_cluster.get(cluster_id, [])
                    firms_df = pd.DataFrame()
                    if cluster_hotspots:
                        hotspot_records = [h['properties'] for h in cluster_hotspots]
                        firms_df = pd.DataFrame.from_records(hotspot_records)
                    
                    # Generate the visualization
                    try:
                        visualizer = MapVisualizer()
                        final_map_image = visualizer.generate_fire_map(
                            base_image_bytes=image_bytes, 
                            image_bbox=image_bbox, 
                            ai_detections=[prediction],
                            firms_hotspots_df=firms_df,
                            acquisition_date_str=run_date
                        )
                        
                        # Convert to base64 for embedding in HTML
                        img_byte_arr = BytesIO()
                        final_map_image.save(img_byte_arr, format='PNG')
                        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                        folium_map_data.append({
                            "cluster_id": cluster_id,
                            "latitude": (image_bbox[1] + image_bbox[3]) / 2,
                            "longitude": (image_bbox[0] + image_bbox[2]) / 2,
                            "detected": prediction.get("detected", False),
                            "confidence": prediction.get("confidence", 0),
                            "encoded_png": encoded_image
                        })
                        
                        logger.info(f"Successfully processed cluster {cluster_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate visualization for cluster {cluster_id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Failed to process a prediction line: '{line[:100]}...'. Error: {e}", exc_info=True)

    # --- Step 5: Generate Final Interactive Report ---
    if folium_map_data:
        logger.info(f"Generating final interactive report for {len(folium_map_data)} clusters.")
        m = folium.Map(location=[-2.5, 118], zoom_start=5)

        for item in folium_map_data:
            image_uri = f"data:image/png;base64,{item['encoded_png']}"
            popup_html = f"""
            <h4>Cluster ID: {item['cluster_id']}</h4>
            <p>
              <b>AI Detection:</b> {'FIRE DETECTED' if item['detected'] else 'No Fire Detected'}<br>
              <b>Confidence:</b> {item['confidence']:.2%}
            </p>
            <img src='{image_uri}' width='400'>
            """
            iframe = folium.IFrame(popup_html, width=430, height=450)
            popup = folium.Popup(iframe, max_width=430)
            marker_color = 'red' if item['detected'] else 'green'
            
            folium.Marker(
                location=[item['latitude'], item['longitude']],
                popup=popup,
                tooltip=item['cluster_id'],
                icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
            ).add_to(m)

        report_filename = f"wildfire_report_{run_date}_{job_id}.html"
        local_temp_path = f"/tmp/{report_filename}"
        m.save(local_temp_path)
        
        report_blob_path = f"{FINAL_OUTPUT_GCS_PREFIX}{run_date}/{report_filename}"
        bucket.blob(report_blob_path).upload_from_filename(local_temp_path, content_type='text/html')
        
        logger.info(f"Successfully generated and uploaded interactive report to: gs://{GCS_BUCKET_NAME}/{report_blob_path}")
        
        # Update job metadata to mark as completed
        try:
            metadata_path = f"incident_outputs/{run_date}/{job_id}/metadata.json"
            metadata_blob = bucket.blob(metadata_path)
            try:
                metadata = json.loads(metadata_blob.download_as_string())
            except:
                metadata = {"job_id": job_id, "run_date": run_date}
                
            metadata["status"] = "completed"
            metadata["completed_at"] = datetime.utcnow().isoformat() + "Z"
            metadata["report_url"] = f"gs://{GCS_BUCKET_NAME}/{report_blob_path}"
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.warning(f"Failed to update job metadata: {e}")
    else:
        logger.error("No data was successfully processed to generate a Folium map.")
        if request and hasattr(request, 'get_json'):
            return {"status": "error", "message": "No data processed"}, 500

    logger.info("Result Processor function finished successfully.")
    
    # Return appropriate response for HTTP trigger
    if request and hasattr(request, 'get_json'):
        if folium_map_data:
            return {"status": "success", "report_url": f"gs://{GCS_BUCKET_NAME}/{report_blob_path}"}, 200
        else:
            return {"status": "error", "message": "No data processed"}, 500
    return
