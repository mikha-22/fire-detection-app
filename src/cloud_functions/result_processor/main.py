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
RETRY_ATTEMPTS = 10
RETRY_DELAY_SECONDS = 60

# --- Initializations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

storage_client = storage.Client()

try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logger.info("Vertex AI client initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize Vertex AI client. Error: {e}")

def result_processor_cloud_function(event, context):
    logger.info("Result Processor function triggered.")

    if not all([GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION]):
        logger.critical("Missing required environment variables. Exiting.")
        return

    run_date = datetime.utcnow().strftime('%Y-%m-%d')
    logger.info(f"Processing results for run_date determined by system clock: {run_date}")

    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # --- Step 1: Load original incidents and batch inputs ---
    try:
        incidents_blob_path = f"incidents/{run_date}/detected_incidents.jsonl"
        logger.info(f"Attempting to download original incidents from: {incidents_blob_path}")
        incidents_content = bucket.blob(incidents_blob_path).download_as_string().decode('utf-8')
        incidents_data = [json.loads(line) for line in incidents_content.strip().split('\n')]
        hotspots_by_cluster = {incident['cluster_id']: incident['hotspots'] for incident in incidents_data}

        master_input_blob_path = f"incident_inputs/{run_date}.jsonl"
        logger.info(f"Attempting to download batch inputs from: {master_input_blob_path}")
        input_content = bucket.blob(master_input_blob_path).download_as_string().decode('utf-8')
        
        input_metadata = {}
        for line in input_content.strip().split('\n'):
            instance = json.loads(line)
            for cluster in instance.get('clusters', []):
                input_metadata[cluster['cluster_id']] = cluster
                
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load prerequisite data (incidents or inputs). Error: {e}", exc_info=True)
        return

    # --- Step 2: Find and load AI prediction results with retries ---
    gcs_output_prefix = f"incident_outputs/{run_date}/"
    prediction_blobs = []
    for attempt in range(RETRY_ATTEMPTS):
        logger.info(f"Checking for prediction files in '{gcs_output_prefix}'... (Attempt {attempt + 1}/{RETRY_ATTEMPTS})")
        prediction_blobs = [b for b in bucket.list_blobs(prefix=gcs_output_prefix) if 'prediction.results' in b.name]
        if prediction_blobs:
            logger.info(f"Found {len(prediction_blobs)} prediction result file(s).")
            break
        if attempt < RETRY_ATTEMPTS - 1:
            logger.warning(f"No prediction files found. Retrying in {RETRY_DELAY_SECONDS} seconds.")
            time.sleep(RETRY_DELAY_SECONDS)

    if not prediction_blobs:
        logger.error(f"No prediction result files found at prefix: {gcs_output_prefix} after {RETRY_ATTEMPTS} attempts. Exiting.")
        return

    # --- Step 3: Process results and generate visualizations ---
    folium_map_data = []
    for prediction_blob in prediction_blobs:
        logger.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        for line in prediction_content_str.strip().split('\n'):
            if not line.strip(): continue

            try:
                full_output_line = json.loads(line)
                prediction = full_output_line.get('prediction', {})
                instance_data = full_output_line.get('instance', {})
                clusters_in_instance = instance_data.get('clusters', [])
                
                if not clusters_in_instance:
                    logger.warning(f"No clusters found in instance: {instance_data}")
                    continue
                
                cluster_data = clusters_in_instance[0]
                cluster_id = cluster_data.get('cluster_id')
                
                if not cluster_id:
                    logger.warning(f"No cluster_id found in cluster data: {cluster_data}")
                    continue
                    
                # --- AGREED FIX: Add robustness check for input metadata ---
                input_data = input_metadata.get(cluster_id)
                if not input_data:
                    logger.warning(f"Could not find matching input metadata for cluster_id '{cluster_id}'. Skipping.")
                    continue

                original_image_uri = input_data['gcs_image_uri']
                image_bbox = input_data['image_bbox']
                
                img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()

                cluster_hotspots = hotspots_by_cluster.get(cluster_id, [])
                firms_df = pd.DataFrame()
                if cluster_hotspots:
                    hotspot_records = [h['properties'] for h in cluster_hotspots]
                    firms_df = pd.DataFrame.from_records(hotspot_records)
                
                visualizer = MapVisualizer()
                final_map_image = visualizer.generate_fire_map(
                    base_image_bytes=image_bytes, 
                    image_bbox=image_bbox, 
                    ai_detections=[prediction],
                    firms_hotspots_df=firms_df,
                    acquisition_date_str=run_date
                )
                
                img_byte_arr = BytesIO()
                final_map_image.save(img_byte_arr, format='PNG')
                encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                folium_map_data.append({
                    "cluster_id": cluster_id,
                    "latitude": (image_bbox[1] + image_bbox[3]) / 2,
                    "longitude": (image_bbox[0] + image_bbox[2]) / 2,
                    "detected": prediction.get("detected"),
                    "confidence": prediction.get("confidence", 0),
                    "encoded_png": encoded_image
                })

            except Exception as e:
                logger.error(f"Failed to process a prediction line: '{line}'. Error: {e}", exc_info=True)

    # --- Step 4: Generate Final Interactive Report ---
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

        report_filename = f"daily_interactive_report_{run_date}.html"
        local_temp_path = f"/tmp/{report_filename}"
        m.save(local_temp_path)
        
        report_blob_path = f"{FINAL_OUTPUT_GCS_PREFIX}{run_date}/{report_filename}"
        bucket.blob(report_blob_path).upload_from_filename(local_temp_path, content_type='text/html')
        
        logger.info(f"Successfully generated and uploaded interactive report to: gs://{GCS_BUCKET_NAME}/{report_blob_path}")
    else:
        logger.error(f"No data was successfully processed to generate a Folium map.")

    logger.info("Result Processor function finished successfully.")
