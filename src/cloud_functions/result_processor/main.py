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
# Added more detailed logging for clarity
log_format = '%(asctime)s - %(levelname)s - [%(component)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

storage_client = storage.Client()

try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logger.info("Vertex AI client initialized.", extra={'component': 'ResultProcessorInit'})
except Exception as e:
    logger.critical(f"Failed to initialize Vertex AI client. Error: {e}", extra={'component': 'ResultProcessorInit'})

def log_with_component(message, severity="INFO", **kwargs):
    """Helper function for structured logging."""
    extra_info = {'component': 'ResultProcessor', **kwargs}
    if severity.upper() == "INFO":
        logger.info(message, extra=extra_info)
    elif severity.upper() == "ERROR":
        logger.error(message, extra=extra_info)
    elif severity.upper() == "WARNING":
        logger.warning(message, extra=extra_info)
    else:
        logger.debug(message, extra=extra_info)

def result_processor_cloud_function(event, context):
    trigger_event_id = context.event_id
    log_with_component("Result Processor function triggered.", trigger_event_id=trigger_event_id)

    if not all([GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION]):
        log_with_component("Missing required environment variables. Exiting.", "CRITICAL")
        return

    run_date = datetime.utcnow().strftime('%Y-%m-%d')
    log_with_component(
        "Processing results based on current system date.",
        run_date=run_date,
        execution_id=trigger_event_id
    )

    # --- Step 1: Load Original Incident Data ---
    log_with_component("Starting Step 1: Loading Original Incident Data.", execution_id=trigger_event_id)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    incidents_blob_path = f"incidents/{run_date}/detected_incidents.jsonl"
    log_with_component("Attempting to download file from GCS.", gcs_path=f"gs://{GCS_BUCKET_NAME}/{incidents_blob_path}")
    try:
        incidents_content = bucket.blob(incidents_blob_path).download_as_string()
        log_with_component("File download successful.", gcs_path=f"gs://{GCS_BUCKET_NAME}/{incidents_blob_path}", bytes_downloaded=len(incidents_content))
        incidents_data = [json.loads(line) for line in incidents_content.decode('utf-8').strip().split('\n')]
        hotspots_by_cluster = {incident['cluster_id']: incident['hotspots'] for incident in incidents_data}
        log_with_component("Step 1 Complete: Loaded original incidents.", incident_count=len(incidents_data), execution_id=trigger_event_id)
    except Exception as e:
        log_with_component(f"Could not read or parse the original incidents file: {incidents_blob_path}. Error: {e}", "ERROR", exc_info=True)
        hotspots_by_cluster = {}

    # --- Step 2: Load Batch Prediction Input Data ---
    log_with_component("Starting Step 2: Loading Batch Prediction Input Data.", execution_id=trigger_event_id)
    master_input_blob_path = f"incident_inputs/{run_date}.jsonl"
    log_with_component("Attempting to download file from GCS.", gcs_path=f"gs://{GCS_BUCKET_NAME}/{master_input_blob_path}")
    try:
        input_instance_str = bucket.blob(master_input_blob_path).download_as_string()
        log_with_component("File download successful.", gcs_path=f"gs://{GCS_BUCKET_NAME}/{master_input_blob_path}", bytes_downloaded=len(input_instance_str))
        input_instance = json.loads(input_instance_str)
        input_metadata = {cluster['cluster_id']: cluster for cluster in input_instance['clusters']}
        log_with_component("Step 2 Complete: Loaded batch prediction inputs.", input_count=len(input_metadata), execution_id=trigger_event_id)
    except Exception as e:
        log_with_component(f"CRITICAL: Could not read or parse the master input file: {master_input_blob_path}. Cannot proceed. Error: {e}", "ERROR", exc_info=True)
        return

    # --- Step 3: Find and Load AI Prediction Results ---
    log_with_component("Starting Step 3: Loading AI Prediction Results.", execution_id=trigger_event_id)
    gcs_output_prefix = f"incident_outputs/{run_date}/"
    prediction_blobs = []
    for attempt in range(RETRY_ATTEMPTS):
        log_with_component(f"Checking for prediction files in '{gcs_output_prefix}'... (Attempt {attempt + 1}/{RETRY_ATTEMPTS})")
        prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
        if prediction_blobs:
            log_with_component(f"Found {len(prediction_blobs)} blob(s) in prefix.")
            break
        if attempt < RETRY_ATTEMPTS - 1:
            log_with_component(f"No prediction files found. Retrying in {RETRY_DELAY_SECONDS} seconds.", "WARNING")
            time.sleep(RETRY_DELAY_SECONDS)

    if not prediction_blobs:
        log_with_component(f"No prediction result files found at prefix: {gcs_output_prefix} after {RETRY_ATTEMPTS} attempts. Exiting.", "ERROR")
        return

    # --- Step 4: Process Results and Generate Visualizations ---
    log_with_component("Starting Step 4: Processing results and generating maps.", execution_id=trigger_event_id)
    folium_map_data = []
    for prediction_blob in prediction_blobs:
        if 'prediction.results' not in prediction_blob.name:
            continue
        
        log_with_component("Processing prediction result file.", gcs_path=f"gs://{GCS_BUCKET_NAME}/{prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        for line in prediction_content_str.strip().split('\n'):
            if not line.strip(): continue

            try:
                # --- THE FINAL FIX IS HERE ---
                # Vertex AI wraps the model's output in a standard format.
                # We need to parse this wrapper to get to our predictions list.
                full_output_line = json.loads(line)
                prediction_wrapper = full_output_line.get('prediction', {})
                ai_detections_list = prediction_wrapper.get('predictions')
                
                if not ai_detections_list:
                    log_with_component("Skipping line, no 'predictions' key found inside the 'prediction' wrapper.", "WARNING", line_content=line)
                    continue

                for prediction in ai_detections_list:
                    cluster_id = prediction.get("instance_id")
                    if not cluster_id:
                        log_with_component("Skipping prediction with no instance_id.", "WARNING", prediction_data=prediction)
                        continue
                    
                    input_data = input_metadata.get(cluster_id)
                    if not input_data:
                        log_with_component(f"Could not find input metadata for cluster_id '{cluster_id}'", "ERROR")
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
                log_with_component(f"Failed to process a prediction line: '{line}'. Error: {e}", "ERROR", exc_info=True)

    # --- Step 5: Generate Final Interactive Report ---
    if folium_map_data:
        log_with_component(f"Starting Step 5: Generating final interactive report for {len(folium_map_data)} clusters.", execution_id=trigger_event_id)
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
        
        log_with_component(f"Successfully generated and uploaded interactive report to: gs://{GCS_BUCKET_NAME}/{report_blob_path}")
    else:
        log_with_component("No data was processed to generate a Folium map. This could be because the prediction files were empty or could not be parsed.", "ERROR", search_prefix=gcs_output_prefix, execution_id=trigger_event_id)

    log_with_component("Result Processor function finished successfully.", execution_id=trigger_event_id)
