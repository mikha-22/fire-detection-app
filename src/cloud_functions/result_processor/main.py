# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
from io import BytesIO

from google.cloud import storage
from PIL import Image
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "fire-app-bucket")
FINAL_OUTPUT_GCS_PREFIX = "final_outputs/"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
storage_client = storage.Client()

def result_processor_cloud_function(event, context):
    """
    Triggered by a Pub/Sub message from a Cloud Logging sink when a Vertex AI
    Batch Prediction job completes. This function processes a results file that may
    contain MULTIPLE predictions.
    """
    logging.info("Result Processor function triggered.")

    try:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)
        
        # --- FINAL FIX: The job ID is in resource.labels for this log type ---
        job_id = log_entry['resource']['labels']['job_id']
        # The run_date is the display name of the job minus the prefix
        run_date = log_entry['protoPayload']['displayName'].replace('batch_prediction_', '')
        
        gcs_output_prefix = f"incident_outputs/{run_date}/"
        logging.info(f"Processing prediction results for run_date '{run_date}' from prefix: {gcs_output_prefix}")

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Could not parse trigger event payload or find GCS path. Error: {e}", extra={"payload": event.get('data')})
        return

    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
    
    if not prediction_blobs:
        logging.error(f"No prediction result files found at prefix: {gcs_output_prefix}")
        return

    # Load the master input file to get metadata for all clusters in this run
    master_input_blob_path = f"incident_inputs/{run_date}.jsonl"
    try:
        all_inputs_str = bucket.blob(master_input_blob_path).download_as_string().decode('utf-8')
        input_metadata = {json.loads(line)['instance_id']: json.loads(line) for line in all_inputs_str.strip().split('\n')}
    except Exception as e:
        logging.error(f"Could not read or parse the master input file: {master_input_blob_path}. Error: {e}")
        return

    for prediction_blob in prediction_blobs:
        if 'prediction' not in prediction_blob.name:
            continue
            
        logging.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        for line in prediction_content_str.strip().split('\n'):
            try:
                prediction_data = json.loads(line)
                ai_detections = prediction_data.get('predictions')
                if not ai_detections: continue

                prediction = ai_detections[0]
                cluster_id = prediction.get("instance_id")
                
                input_data = input_metadata.get(cluster_id)
                if not input_data:
                    logging.error(f"Could not find input metadata for {cluster_id}")
                    continue

                original_image_uri = input_data['gcs_image_uri']
                image_bbox = input_data['image_bbox']
                
                img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
                
                visualizer = MapVisualizer()
                acquisition_date_str = run_date
                
                final_map_image = visualizer.generate_fire_map(
                    base_image_bytes=image_bytes, image_bbox=image_bbox, ai_detections=ai_detections, acquisition_date_str=acquisition_date_str
                )
                
                map_output_path = f"{FINAL_OUTPUT_GCS_PREFIX}{run_date}/{cluster_id}_map.png"
                map_blob = bucket.blob(map_output_path)
                
                img_byte_arr = BytesIO()
                final_map_image.save(img_byte_arr, format='PNG')
                map_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
                
                logging.info(f"Successfully generated map for {cluster_id} in run {run_date}.")

            except Exception as e:
                logging.error(f"Failed to process a prediction line: '{line}'. Error: {e}", exc_info=True)

    logging.info("Result Processor function finished successfully.")
