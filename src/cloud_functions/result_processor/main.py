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
    Triggered by a Pub/Sub message when a Vertex AI Batch Prediction job completes.
    This function now processes a results file that may contain MULTIPLE predictions.
    1. Fetches the AI prediction results file (JSONL format).
    2. Loops through each prediction in the file.
    3. For each prediction, retrieves the original image and metadata.
    4. Invokes the MapVisualizer to create a final map.
    5. Saves the final map and metadata to GCS.
    """
    logging.info("Result Processor function triggered.")

    try:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)
        
        # --- MODIFIED: The log entry structure for ml_job is different ---
        # We get the GCS output path directly from the resource labels in the log.
        gcs_output_uri = log_entry['resource']['labels']['job_id'].split('/')[-1]
        gcs_output_prefix = f"incident_outputs/{gcs_output_uri}/"
        logging.info(f"Processing prediction results from prefix: {gcs_output_prefix}")

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Could not parse trigger event payload or find GCS path. Error: {e}", extra={"payload": event.get('data')})
        return

    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
    
    if not prediction_blobs:
        logging.error(f"No prediction result file found at prefix: {gcs_output_prefix}")
        return

    # --- MODIFIED LOGIC: Process each prediction file (usually one) ---
    for prediction_blob in prediction_blobs:
        if 'prediction' not in prediction_blob.name:
            continue
            
        logging.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        # A JSONL file has one JSON object per line
        prediction_lines = prediction_content_str.strip().split('\n')

        # --- MODIFIED LOGIC: Loop through each prediction (each line) ---
        for line in prediction_lines:
            try:
                # The custom predictor wraps its output, so we unwrap it here.
                prediction_data = json.loads(line)
                ai_detections = prediction_data.get('predictions')
                if not ai_detections:
                    logging.warning(f"Skipping malformed prediction line: {line}")
                    continue

                prediction = ai_detections[0] # There's one prediction per instance
                cluster_id = prediction.get("instance_id")
                logging.info(f"Processing parsed prediction for cluster: {cluster_id}")
                
                # --- The rest of the logic is per-cluster and remains the same ---
                input_data_blob_path = f"incident_inputs/{cluster_id}.jsonl" # This needs to be adjusted
                
                # Find the master batch file that contains this cluster_id
                batch_job_id = gcs_output_prefix.split('/')[1]
                master_input_blob_path = f"incident_inputs/{batch_job_id}.jsonl"
                master_input_blob = bucket.blob(master_input_blob_path)
                
                input_data = None
                if master_input_blob.exists():
                    all_inputs_str = master_input_blob.download_as_string().decode('utf-8')
                    for input_line in all_inputs_str.strip().split('\n'):
                        line_data = json.loads(input_line)
                        if line_data.get("instance_id") == cluster_id:
                            input_data = line_data
                            break

                if not input_data:
                    logging.error(f"Could not find input metadata for {cluster_id} in {master_input_blob_path}")
                    continue

                original_image_uri = input_data['gcs_image_uri']
                image_bbox = input_data['image_bbox']
                
                img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
                
                visualizer = MapVisualizer()
                acquisition_date = cluster_id.split('_')[2]
                
                final_map_image = visualizer.generate_fire_map(
                    base_image_bytes=image_bytes,
                    image_bbox=image_bbox,
                    ai_detections=ai_detections,
                    firms_hotspots_df=None,
                    acquisition_date_str=acquisition_date
                )
                
                map_output_path = f"{FINAL_OUTPUT_GCS_PREFIX}{cluster_id}_map.png"
                map_blob = bucket.blob(map_output_path)
                
                img_byte_arr = BytesIO()
                final_map_image.save(img_byte_arr, format='PNG')
                map_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
                
                logging.info(f"Successfully generated and saved map for {cluster_id} to gs://{GCS_BUCKET_NAME}/{map_output_path}")

            except Exception as e:
                logging.error(f"Failed to process a prediction line: '{line}'. Error: {e}", exc_info=True)

    logging.info("Result Processor function finished successfully.")
