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
    logging.info("Result Processor function triggered.")

    try:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)
        
        # --- MODIFIED: Extract the output path directly. It contains the run_date. ---
        gcs_output_uri = log_entry['protoPayload']['metadata']['batchPredictionJob']['outputInfo']['gcsOutputDirectory']
        logging.info(f"Processing prediction results from: {gcs_output_uri}")

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Could not parse trigger event payload or find GCS path. Error: {e}", exc_info=True)
        return

    bucket_name, gcs_output_prefix = gcs_output_uri.strip('/').split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
    
    if not prediction_blobs:
        logging.error(f"No prediction result files found at prefix: {gcs_output_prefix}")
        return

    # --- MODIFIED: Reconstruct paths based on the run_date from the output folder ---
    run_date = gcs_output_prefix.strip('/').split('/')[-1]
    master_input_blob_path = f"incident_inputs/{run_date}.jsonl"
    all_inputs_str = bucket.blob(master_input_blob_path).download_as_string().decode('utf-8')
    input_metadata = {json.loads(line)['instance_id']: json.loads(line) for line in all_inputs_str.strip().split('\n')}

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
                    logging.error(f"Could not find input metadata for {cluster_id} in {master_input_blob_path}")
                    continue

                original_image_uri = input_data['gcs_image_uri']
                image_bbox = input_data['image_bbox']
                
                img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
                
                visualizer = MapVisualizer()
                acquisition_date = cluster_id.split('_')[2]
                
                final_map_image = visualizer.generate_fire_map(
                    base_image_bytes=image_bytes, image_bbox=image_bbox, ai_detections=ai_detections, acquisition_date_str=acquisition_date
                )
                
                # --- MODIFIED: Save maps into a date-stamped folder ---
                map_output_path = f"{FINAL_OUTPUT_GCS_PREFIX}{run_date}/{cluster_id}_map.png"
                map_blob = bucket.blob(map_output_path)
                
                img_byte_arr = BytesIO()
                final_map_image.save(img_byte_arr, format='PNG')
                map_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
                
                logging.info(f"Successfully generated map for {cluster_id} in run {run_date}.")

            except Exception as e:
                logging.error(f"Failed to process prediction line: '{line}'. Error: {e}", exc_info=True)

    logging.info("Result Processor function finished successfully.")
