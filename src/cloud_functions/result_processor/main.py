# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
from io import BytesIO

from google.cloud import storage
from google.cloud import aiplatform 
from PIL import Image
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION", "asia-southeast2") # Add region with a default
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "fire-app-bucket")
FINAL_OUTPUT_GCS_PREFIX = "final_outputs/"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
storage_client = storage.Client()

def result_processor_cloud_function(event, context):
    """
    Triggered by a Pub/Sub message when a Vertex AI Batch Prediction job completes.
    This function now uses the AI Platform API to find the correct output path.
    """
    logging.info("Result Processor function triggered.")

    try:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)
        
        job_id = log_entry['resource']['labels']['job_id']
        job_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/batchPredictionJobs/{job_id}"
        
        # --- NEW: Use the AI Platform client to get job details ---
        client_options = {"api_endpoint": f"{GCP_REGION}-aiplatform.googleapis.com"}
        job_client = aiplatform.gapic.JobServiceClient(client_options=client_options)
        
        batch_job = job_client.get_batch_prediction_job(name=job_resource_name)
        gcs_output_uri = batch_job.output_info.gcs_output_directory
        logging.info(f"Successfully fetched job details. Processing results from: {gcs_output_uri}")

    except Exception as e:
        logging.error(f"Could not parse trigger or fetch job details. Error: {e}", exc_info=True)
        return

    bucket_name, gcs_output_prefix = gcs_output_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
    
    if not prediction_blobs:
        logging.error(f"No prediction result file found at prefix: {gcs_output_prefix}")
        return

    for prediction_blob in prediction_blobs:
        if 'prediction' not in prediction_blob.name:
            continue
            
        logging.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        prediction_lines = prediction_content_str.strip().split('\n')

        for line in prediction_lines:
            try:
                prediction_data = json.loads(line)
                ai_detections = prediction_data.get('predictions')
                if not ai_detections:
                    logging.warning(f"Skipping malformed prediction line: {line}")
                    continue

                prediction = ai_detections[0]
                cluster_id = prediction.get("instance_id")
                logging.info(f"Processing parsed prediction for cluster: {cluster_id}")
                
                # The input data is in the original input file for the batch job
                batch_job_input_uri = batch_job.input_config.gcs_source.uris[0]
                input_bucket_name, master_input_blob_path = batch_job_input_uri.replace("gs://", "").split("/", 1)
                master_input_blob = storage_client.bucket(input_bucket_name).blob(master_input_blob_path)

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
