# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
from io import BytesIO

from google.cloud import storage, aiplatform
from PIL import Image
# Make sure MapVisualizer is accessible. The Cloud Build config already copies the 'src' directory.
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
# These must be set as environment variables in the Cloud Function deployment
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

FINAL_OUTPUT_GCS_PREFIX = "final_outputs/"

# --- Initializations ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
storage_client = storage.Client()
# Initialize the AI Platform client.
# It's best to do this once outside the function handler for efficiency.
try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logging.info(f"Vertex AI client initialized for project '{GCP_PROJECT_ID}' and region '{GCP_REGION}'.")
except Exception as e:
    logging.critical(f"Failed to initialize Vertex AI client. Ensure GCP_PROJECT_ID and GCP_REGION env vars are set. Error: {e}")


def result_processor_cloud_function(event, context):
    """
    Triggered by a Pub/Sub message from a Cloud Logging sink when a Vertex AI
    Batch Prediction job completes. This function processes the results.
    """
    logging.info("Result Processor function triggered.")

    if not all([GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION]):
        logging.critical("Missing one or more required environment variables: GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION.")
        return

    try:
        # Decode the incoming Pub/Sub message, which is a log entry
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)

        # Get job_id from the log and fetch job details via API
        job_id = log_entry['resource']['labels']['job_id']
        logging.info(f"Extracted Vertex AI Batch Prediction job ID: {job_id}")

        # Fetch the full job object using its resource name
        job_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/batchPredictionJobs/{job_id}"
        batch_job = aiplatform.BatchPredictionJob(job_resource_name)
        
        # The displayName was set by the image_processor (e.g., "batch_prediction_2025-06-13")
        job_display_name = batch_job.display_name
        
        # Extract the run_date from the displayName
        run_date = job_display_name.replace('batch_prediction_', '')
        
        if not run_date:
            raise ValueError(f"Could not extract run_date from job display name: '{job_display_name}'")

        gcs_output_prefix = f"incident_outputs/{run_date}/"
        logging.info(f"Processing prediction results for run_date '{run_date}' from GCS prefix: {gcs_output_prefix}")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logging.error(f"Could not parse trigger event or determine run details. Error: {e}", extra={"payload": event.get('data')})
        return

    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    # List all prediction result files in the specific date-based output directory
    prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
    
    if not prediction_blobs:
        logging.error(f"No prediction result files found at prefix: {gcs_output_prefix}")
        return

    # Load the master input file to get metadata for all clusters in this run
    master_input_blob_path = f"incident_inputs/{run_date}.jsonl"
    try:
        all_inputs_str = bucket.blob(master_input_blob_path).download_as_string().decode('utf-8')
        # Create a dictionary mapping each cluster_id to its input metadata for easy lookup
        input_metadata = {json.loads(line)['instance_id']: json.loads(line) for line in all_inputs_str.strip().split('\n')}
    except Exception as e:
        logging.error(f"Could not read or parse the master input file: {master_input_blob_path}. Error: {e}")
        return

    # Iterate through each prediction results file (usually just one)
    for prediction_blob in prediction_blobs:
        # The prediction files from Vertex AI typically start with "prediction.results-".
        if 'prediction' not in prediction_blob.name:
            continue
            
        logging.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        # Each line in the file is a separate JSON object for a prediction
        for line in prediction_content_str.strip().split('\n'):
            try:
                # --- FINAL FIX: Ensure the line is not empty before parsing ---
                if not line.strip():
                    continue # Skip blank or whitespace-only lines
                # --- END OF FIX ---

                prediction_data = json.loads(line)
                # The actual prediction is nested under the 'predictions' key
                ai_detections = prediction_data.get('predictions')
                if not ai_detections: continue

                # We only expect one prediction per line for this setup
                prediction = ai_detections[0]
                cluster_id = prediction.get("instance_id")
                
                # Look up the corresponding input data using the cluster_id
                input_data = input_metadata.get(cluster_id)
                if not input_data:
                    logging.error(f"Could not find input metadata for cluster_id '{cluster_id}'")
                    continue

                # Get the original image URI and its bounding box from the input metadata
                original_image_uri = input_data['gcs_image_uri']
                image_bbox = input_data['image_bbox']
                
                # Download the original satellite image from GCS
                img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
                
                # Generate the visual map
                visualizer = MapVisualizer()
                
                final_map_image = visualizer.generate_fire_map(
                    base_image_bytes=image_bytes, 
                    image_bbox=image_bbox, 
                    ai_detections=ai_detections, # Pass the full list of detections
                    acquisition_date_str=run_date # The acquisition date is the run date
                )
                
                # Define the final output path for the generated map
                map_output_path = f"{FINAL_OUTPUT_GCS_PREFIX}{run_date}/{cluster_id}_map.png"
                map_blob = bucket.blob(map_output_path)
                
                # Save the generated map image to a byte buffer and upload to GCS
                img_byte_arr = BytesIO()
                final_map_image.save(img_byte_arr, format='PNG')
                map_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
                
                logging.info(f"Successfully generated and uploaded map for '{cluster_id}' in run '{run_date}' to '{map_output_path}'.")

            except Exception as e:
                logging.error(f"Failed to process a prediction line: '{line}'. Error: {e}", exc_info=True)

    logging.info("Result Processor function finished successfully.")
