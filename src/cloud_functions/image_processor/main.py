# src/cloud_functions/image_processor/main.py

import os
import json
import base64
import logging
from datetime import datetime

from google.cloud import aiplatform
from google.cloud import storage

# We still use the acquirer, but in a more targeted way
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
# IMPORTANT: This must point to your new object detection model in the Vertex AI Model Registry
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_OBJECT_DETECTION_MODEL") 

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def image_processor_cloud_function(event, context):
    """
    This function is triggered by a single fire cluster detection. It:
    1. Fetches a satellite image for the cluster's location.
    2. Submits a Vertex AI Batch Prediction job for that single image.
    """
    logging.info("Image Processor function triggered.")
    
    # 1. Parse the incoming message from the IncidentDetector
    if 'data' not in event:
        logging.error("No data found in the trigger event.")
        return

    message_data = base64.b64decode(event['data']).decode('utf-8')
    cluster_data = json.loads(message_data)
    
    cluster_id = cluster_data.get("cluster_id")
    lat = cluster_data.get("centroid_latitude")
    lon = cluster_data.get("centroid_longitude")

    if not all([cluster_id, lat, lon]):
        logging.error(f"Missing critical data in message: {cluster_data}")
        return

    logging.info(f"Processing cluster {cluster_id} at ({lat}, {lon}).")

    # 2. Fetch a targeted satellite image
    # We define a small bounding box around the cluster centroid.
    # A 0.1 x 0.1 degree box is roughly 11km x 11km, good for a detailed view.
    bbox_size = 0.1 
    cluster_bbox = [lon - bbox_size/2, lat - bbox_size/2, lon + bbox_size/2, lat + bbox_size/2]
    
    # We create a temporary "region" object to pass to our existing acquirer function.
    # This shows the flexibility of the original acquirer code.
    region_for_acquirer = {
        "id": cluster_id,
        "name": f"Incident area for {cluster_id}",
        "bbox": cluster_bbox,
        "description": "Dynamically generated region for a specific fire cluster."
    }

    try:
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        # We use the existing acquirer, which is flexible enough to handle this dynamic region
        gcs_image_uris = imagery_acquirer.acquire_and_export_imagery([region_for_acquirer])

        if not gcs_image_uris or cluster_id not in gcs_image_uris:
            logging.error(f"Failed to acquire satellite image for cluster {cluster_id}.")
            return

        image_gcs_uri = gcs_image_uris[cluster_id]
        logging.info(f"Successfully initiated image export for cluster {cluster_id} to {image_gcs_uri}")

        # 3. Trigger the Vertex AI Object Detection Model
        # A Batch Prediction job is still a good, robust choice for single images
        # because it's asynchronous and handles retries automatically.
        
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        
        model_list = aiplatform.Model.list(filter=f'display_name="{VERTEX_AI_MODEL_NAME}"')
        if not model_list:
            logging.error(f"Could not find Vertex AI Model with display name: {VERTEX_AI_MODEL_NAME}")
            return
        model = model_list[0]

        # The input for the batch job is a simple JSONL file with one line
        batch_input = {"gcs_image_uri": image_gcs_uri, "instance_id": cluster_id}
        
        # Create a unique GCS path for this job's input file
        input_filename = f"incident_inputs/{cluster_id}.jsonl"
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(input_filename)
        blob.upload_from_string(json.dumps(batch_input))

        # Define and start the batch prediction job
        job = model.batch_predict(
            job_display_name=f"object_detection_{cluster_id.replace('-', '_')}",
            gcs_source=f"gs://{GCS_BUCKET_NAME}/{input_filename}",
            gcs_destination_prefix=f"gs://{GCS_BUCKET_NAME}/incident_outputs/{cluster_id}/",
            sync=False # Run asynchronously and let the function exit
        )
        logging.info(f"Started Vertex AI Batch Prediction job for {cluster_id}. Job name: {job.resource_name}")

    except Exception as e:
        logging.error(f"An error occurred during image processing for cluster {cluster_id}. Error: {e}", exc_info=True)
        # Re-raise the exception to have the function retry automatically
        raise

    logging.info(f"Image Processor function finished for cluster {cluster_id}.")
