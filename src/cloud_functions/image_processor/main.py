# src/cloud_functions/image_processor/main.py

import os
import json
import base64
import logging
from datetime import datetime

from google.cloud import aiplatform
from google.cloud import storage
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_OBJECT_DETECTION_MODEL")
BATCH_PREDICTION_MACHINE_TYPE = "n1-standard-4"
VERTEX_AI_BATCH_SERVICE_ACCOUNT = "fire-app-vm-service-account@haryo-kebakaran.iam.gserviceaccount.com"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def image_processor_cloud_function(event, context):
    """
    This function is triggered by a BATCH of fire clusters. It:
    1. Fetches satellite images for ALL clusters' locations.
    2. Creates a single JSONL file for all images.
    3. Submits ONE Vertex AI Batch Prediction job for all images.
    """
    logging.info("Image Processor function triggered for a batch of incidents.")

    if 'data' not in event:
        logging.error("No data found in the trigger event.")
        return

    message_data = base64.b64decode(event['data']).decode('utf-8')
    batch_data = json.loads(message_data)
    incidents = batch_data.get("incidents", [])

    if not incidents:
        logging.warning("Received a batch message with no incidents. Exiting.")
        return

    logging.info(f"Processing a batch of {len(incidents)} incidents from date {batch_data.get('incident_date')}.")

    regions_to_acquire = []
    for incident in incidents:
        cluster_id = incident.get("cluster_id")
        lat = incident.get("centroid_latitude")
        lon = incident.get("centroid_longitude")

        if not all([cluster_id, lat, lon]):
            logging.warning(f"Skipping incident with missing data: {incident}")
            continue

        bbox_size = 0.1
        cluster_bbox = [lon - bbox_size/2, lat - bbox_size/2, lon + bbox_size/2, lat + bbox_size/2]

        regions_to_acquire.append({
            "id": cluster_id,
            "name": f"Incident area for {cluster_id}",
            "bbox": cluster_bbox
        })

    try:
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        gcs_image_uris = imagery_acquirer.acquire_and_export_imagery(regions_to_acquire)

        if not gcs_image_uris:
            logging.error("Failed to acquire any satellite images for the batch.")
            return
        
        logging.info(f"Successfully acquired {len(gcs_image_uris)} of {len(regions_to_acquire)} requested images.")

        batch_input_lines = []
        for region in regions_to_acquire:
            cluster_id = region["id"]
            if cluster_id in gcs_image_uris:
                batch_input = {
                    "instance_id": cluster_id,
                    "gcs_image_uri": gcs_image_uris[cluster_id],
                    "image_bbox": region["bbox"] 
                }
                batch_input_lines.append(json.dumps(batch_input))
        
        if not batch_input_lines:
            logging.error("No images were successfully processed to create a batch input file.")
            return

        jsonl_content = "\n".join(batch_input_lines)
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        input_filename = f"incident_inputs/{batch_id}.jsonl"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(input_filename)
        blob.upload_from_string(jsonl_content)
        logging.info(f"Uploaded batch input file to gs://{GCS_BUCKET_NAME}/{input_filename}")

        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model_list = aiplatform.Model.list(filter=f'display_name="{VERTEX_AI_MODEL_NAME}"')
        if not model_list:
            logging.error(f"Could not find Vertex AI Model: {VERTEX_AI_MODEL_NAME}")
            return
        model = model_list[0]

        job_display_name = f"batch_prediction_{batch_id}"
        job = model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=f"gs://{GCS_BUCKET_NAME}/{input_filename}",
            gcs_destination_prefix=f"gs://{GCS_BUCKET_NAME}/incident_outputs/{batch_id}/",
            sync=False,
            machine_type=BATCH_PREDICTION_MACHINE_TYPE,
            service_account=VERTEX_AI_BATCH_SERVICE_ACCOUNT,
        )
        
        # --- FIX: Log the display_name, which is available immediately ---
        logging.info(f"Successfully submitted Vertex AI Batch Prediction job. Display name: {job.display_name}")

    except Exception as e:
        logging.error(f"An error occurred during batch image processing. Error: {e}", exc_info=True)
        # Re-raising the exception will cause the Cloud Function to report a failure.
        raise

    logging.info("Image Processor function finished for the batch.")
