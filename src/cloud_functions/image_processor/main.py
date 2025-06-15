# src/cloud_functions/image_processor/main.py

import os
import json
import base64
import logging
from datetime import datetime

from google.cloud import aiplatform, storage
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_OBJECT_DETECTION_MODEL")
BATCH_PREDICTION_MACHINE_TYPE = "n1-standard-4"
VERTEX_AI_BATCH_SERVICE_ACCOUNT = "fire-app-vm-service-account@haryo-kebakaran.iam.gserviceaccount.com"
INCIDENTS_GCS_PREFIX = "incidents" # Folder where incident data is stored

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def image_processor_cloud_function(event, context):
    logging.info("Image Processor function triggered.")

    # --- MODIFIED: The function now determines its own execution date. ---
    # It no longer relies on the Pub/Sub message for the date.
    # This assumes the function runs on the same UTC day as the incident_detector.
    run_date = datetime.utcnow().strftime('%Y-%m-%d')
    logging.info(f"Processing for run_date determined by system clock: {run_date}.")

    # --- Read the incidents from the GCS file based on the determined date ---
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    incidents_blob_name = f"{INCIDENTS_GCS_PREFIX}/{run_date}/detected_incidents.jsonl"
    
    try:
        incidents_content = bucket.blob(incidents_blob_name).download_as_string().decode('utf-8')
        incidents = [json.loads(line) for line in incidents_content.strip().split('\n')]
        logging.info(f"Successfully loaded {len(incidents)} incidents from GCS for date {run_date}.")
    except Exception as e:
        # This will fail if this function runs on a different UTC day than the previous one.
        logging.error(f"Failed to read incidents file from GCS: gs://{GCS_BUCKET_NAME}/{incidents_blob_name}. Error: {e}")
        return

    if not incidents:
        logging.warning("Incidents file was empty or could not be parsed. Exiting.")
        return

    regions_to_acquire = []
    for incident in incidents:
        cluster_id = incident.get("cluster_id")
        lat = incident.get("centroid_latitude")
        lon = incident.get("centroid_longitude")
        if not all([cluster_id, lat, lon]): continue
        bbox_size = 0.1
        cluster_bbox = [lon - bbox_size/2, lat - bbox_size/2, lon + bbox_size/2, lat + bbox_size/2]
        regions_to_acquire.append({"id": cluster_id, "name": f"Incident area for {cluster_id}", "bbox": cluster_bbox})

    try:
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        gcs_image_uris = imagery_acquirer.acquire_and_export_imagery(regions_to_acquire, acquisition_date=run_date)

        if not gcs_image_uris:
            logging.error("Failed to acquire any satellite images for the batch.")
            return
        
        logging.info(f"Successfully initiated export for {len(gcs_image_uris)} of {len(regions_to_acquire)} requested images.")

        clusters_for_batch = []
        for region in regions_to_acquire:
            cluster_id = region["id"]
            if cluster_id in gcs_image_uris:
                cluster_data = {
                    "cluster_id": cluster_id,
                    "gcs_image_uri": gcs_image_uris[cluster_id],
                    "image_bbox": region["bbox"]
                }
                clusters_for_batch.append(cluster_data)
        
        if not clusters_for_batch:
            logging.error("No images were successfully processed to create a batch input file.")
            return

        batch_instance = {
            "instance_id": f"batch_run_{run_date.replace('-', '')}",
            "clusters": clusters_for_batch
        }
        
        jsonl_content = json.dumps(batch_instance)
        input_filename = f"incident_inputs/{run_date}.jsonl"
        blob = bucket.blob(input_filename)
        blob.upload_from_string(jsonl_content)
        logging.info(f"Uploaded single-instance batch input file to gs://{GCS_BUCKET_NAME}/{input_filename}")

        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model = aiplatform.Model.list(filter=f'display_name="{VERTEX_AI_MODEL_NAME}"')[0]

        job_display_name = f"batch_prediction_{run_date}"
        job = model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=f"gs://{GCS_BUCKET_NAME}/{input_filename}",
            gcs_destination_prefix=f"gs://{GCS_BUCKET_NAME}/incident_outputs/{run_date}/",
            sync=False,
            machine_type=BATCH_PREDICTION_MACHINE_TYPE,
            service_account=VERTEX_AI_BATCH_SERVICE_ACCOUNT,
        )
        
        logging.info(f"Successfully submitted Vertex AI Batch Prediction job. Display name: {job_display_name}")

    except Exception as e:
        logging.error(f"An error occurred during batch image processing. Error: {e}", exc_info=True)
        raise

    logging.info("Image Processor function finished for the batch.")
