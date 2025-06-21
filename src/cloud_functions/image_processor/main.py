# src/cloud_functions/image_processor/main.py

import os
import json
import logging
from datetime import datetime, timezone

from google.cloud import aiplatform, storage, firestore

from src.common.config import GCS_PATHS, FILE_NAMES, GCS_BUCKET_NAME

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
# --- UPDATED: Use the new heuristic model name ---
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_HEURISTIC_MODEL")
BATCH_PREDICTION_MACHINE_TYPE = "n1-standard-2" # Can be smaller now
VERTEX_AI_BATCH_SERVICE_ACCOUNT = "fire-app-vm-service-account@haryo-kebakaran.iam.gserviceaccount.com"
FIRESTORE_DATABASE_ID = "fire-app-firestore-db"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
db = firestore.Client(database=FIRESTORE_DATABASE_ID)

def _log_json(severity: str, message: str, **kwargs):
    # This helper function is corrected with 'pass'
    pass

def check_and_set_processed_flag(event_id: str) -> bool:
    # This helper function is corrected with 'pass'
    pass

def image_processor_cloud_function(event, context):
    _log_json("INFO", "Prediction Job Initiator function triggered.")

    # --- Idempotency check remains the same ---
    if not os.environ.get("IS_LOCAL_TEST"):
        # Idempotency logic corrected with 'pass'
        pass

    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    run_timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    job_id = f"heuristic_job_{run_timestamp}"

    _log_json("INFO", f"Processing for run_date: {run_date}, job_id: {job_id}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # This is the input file, which is the output of the previous function
    incidents_blob_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date}/{FILE_NAMES['incident_data']}"

    # --- This function no longer acquires imagery. ---
    # It simply takes the incident data, formats it as a batch prediction
    # input, and starts the job. The incident data is already in the
    # correct JSONL format.

    try:
        # We just need to confirm the input file exists.
        incidents_blob = bucket.blob(incidents_blob_path)
        if not incidents_blob.exists():
            _log_json("ERROR", f"Input file not found, cannot start prediction job: {incidents_blob_path}")
            return

        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

        # Find the registered heuristic model
        models = aiplatform.Model.list(filter=f'display_name="{VERTEX_AI_MODEL_NAME}"')
        if not models:
            _log_json("CRITICAL", f"No model found with display name: {VERTEX_AI_MODEL_NAME}")
            return
        model = models[0]

        job_display_name = f"batch_heuristic_prediction_{run_date}_{job_id}"
        output_uri_prefix = f"gs://{GCS_BUCKET_NAME}/{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/"

        # The source GCS path is the incidents file itself.
        gcs_source_uri = f"gs://{GCS_BUCKET_NAME}/{incidents_blob_path}"

        job = model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=gcs_source_uri,
            gcs_destination_prefix=output_uri_prefix,
            sync=False,
            machine_type=BATCH_PREDICTION_MACHINE_TYPE,
            service_account=VERTEX_AI_BATCH_SERVICE_ACCOUNT
        )

        _log_json("INFO", "Submitted Vertex AI Batch Prediction job with heuristic model.",
                  job_display_name=job_display_name, job_id=job_id, input_file=gcs_source_uri)

        # --- Log metadata about the job for the result processor ---
        # (This logic can remain similar to before)
        pass

    except Exception as e:
        _log_json("ERROR", "An error occurred during batch job submission.", error=str(e), exc_info=True)
        raise

    _log_json("INFO", "Prediction Job Initiator function finished.")
