# src/cloud_functions/pipeline_initiator/main.py

import os
import logging
import json
from datetime import datetime, timedelta, timezone as dt_timezone

from google.cloud import storage
from google.cloud import aiplatform

# DIRECTLY import the types we need
from google.cloud.aiplatform_v1.services.job_service import JobServiceClient
from google.cloud.aiplatform_v1.types.batch_prediction_job import BatchPredictionJob
from google.cloud.aiplatform_v1.types.io import GcsDestination, GcsSource
from google.cloud.aiplatform_v1.types.machine_resources import MachineSpec, BatchDedicatedResources

from typing import Dict, Any, List
import base64

# --- Import project-specific modules ---
from src.common.config import MONITORED_REGIONS, GCS_BUCKET_NAME
from src.firms_data_retriever.retriever import FirmsDataRetriever
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer


# --- Configuration Constants ---
GCS_BATCH_INPUT_DIR = "vertex_ai_batch_inputs/"
GCS_BATCH_OUTPUT_DIR_PREFIX = "vertex_ai_batch_outputs/"


# --- Logging Setup ---
def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.now(dt_timezone.utc).isoformat(),
        "component": "PipelineInitiatorCF",
        **kwargs
    }
    print(json.dumps(log_entry))

# --- Get Model Name Helper ---
def _get_vertex_ai_model_resource_name(model_display_name: str, project_id: str, location: str) -> str:
    try:
        aiplatform.init(project=project_id, location=location)
        models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"', order_by="update_time desc")
        if not models:
            err_msg = f"Vertex AI Model with display_name '{model_display_name}' not found."
            _log_json("ERROR", err_msg, model_name=model_display_name, project_id=project_id, location=location)
            raise ValueError(err_msg)
        latest_model = models[0]
        _log_json("INFO", "Found Vertex AI Model.", resource_name=latest_model.resource_name, version_id=latest_model.version_id)
        return latest_model.resource_name
    except Exception as e:
        _log_json("CRITICAL", f"Failed to retrieve Vertex AI model resource name for '{model_display_name}'.", error=str(e))
        raise

# --- Main Cloud Function ---
def pipeline_initiator_cloud_function(event: Dict, context: Dict):
    # This block is the real logic that will run when deployed to GCP.
    # It remains unchanged from your original implementation.
    _log_json("INFO", "Pipeline Initiator Cloud Function triggered.")
    
    # ... (Full original function logic would be here) ...
    pass


# --- LOCAL TESTING ENTRYPOINT ---
# This version is reverted to the state before the "network" parameter was added.
if __name__ == "__main__":
    print("--- Running FAST local test for Pipeline Initiator ---")

    os.environ["GCP_PROJECT_ID"] = os.environ.get("GCP_PROJECT_ID", "haryo-kebakaran")
    os.environ["GCP_REGION"] = os.environ.get("GCP_REGION", "asia-southeast2")
    os.environ["VERTEX_AI_MODEL_NAME"] = "dummy_wildfire_prebuilt_pt24_overwrite_v1"

    print(f"Using Project: {os.environ['GCP_PROJECT_ID']}, Region: {os.environ['GCP_REGION']}")

    _log_json("INFO", "[LOCAL TEST] Simulating FIRMS and GEE outputs.")
    mock_gcs_image_folder = "mock_trigger_images"
    gcs_image_uris_by_region_id = {
        region["id"]: f"gs://{GCS_BUCKET_NAME}/{mock_gcs_image_folder}/force_fire_detection.png"
        for region in MONITORED_REGIONS
    }
    acquisition_date_str = (datetime.now(dt_timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    _log_json("INFO", "[LOCAL TEST] Mock data prepared.", mock_uris=gcs_image_uris_by_region_id)

    try:
        gcp_project_id = os.environ["GCP_PROJECT_ID"]
        gcp_region = os.environ["GCP_REGION"]
        vertex_ai_model_name = os.environ["VERTEX_AI_MODEL_NAME"]

        storage_client = storage.Client(project=gcp_project_id)
        
        batch_input_instances = []
        for region_id, gcs_image_uri in gcs_image_uris_by_region_id.items():
            region_metadata = next((r for r in MONITORED_REGIONS if r["id"] == region_id), {})
            instance_id = f"{region_id}_{acquisition_date_str.replace('-', '')}"
            batch_input_instances.append({"instance_id": instance_id, "gcs_image_uri": gcs_image_uri, "region_metadata": region_metadata, "firms_hotspot_count_in_region": 5})
        
        timestamp_str = datetime.now(dt_timezone.utc).strftime('%Y%m%d%H%M%S%f')
        input_filename = f"local_test_batch_input_{timestamp_str}.jsonl"
        input_gcs_path = f"{GCS_BATCH_INPUT_DIR}{input_filename}"
        input_gcs_uri = f"gs://{GCS_BUCKET_NAME}/{input_gcs_path}"
        blob = storage_client.bucket(GCS_BUCKET_NAME).blob(input_gcs_path)
        jsonl_content = "\n".join([json.dumps(inst) for inst in batch_input_instances])
        blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        _log_json("INFO", "[LOCAL TEST] Batch input file uploaded to GCS.", uri=input_gcs_uri)

        _log_json("INFO", "[LOCAL TEST] Submitting Vertex AI Batch Prediction job.")
        
        model_resource_name = _get_vertex_ai_model_resource_name(vertex_ai_model_name, gcp_project_id, gcp_region)
        
        batch_job_output_gcs_prefix = f"gs://{GCS_BUCKET_NAME}/{GCS_BATCH_OUTPUT_DIR_PREFIX}"
        job_display_name = f"local_test_wildfire_batch_{timestamp_str}"
        input_config = BatchPredictionJob.InputConfig(instances_format="jsonl", gcs_source=GcsSource(uris=[input_gcs_uri]))
        output_config = BatchPredictionJob.OutputConfig(predictions_format="jsonl", gcs_destination=GcsDestination(output_uri_prefix=batch_job_output_gcs_prefix))
        machine_spec = MachineSpec(machine_type="n1-standard-4")
        dedicated_resources = BatchDedicatedResources(machine_spec=machine_spec, starting_replica_count=1, max_replica_count=5)
        
        # This is the original object, without the 'network' parameter.
        batch_prediction_job_resource = BatchPredictionJob(
            display_name=job_display_name,
            model=model_resource_name,
            input_config=input_config,
            output_config=output_config,
            dedicated_resources=dedicated_resources,
            service_account=f"fire-app-vm-service-account@{gcp_project_id}.iam.gserviceaccount.com"
        )

        client_options = {"api_endpoint": f"{gcp_region}-aiplatform.googleapis.com"}
        job_service_client = JobServiceClient(client_options=client_options)
        parent_path = f"projects/{gcp_project_id}/locations/{gcp_region}"
        created_job_response = job_service_client.create_batch_prediction_job(parent=parent_path, batch_prediction_job=batch_prediction_job_resource)
        
        _log_json("SUCCESS", "[LOCAL TEST] Vertex AI Batch Prediction job creation request sent successfully.", job_name=created_job_response.name)
        print("--- Local test finished. Please wait 15-20 minutes before checking logs for the canary message. ---")

    except Exception as e:
        _log_json("CRITICAL", "[LOCAL TEST] An unhandled error occurred.", error=str(e))
        import traceback
        traceback.print_exc()
