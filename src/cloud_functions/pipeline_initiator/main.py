# src/cloud_functions/pipeline_initiator/main.py

import os
import logging
import json
from datetime import datetime, timedelta, timezone as dt_timezone

from google.cloud import storage
from google.cloud import aiplatform

# DIRECTLY import the types we need from their specific modules
from google.cloud.aiplatform_v1.services.job_service import JobServiceClient
from google.cloud.aiplatform_v1.types.batch_prediction_job import BatchPredictionJob
from google.cloud.aiplatform_v1.types.io import GcsDestination, GcsSource
from google.cloud.aiplatform_v1.types.machine_resources import MachineSpec, BatchDedicatedResources

from typing import Dict, Any, List
import base64

# --- Logging Setup ---
logger_global = logging.getLogger(__name__)

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.now(dt_timezone.utc).isoformat(),
        "component": "PipelineInitiatorCF",
        **kwargs
    }
    print(json.dumps(log_entry))
# --- End Logging Setup ---

# Import components from our project structure
from src.common.config import MONITORED_REGIONS, GCS_BUCKET_NAME
from src.firms_data_retriever.retriever import FirmsDataRetriever
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer

# --- Configuration Constants ---
GCS_BATCH_INPUT_DIR = "vertex_ai_batch_inputs/"
GCS_BATCH_OUTPUT_DIR_PREFIX = "vertex_ai_batch_outputs/"


def _get_vertex_ai_model_resource_name(model_display_name: str, project_id: str, location: str) -> str:
    """
    Retrieves the full resource name for the latest version of a Vertex AI model
    based on its display name.
    """
    try:
        aiplatform.init(project=project_id, location=location)
        models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"',
            order_by="update_time desc"
        )
        if not models:
            err_msg = f"Vertex AI Model with display_name '{model_display_name}' not found."
            _log_json("ERROR", err_msg, model_name=model_display_name, project_id=project_id, location=location)
            raise ValueError(err_msg)

        latest_model = models[0]
        _log_json("INFO", "Found Vertex AI Model.",
                   model_display_name=model_display_name,
                   resource_name=latest_model.resource_name,
                   version_id=latest_model.version_id)
        return latest_model.resource_name
    except Exception as e:
        _log_json("CRITICAL", f"Failed to retrieve Vertex AI model resource name for '{model_display_name}'.", error=str(e))
        raise


# --- Main Cloud Function entry point ---
def pipeline_initiator_cloud_function(event: Dict, context: Dict):
    _log_json("INFO", "Pipeline Initiator Cloud Function triggered.")

    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    gcp_region = os.environ.get("GCP_REGION")
    firms_api_key = os.environ.get("FIRMS_API_KEY")
    vertex_ai_model_name = os.environ.get("VERTEX_AI_MODEL_NAME")

    core_env_vars = {
        "GCP_PROJECT_ID": gcp_project_id, "GCP_REGION": gcp_region,
        "FIRMS_API_KEY": firms_api_key, "VERTEX_AI_MODEL_NAME": vertex_ai_model_name,
        "GCS_BUCKET_NAME (from config)": GCS_BUCKET_NAME
    }
    missing_vars = [var for var, val in core_env_vars.items() if not val]

    if missing_vars:
        _log_json("CRITICAL", "Missing one or more required environment variables.", missing_variables=missing_vars)
        raise ValueError(f"Missing required configurations: {', '.join(missing_vars)}")

    _log_json("INFO", "Environment variables and configurations loaded.")
    
    target_date_for_data = datetime.now(dt_timezone.utc) - timedelta(days=1)
    acquisition_date_str = target_date_for_data.strftime('%Y-%m-%d')
    _log_json("INFO", f"Target acquisition date for data: {acquisition_date_str}")

    try:
        storage_client = storage.Client(project=gcp_project_id)
        aiplatform.init(project=gcp_project_id, location=gcp_region)

        _log_json("INFO", "Initiating FIRMS data retrieval.")
        firms_retriever = FirmsDataRetriever(
            api_key=firms_api_key,
            base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/",
            sensors=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
        )
        firms_hotspots_df = firms_retriever.get_and_filter_firms_data(MONITORED_REGIONS)
        _log_json("INFO", "FIRMS data retrieval complete.", firms_hotspots_count=len(firms_hotspots_df))

        _log_json("INFO", "Initiating satellite imagery acquisition (GEE export tasks).")
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        gcs_image_uris_by_region_id: Dict[str, str] = imagery_acquirer.acquire_and_export_imagery(
            MONITORED_REGIONS, acquisition_date=acquisition_date_str
        )
        _log_json("INFO", "Satellite imagery GEE export tasks initiation complete.")

        if not gcs_image_uris_by_region_id:
            _log_json("WARNING", "No satellite imagery export tasks were initiated. Skipping Vertex AI Batch Prediction.")
            return

        _log_json("INFO", "Preparing Vertex AI Batch Prediction input file (JSONL).")
        
        # *** CORRECTED INSTANCE FORMATTING FOR THE DEPLOYED FUNCTION ***
        batch_input_instances: List[Dict[str, Any]] = []
        for region_id, gcs_image_uri in gcs_image_uris_by_region_id.items():
            region_metadata = next((r for r in MONITORED_REGIONS if r["id"] == region_id), None)
            if not region_metadata: continue
            
            region_firms_count = 0
            if not firms_hotspots_df.empty and 'monitored_region_id' in firms_hotspots_df.columns:
                region_firms_count = firms_hotspots_df[firms_hotspots_df['monitored_region_id'] == region_id].shape[0]

            instance_id = f"{region_id}_{acquisition_date_str.replace('-', '')}"
            # Create a flat dictionary, as expected by predictor.py
            batch_input_instances.append({
                "instance_id": instance_id,
                "gcs_image_uri": gcs_image_uri,
                "region_metadata": region_metadata,
                "firms_hotspot_count_in_region": region_firms_count
            })

        if not batch_input_instances:
            _log_json("WARNING", "No valid instances prepared for Vertex AI Batch Prediction. Exiting.")
            return

        timestamp_str = datetime.now(dt_timezone.utc).strftime('%Y%m%d%H%M%S%f')
        input_filename = f"batch_input_{acquisition_date_str.replace('-', '')}_{timestamp_str}.jsonl"
        input_gcs_path_in_bucket = f"{GCS_BATCH_INPUT_DIR}{input_filename}"
        input_gcs_uri = f"gs://{GCS_BUCKET_NAME}/{input_gcs_path_in_bucket}"

        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(input_gcs_path_in_bucket)
        jsonl_content = "\n".join([json.dumps(instance) for instance in batch_input_instances])
        blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        _log_json("INFO", "Vertex AI Batch Prediction input file uploaded to GCS.", input_gcs_uri=input_gcs_uri)

        model_resource_name = _get_vertex_ai_model_resource_name(vertex_ai_model_name, gcp_project_id, gcp_region)
        batch_job_output_gcs_prefix = f"gs://{GCS_BUCKET_NAME}/{GCS_BATCH_OUTPUT_DIR_PREFIX}"
        job_display_name = f"wildfire_detection_batch_{acquisition_date_str.replace('-', '')}_{timestamp_str}"
        
        input_config = BatchPredictionJob.InputConfig(instances_format="jsonl", gcs_source=GcsSource(uris=[input_gcs_uri]))
        output_config = BatchPredictionJob.OutputConfig(predictions_format="jsonl", gcs_destination=GcsDestination(output_uri_prefix=batch_job_output_gcs_prefix))
        machine_spec = MachineSpec(machine_type="n1-standard-4")
        dedicated_resources = BatchDedicatedResources(machine_spec=machine_spec, starting_replica_count=1, max_replica_count=5)

        batch_prediction_job_resource = BatchPredictionJob(
            display_name=job_display_name, model=model_resource_name,
            input_config=input_config, output_config=output_config,
            dedicated_resources=dedicated_resources,
            service_account=f"fire-app-vm-service-account@{gcp_project_id}.iam.gserviceaccount.com",
        )

        client_options = {"api_endpoint": f"{gcp_region}-aiplatform.googleapis.com"}
        job_service_client = JobServiceClient(client_options=client_options)
        parent_path = f"projects/{gcp_project_id}/locations/{gcp_region}"

        created_job_response = job_service_client.create_batch_prediction_job(parent=parent_path, batch_prediction_job=batch_prediction_job_resource)
        _log_json("INFO", "Vertex AI Batch Prediction job creation request sent successfully.", job_name=created_job_response.name)

    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during pipeline initiation.", error=str(e))
        import traceback
        _log_json("ERROR", "Traceback:", traceback_details=traceback.format_exc())
        raise e

    _log_json("INFO", "Pipeline Initiator Cloud Function execution finished.")


# --- Main Local Testing Entrypoint ---
if __name__ == "__main__":
    print("--- Running FAST local test for Pipeline Initiator ---")

    os.environ["GCP_PROJECT_ID"] = os.environ.get("GCP_PROJECT_ID", "haryo-kebakaran")
    os.environ["GCP_REGION"] = os.environ.get("GCP_REGION", "asia-southeast2")
    os.environ["VERTEX_AI_MODEL_NAME"] = "wildfire-cpr-predictor-model"
    
    print(f"Using Project: {os.environ['GCP_PROJECT_ID']}, Region: {os.environ['GCP_REGION']}")
    print(f"Using Model: {os.environ['VERTEX_AI_MODEL_NAME']}")

    _log_json("INFO", "[LOCAL TEST] Simulating FIRMS and GEE outputs.")
    
    mock_gcs_image_folder = "mock_trigger_images" 
    
    gcs_image_uris_by_region_id = {
        region["id"]: f"gs://{GCS_BUCKET_NAME}/{mock_gcs_image_folder}/" + \
                      ("force_fire_detection.png" if i % 2 == 0 else "force_no_fire_detection.png")
        for i, region in enumerate(MONITORED_REGIONS)
    }
    
    acquisition_date_str = (datetime.now(dt_timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    _log_json("INFO", "[LOCAL TEST] Mock data prepared.", mock_uris=gcs_image_uris_by_region_id)

    try:
        gcp_project_id = os.environ["GCP_PROJECT_ID"]
        gcp_region = os.environ["GCP_REGION"]
        vertex_ai_model_name = os.environ["VERTEX_AI_MODEL_NAME"]

        storage_client = storage.Client(project=gcp_project_id)
        
        _log_json("INFO", "[LOCAL TEST] Preparing Vertex AI Batch Prediction input file.")
        
        # *** CORRECTED INSTANCE FORMATTING FOR THE LOCAL TEST HARNESS ***
        batch_input_instances = []
        for region_id, gcs_image_uri in gcs_image_uris_by_region_id.items():
            region_metadata = next((r for r in MONITORED_REGIONS if r["id"] == region_id), {})
            instance_id = f"{region_id}_{acquisition_date_str.replace('-', '')}"
            # Create a flat dictionary, as expected by predictor.py
            batch_input_instances.append({
                "instance_id": instance_id,
                "gcs_image_uri": gcs_image_uri,
                "region_metadata": region_metadata,
                "firms_hotspot_count_in_region": 5 
            })

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
        batch_prediction_job_resource = BatchPredictionJob(display_name=job_display_name, model=model_resource_name, input_config=input_config, output_config=output_config, dedicated_resources=dedicated_resources, service_account=f"fire-app-vm-service-account@{gcp_project_id}.iam.gserviceaccount.com")
        
        client_options = {"api_endpoint": f"{gcp_region}-aiplatform.googleapis.com"}
        job_service_client = JobServiceClient(client_options=client_options)
        parent_path = f"projects/{gcp_project_id}/locations/{gcp_region}"
        created_job_response = job_service_client.create_batch_prediction_job(parent=parent_path, batch_prediction_job=batch_prediction_job_resource)
        
        _log_json("SUCCESS", "[LOCAL TEST] Vertex AI Batch Prediction job creation request sent successfully.", job_name=created_job_response.name)
        print("--- Local test finished. Check the Vertex AI console for the batch job. ---")

    except Exception as e:
        _log_json("CRITICAL", "[LOCAL TEST] An unhandled error occurred.", error=str(e))
        import traceback
        traceback.print_exc()
