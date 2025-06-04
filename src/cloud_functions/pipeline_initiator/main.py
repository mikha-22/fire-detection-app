# src/cloud_functions/pipeline_initiator/main.py

import os
import logging
import json
from datetime import datetime, timedelta, timezone as dt_timezone

from google.cloud import storage
from google.cloud import aiplatform # Main SDK client initialization

# DIRECTLY import the types we need from their specific modules
from google.cloud.aiplatform_v1.services.job_service import JobServiceClient
from google.cloud.aiplatform_v1.types.batch_prediction_job import BatchPredictionJob
from google.cloud.aiplatform_v1.types.io import GcsDestination, GcsSource
from google.cloud.aiplatform_v1.types.machine_resources import MachineSpec, BatchDedicatedResources

from typing import Dict, Any, List
import base64 # For local testing Pub/Sub message encoding

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
# Ensure these imported files are also the latest corrected versions
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
        # aiplatform.init already called globally if needed, or can be called here too.
        # For safety, ensure it's initialized for the context of this function.
        aiplatform.init(project=project_id, location=location)
        models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"',
            order_by="update_time desc" # Get the latest version
        )
        if not models:
            err_msg = f"Vertex AI Model with display_name '{model_display_name}' not found in project '{project_id}', region '{location}'."
            _log_json("ERROR", err_msg, model_name=model_display_name, project_id=project_id, location=location)
            raise ValueError(err_msg)

        latest_model = models[0]
        _log_json("INFO", "Found Vertex AI Model.",
                   model_display_name=model_display_name,
                   resource_name=latest_model.resource_name,
                   version_id=latest_model.version_id,
                   update_time=latest_model.update_time.isoformat() if latest_model.update_time else "N/A")
        return latest_model.resource_name
    except Exception as e:
        _log_json("CRITICAL", f"Failed to retrieve Vertex AI model resource name for '{model_display_name}'.",
                   model_display_name=model_display_name, error=str(e), error_type=type(e).__name__)
        raise


# Main Cloud Function entry point
def pipeline_initiator_cloud_function(event: Dict, context: Dict):
    _log_json("INFO", "Pipeline Initiator Cloud Function triggered.",
               event_id=context.event_id if hasattr(context, 'event_id') else 'N/A',
               event_type=context.event_type if hasattr(context, 'event_type') else 'N/A',
               trigger_resource=context.resource.get("name") if hasattr(context, 'resource') and isinstance(context.resource, dict) else str(getattr(context, 'resource', 'N/A')),
               timestamp=context.timestamp if hasattr(context, 'timestamp') else 'N/A')

    # --- Configuration & Validation ---
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    gcp_region = os.environ.get("GCP_REGION")
    firms_api_key = os.environ.get("FIRMS_API_KEY")
    vertex_ai_model_name = os.environ.get("VERTEX_AI_MODEL_NAME", "dummy_wildfire_detector_v1")
    # VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME is no longer used by this function to configure the job directly.
    # It will be the target for the Cloud Logging Sink.

    core_env_vars = {
        "GCP_PROJECT_ID": gcp_project_id, "GCP_REGION": gcp_region,
        "FIRMS_API_KEY": firms_api_key, "VERTEX_AI_MODEL_NAME": vertex_ai_model_name,
        "GCS_BUCKET_NAME (from config)": GCS_BUCKET_NAME
    }
    missing_vars = [var for var, val in core_env_vars.items() if not val]

    if missing_vars:
        _log_json("CRITICAL", "Missing one or more required environment variables or configurations. Cannot proceed.",
                   missing_variables=missing_vars)
        raise ValueError(f"Missing required configurations: {', '.join(missing_vars)}")

    _log_json("INFO", "Environment variables and configurations loaded.",
               project_id=gcp_project_id, region=gcp_region,
               gcs_bucket=GCS_BUCKET_NAME, model_name=vertex_ai_model_name,
               firms_key_present=bool(firms_api_key))
    _log_json("INFO", "Note: Direct Vertex AI job notifications (NotificationSpec) are NOT configured by this function "
                      "due to client library limitations. A Cloud Logging Sink should be used to forward job "
                      "completion events to the appropriate Pub/Sub topic for the ResultProcessorCF.")


    target_date_for_data = datetime.now(dt_timezone.utc) - timedelta(days=1) # Process yesterday's data
    acquisition_date_str = target_date_for_data.strftime('%Y-%m-%d')
    _log_json("INFO", f"Target acquisition date for FIRMS and Satellite Imagery: {acquisition_date_str}")

    try:
        # Initialize clients
        storage_client = storage.Client(project=gcp_project_id)
        # Initialize aiplatform client, explicitly setting project and location for clarity
        aiplatform.init(project=gcp_project_id, location=gcp_region)

        # --- 1. FIRMS Data Retrieval ---
        _log_json("INFO", "Initiating FIRMS data retrieval.")
        firms_retriever = FirmsDataRetriever(
            api_key=firms_api_key,
            base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/",
            sensors=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
        )
        firms_hotspots_df = firms_retriever.get_and_filter_firms_data(MONITORED_REGIONS)
        _log_json("INFO", "FIRMS data retrieval complete.", firms_hotspots_count=len(firms_hotspots_df))

        # --- 2. Satellite Imagery Acquisition ---
        _log_json("INFO", "Initiating satellite imagery acquisition (GEE export tasks).")
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        gcs_image_uris_by_region_id: Dict[str, str] = imagery_acquirer.acquire_and_export_imagery(
            MONITORED_REGIONS, acquisition_date=acquisition_date_str
        )
        _log_json("INFO", "Satellite imagery GEE export tasks initiation complete.",
                           image_tasks_initiated_count=len(gcs_image_uris_by_region_id))

        if not gcs_image_uris_by_region_id:
            _log_json("WARNING", "No satellite imagery export tasks were initiated. Skipping Vertex AI Batch Prediction.")
            return # Exit if no images to process

        # --- 3. Prepare Vertex AI Batch Prediction Input ---
        _log_json("INFO", "Preparing Vertex AI Batch Prediction input file (JSONL).")
        batch_input_instances: List[Dict[str, Any]] = []
        for region_id, gcs_image_uri in gcs_image_uris_by_region_id.items():
            region_metadata = next((r for r in MONITORED_REGIONS if r["id"] == region_id), None)
            if not region_metadata:
                _log_json("WARNING", f"Metadata for region_id '{region_id}' not found in MONITORED_REGIONS. Skipping instance.",
                                   region_id_from_imagery=region_id)
                continue
            
            region_firms_count = 0
            if not firms_hotspots_df.empty and 'monitored_region_id' in firms_hotspots_df.columns:
                region_specific_firms = firms_hotspots_df[firms_hotspots_df['monitored_region_id'] == region_id]
                region_firms_count = region_specific_firms.shape[0]

            # Unique instance ID for each image, incorporating date
            instance_id = f"{region_id}_{acquisition_date_str.replace('-', '')}"
            batch_input_instances.append({
                "instance_id": instance_id,
                "gcs_image_uri": gcs_image_uri,
                "region_metadata": region_metadata, # Include full region metadata
                "firms_hotspot_count_in_region": region_firms_count
            })

        if not batch_input_instances:
            _log_json("WARNING", "No valid instances prepared for Vertex AI Batch Prediction after filtering. Exiting.")
            return

        # Upload batch input file to GCS
        timestamp_str = datetime.now(dt_timezone.utc).strftime('%Y%m%d%H%M%S%f') # Microsecond precision for uniqueness
        input_filename = f"batch_input_{acquisition_date_str.replace('-', '')}_{timestamp_str}.jsonl"
        input_gcs_path_in_bucket = f"{GCS_BATCH_INPUT_DIR}{input_filename}"
        input_gcs_uri = f"gs://{GCS_BUCKET_NAME}/{input_gcs_path_in_bucket}"

        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(input_gcs_path_in_bucket)
        jsonl_content = "\n".join([json.dumps(instance) for instance in batch_input_instances])
        blob.upload_from_string(jsonl_content, content_type="application/jsonl", timeout=60)
        _log_json("INFO", "Vertex AI Batch Prediction input file uploaded to GCS.",
                           input_gcs_uri=input_gcs_uri, num_instances=len(batch_input_instances))

        # --- 4. Submit Vertex AI Batch Prediction Job ---
        _log_json("INFO", "Constructing Vertex AI Batch Prediction job arguments.")
        model_resource_name = _get_vertex_ai_model_resource_name(vertex_ai_model_name, gcp_project_id, gcp_region)
        
        # Output prefix in GCS for this job's results
        batch_job_output_gcs_prefix = f"gs://{GCS_BUCKET_NAME}/{GCS_BATCH_OUTPUT_DIR_PREFIX}"
        
        job_display_name = f"wildfire_detection_batch_{acquisition_date_str.replace('-', '')}_{timestamp_str}"
        
        input_config = BatchPredictionJob.InputConfig(
            instances_format="jsonl",
            gcs_source=GcsSource(uris=[input_gcs_uri])
        )
        output_config = BatchPredictionJob.OutputConfig(
            predictions_format="jsonl", # Output from handler will be JSONL
            gcs_destination=GcsDestination(output_uri_prefix=batch_job_output_gcs_prefix)
        )
        
        # Define machine resources for the batch prediction job
        machine_spec = MachineSpec(machine_type="n1-standard-4") # Example, adjust based on model needs
        dedicated_resources = BatchDedicatedResources(
            machine_spec=machine_spec,
            starting_replica_count=1, # Can be 0 for serverless, or 1+ for dedicated
            max_replica_count=5       # Adjust based on expected load and budget
        )

        # Construct the BatchPredictionJob object
        # NO notification_spec here
        batch_prediction_job_resource = BatchPredictionJob(
            display_name=job_display_name,
            model=model_resource_name,
            input_config=input_config,
            output_config=output_config,
            dedicated_resources=dedicated_resources,
            # Ensure this service account has necessary permissions (Vertex AI User, Storage R/W)
            service_account=f"fire-app-vm-service-account@{gcp_project_id}.iam.gserviceaccount.com",
            # generate_explanation=False, # Set to True if model explanations are needed and configured
            # model_parameters={}, # If your model accepts parameters
        )

        # Create the JobServiceClient with the regional endpoint
        client_options = {"api_endpoint": f"{gcp_region}-aiplatform.googleapis.com"}
        job_service_client = JobServiceClient(client_options=client_options)
        
        parent_path = f"projects/{gcp_project_id}/locations/{gcp_region}"

        # Create the batch prediction job
        created_job_response = job_service_client.create_batch_prediction_job(
            parent=parent_path,
            batch_prediction_job=batch_prediction_job_resource
        )

        _log_json("INFO", "Vertex AI Batch Prediction job creation request sent successfully.",
                           job_display_name=created_job_response.display_name,
                           job_resource_name=created_job_response.name,
                           job_state=str(created_job_response.state)) # Log initial state

    except ValueError as ve:
        _log_json("CRITICAL", f"A configuration or validation error occurred: {str(ve)}", error_type=type(ve).__name__)
        raise # Re-raise to signal Cloud Function failure
    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during pipeline initiation.",
                   error=str(e), error_type=type(e).__name__)
        import traceback
        _log_json("ERROR", "Traceback:", traceback_details=traceback.format_exc())
        raise e # Re-raise to signal Cloud Function failure

    _log_json("INFO", "Pipeline Initiator Cloud Function execution finished.")


# --- Local Testing Entrypoint ---
if __name__ == "__main__":
    print("--- Running local test for Pipeline Initiator Cloud Function ---")

    # Set environment variables for local testing
    # Ensure these are appropriate for your test environment
    os.environ["GCP_PROJECT_ID"] = os.environ.get("GCP_PROJECT_ID", "haryo-kebakaran")
    os.environ["GCP_REGION"] = os.environ.get("GCP_REGION", "asia-southeast2")
    os.environ["FIRMS_API_KEY"] = os.environ.get("FIRMS_API_KEY", "0331973a7ee830ca7f026493faaa367a") # Replace if needed
    os.environ["VERTEX_AI_MODEL_NAME"] = os.environ.get("VERTEX_AI_MODEL_NAME", "dummy_wildfire_detector_v1")
    # Note: VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME is not used by this script to configure the job.
    # The Pub/Sub topic (e.g., vertex-job-completion-topic) should be the destination for a Cloud Logging Sink.

    if os.environ.get("FIRMS_API_KEY") == "YOUR_FIRMS_API_KEY_PLACEHOLDER": # Example placeholder check
        print("WARNING: FIRMS_API_KEY is set to a placeholder. Real FIRMS data retrieval will likely fail.")

    if GCS_BUCKET_NAME == "fire-app-bucket": # Or your actual bucket name
        print(f"Using configured GCS_BUCKET_NAME: {GCS_BUCKET_NAME} for local test.")
    else:
        print(f"ERROR: GCS_BUCKET_NAME in src/common/config.py ('{GCS_BUCKET_NAME}') might be incorrect for local test.")

    print("Local test will NOT attempt to configure direct Pub/Sub notifications from the Vertex AI Batch Job.")
    print("A Cloud Logging Sink should be configured separately to forward job completion events to Pub/Sub.")
    print(f"Ensure the Vertex AI model '{os.environ['VERTEX_AI_MODEL_NAME']}' exists in project "
          f"'{os.environ['GCP_PROJECT_ID']}' region '{os.environ['GCP_REGION']}'.")
    print("Ensure your local environment is authenticated to GCP (e.g., `gcloud auth application-default login`).")
    print("Ensure 'fire-app-vm-service-account' exists and has necessary roles (Vertex AI User, Storage Object Admin, etc.).")
    print("Ensure Google Earth Engine API is enabled for the project and the service account has 'Earth Engine Data Writer'.")

    # Simulate a Pub/Sub trigger event for local testing
    mock_event_data_str = json.dumps({"message": "Daily trigger from Cloud Scheduler - Local Test (Log Sink for notifications)"})
    mock_event_data_b64 = base64.b64encode(mock_event_data_str.encode('utf-8')).decode('utf-8')
    mock_event = {"data": mock_event_data_b64}

    # Mock the context object
    class MockContext:
        def __init__(self, event_id="test-event-local-initiator-log-sink", timestamp=datetime.now(dt_timezone.utc).isoformat()):
            self.event_id = event_id
            self.timestamp = timestamp
            self.event_type = "google.cloud.pubsub.topic.v1.messagePublished" # Typical for Pub/Sub trigger
            self.resource = {"name": f"projects/{os.environ['GCP_PROJECT_ID']}/topics/your-scheduler-topic", "service": "pubsub.googleapis.com"}

    mock_context_obj = MockContext()

    try:
        pipeline_initiator_cloud_function(mock_event, mock_context_obj)
        print("--- Local test of Pipeline Initiator completed. Check logs, GCS, and Vertex AI for batch job. ---")
        print("--- Remember to set up a Cloud Logging Sink to trigger ResultProcessorCF. ---")
    except ValueError as ve:
        print(f"--- Local test failed due to ValueError: {ve} ---")
    except Exception as e:
        print(f"--- Local test failed with an unexpected error: {type(e).__name__} - {e} ---")
        import traceback
        traceback.print_exc()
