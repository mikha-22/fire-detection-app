# src/cloud_functions/pipeline_initiator/main.py

import os
import logging
import json
from datetime import datetime, timedelta, timezone as dt_timezone

from google.cloud import storage
from google.cloud import aiplatform # Main SDK client initialization

# Import the specific client for submitting jobs
from google.cloud.aiplatform_v1.services.job_service import JobServiceClient

# DIRECTLY import the types we need from their specific modules
# This is the most explicit way and should work if the library is correctly installed.
from google.cloud.aiplatform_v1.types.batch_prediction_job import BatchPredictionJob
# from google.cloud.aiplatform_v1.types.job_service import NotificationSpec # Removed for try-except
from google.cloud.aiplatform_v1.types.io import GcsDestination, GcsSource
from google.cloud.aiplatform_v1.types.machine_resources import MachineSpec, BatchDedicatedResources

from typing import Dict, Any, List
import base64 # For local testing Pub/Sub message encoding

# --- Logging Setup (moved _log_json definition up for early use) ---
logger_global = logging.getLogger(__name__) # Standard logger, used by _log_json

def _log_json(severity: str, message: str, **kwargs):
    """
    Helper to log structured JSON messages to stdout for GCP Cloud Logging.
    """
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.now(dt_timezone.utc).isoformat(),
        "component": "PipelineInitiatorCF",
        **kwargs
    }
    print(json.dumps(log_entry))
# --- End Logging Setup ---


# --- Attempt to import NotificationSpec and flag its availability ---
try:
    from google.cloud.aiplatform_v1.types.job_service import NotificationSpec
    _NOTIFICATION_SPEC_AVAILABLE = True
    _log_json("INFO", "NotificationSpec imported successfully. Batch job notifications will be configured if topic is provided.")
except ImportError:
    _NOTIFICATION_SPEC_AVAILABLE = False
    NotificationSpec = None # Define it as None so later code doesn't break if it checks type
    _log_json("WARNING", "Could not import NotificationSpec from google.cloud.aiplatform_v1.types.job_service. Batch job notifications will be DISABLED. ResultProcessorCF will NOT be triggered automatically.")
except Exception as e_import:
    _NOTIFICATION_SPEC_AVAILABLE = False
    NotificationSpec = None
    _log_json("ERROR", f"An unexpected error occurred during NotificationSpec import: {str(e_import)}. Notifications DISABLED.", error_type=type(e_import).__name__)
# --- End of NotificationSpec import attempt ---


# Import components from our project structure
from src.common.config import MONITORED_REGIONS, GCS_BUCKET_NAME
from src.firms_data_retriever.retriever import FirmsDataRetriever
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer

# --- Configuration Constants ---
GCS_BATCH_INPUT_DIR = "vertex_ai_batch_inputs/"
GCS_BATCH_OUTPUT_DIR_PREFIX = "vertex_ai_batch_outputs/"

# Logger already defined globally for _log_json
# logger = logging.getLogger(__name__) # Standard logger


def _get_vertex_ai_model_resource_name(model_display_name: str, project_id: str, location: str) -> str:
    """
    Retrieves the full resource name for the latest version of a Vertex AI model
    based on its display name.
    """
    try:
        aiplatform.init(project=project_id, location=location) # Ensure initialized
        models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"',
            order_by="update_time desc"
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

    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    gcp_region = os.environ.get("GCP_REGION")
    firms_api_key = os.environ.get("FIRMS_API_KEY")
    vertex_ai_model_name = os.environ.get("VERTEX_AI_MODEL_NAME", "dummy_wildfire_detector_v1")
    vertex_notification_pubsub_topic_name = os.environ.get("VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME") # Still get it

    # Basic validation for core components
    core_env_vars = {
        "GCP_PROJECT_ID": gcp_project_id, "GCP_REGION": gcp_region,
        "FIRMS_API_KEY": firms_api_key, "VERTEX_AI_MODEL_NAME": vertex_ai_model_name,
        "GCS_BUCKET_NAME (from config)": GCS_BUCKET_NAME
    }
    missing_vars = [var for var, val in core_env_vars.items() if not val]

    # Conditional check for notification topic based on import success
    log_info_vars = {
        "project_id": gcp_project_id, "region": gcp_region,
        "gcs_bucket": GCS_BUCKET_NAME, "model_name": vertex_ai_model_name,
        "firms_key_present": bool(firms_api_key),
        "notification_spec_available": _NOTIFICATION_SPEC_AVAILABLE
    }

    if _NOTIFICATION_SPEC_AVAILABLE:
        if not vertex_notification_pubsub_topic_name:
            _log_json("WARNING", "NotificationSpec is available, but VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME environment variable is not set. Batch job notifications will be disabled.")
            # Not adding to missing_vars as it's now optional if spec is available but topic isn't.
        else:
            log_info_vars["notification_topic_name"] = vertex_notification_pubsub_topic_name
    elif vertex_notification_pubsub_topic_name: # Topic is set, but Spec is not available
        _log_json("WARNING", "VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME is set, but NotificationSpec could not be imported. Batch job notifications will be DISABLED.")
        # Don't add to missing_vars as notifications can't be configured anyway.
        log_info_vars["notification_topic_name_attempted"] = vertex_notification_pubsub_topic_name


    if missing_vars:
        _log_json("CRITICAL", "Missing one or more required environment variables or configurations. Cannot proceed.",
                   missing_variables=missing_vars)
        raise ValueError(f"Missing required configurations: {', '.join(missing_vars)}")

    _log_json("INFO", "Environment variables and configurations loaded.", **log_info_vars)


    target_date_for_data = datetime.now(dt_timezone.utc) - timedelta(days=1)
    acquisition_date_str = target_date_for_data.strftime('%Y-%m-%d')
    _log_json("INFO", f"Target acquisition date for FIRMS and Satellite Imagery: {acquisition_date_str}")

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
        _log_json("INFO", "Satellite imagery GEE export tasks initiation complete.",
                           image_tasks_initiated_count=len(gcs_image_uris_by_region_id))

        if not gcs_image_uris_by_region_id:
            _log_json("WARNING", "No satellite imagery export tasks were initiated. Skipping Vertex AI Batch Prediction.")
            return

        _log_json("INFO", "Preparing Vertex AI Batch Prediction input file (JSONL).")
        batch_input_instances: List[Dict[str, Any]] = []
        for region_id, gcs_image_uri in gcs_image_uris_by_region_id.items():
            region_metadata = next((r for r in MONITORED_REGIONS if r["id"] == region_id), None)
            if not region_metadata:
                _log_json("WARNING", f"Metadata for region_id '{region_id}' not found. Skipping instance.",
                                   region_id_from_imagery=region_id)
                continue
            region_firms_count = 0
            if not firms_hotspots_df.empty and 'monitored_region_id' in firms_hotspots_df.columns:
                region_firms_count = firms_hotspots_df[
                    firms_hotspots_df['monitored_region_id'] == region_id
                ].shape[0]
            instance_id = f"{region_id}_{acquisition_date_str.replace('-', '')}"
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
        blob.upload_from_string(jsonl_content, content_type="application/jsonl", timeout=60)
        _log_json("INFO", "Vertex AI Batch Prediction input file uploaded to GCS.",
                           input_gcs_uri=input_gcs_uri, num_instances=len(batch_input_instances))

        _log_json("INFO", "Constructing Vertex AI Batch Prediction job arguments.")
        model_resource_name = _get_vertex_ai_model_resource_name(vertex_ai_model_name, gcp_project_id, gcp_region)
        batch_job_output_gcs_prefix = f"gs://{GCS_BUCKET_NAME}/{GCS_BATCH_OUTPUT_DIR_PREFIX}"
        job_display_name = f"wildfire_detection_batch_{acquisition_date_str.replace('-', '')}_{timestamp_str}"
        
        input_config = BatchPredictionJob.InputConfig(
            instances_format="jsonl",
            gcs_source=GcsSource(uris=[input_gcs_uri])
        )
        output_config = BatchPredictionJob.OutputConfig(
            predictions_format="jsonl",
            gcs_destination=GcsDestination(output_uri_prefix=batch_job_output_gcs_prefix)
        )
        machine_spec = MachineSpec(machine_type="n1-standard-4")
        dedicated_resources = BatchDedicatedResources(
            machine_spec=machine_spec,
            starting_replica_count=1,
            max_replica_count=5
        )

        batch_prediction_job_args = {
            "display_name": job_display_name,
            "model": model_resource_name,
            "input_config": input_config,
            "output_config": output_config,
            "dedicated_resources": dedicated_resources,
            "service_account": f"fire-app-vm-service-account@{gcp_project_id}.iam.gserviceaccount.com",
        }

        # Conditionally add notification_spec
        if _NOTIFICATION_SPEC_AVAILABLE and NotificationSpec is not None and vertex_notification_pubsub_topic_name:
            pubsub_topic_resource_name = f"projects/{gcp_project_id}/topics/{vertex_notification_pubsub_topic_name}"
            notification_config = NotificationSpec(
                pubsub_topic_name=pubsub_topic_resource_name,
            )
            batch_prediction_job_args["notification_spec"] = notification_config
            _log_json("INFO", "NotificationSpec is available and topic is set. Configuring batch job notifications.")
        else:
            _log_json("WARNING", "Batch job notifications will NOT be configured (NotificationSpec unavailable or topic not set). ResultProcessorCF will need manual triggering if this is not a local test.")

        batch_prediction_job_resource = BatchPredictionJob(**batch_prediction_job_args)

        client_options = {"api_endpoint": f"{gcp_region}-aiplatform.googleapis.com"}
        job_service_client = JobServiceClient(client_options=client_options)
        parent_path = f"projects/{gcp_project_id}/locations/{gcp_region}"

        created_job_response = job_service_client.create_batch_prediction_job(
            parent=parent_path,
            batch_prediction_job=batch_prediction_job_resource
        )

        _log_json("INFO", "Vertex AI Batch Prediction job creation request sent successfully.",
                           job_display_name=created_job_response.display_name,
                           job_resource_name=created_job_response.name,
                           job_state=str(created_job_response.state))

    except ValueError as ve:
        _log_json("CRITICAL", f"A configuration or validation error occurred: {str(ve)}", error_type=type(ve).__name__)
        raise
    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during pipeline initiation.",
                   error=str(e), error_type=type(e).__name__)
        import traceback
        _log_json("ERROR", "Traceback:", traceback_details=traceback.format_exc())
        raise e

    _log_json("INFO", "Pipeline Initiator Cloud Function execution finished.")


if __name__ == "__main__":
    print("--- Running local test for Pipeline Initiator Cloud Function ---")

    # --- Local Test Setup ---
    # Ensure these environment variables are set or replaced with your actual values
    # For a real local test, you'd typically use `python-dotenv` or set them in your shell
    os.environ["GCP_PROJECT_ID"] = os.environ.get("GCP_PROJECT_ID", "haryo-kebakaran")
    os.environ["GCP_REGION"] = os.environ.get("GCP_REGION", "asia-southeast2")
    os.environ["FIRMS_API_KEY"] = os.environ.get("FIRMS_API_KEY", "your_firms_api_key_here") # Replace if needed
    os.environ["VERTEX_AI_MODEL_NAME"] = os.environ.get("VERTEX_AI_MODEL_NAME", "dummy_wildfire_detector_v1")
    # Set this if you want to test the notification path and NotificationSpec is available
    os.environ["VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME"] = os.environ.get("VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME", "vertex-job-completion-topic")
    # --- End Local Test Setup ---

    if os.environ["FIRMS_API_KEY"] == "your_firms_api_key_here":
        print("WARNING: FIRMS_API_KEY is set to a placeholder. Real FIRMS data retrieval will likely fail.")
        print("         For local testing that includes FIRMS, set this environment variable to your actual key.")


    if GCS_BUCKET_NAME == "fire-app-bucket": # Or your actual one
        print(f"Using configured GCS_BUCKET_NAME: {GCS_BUCKET_NAME} for local test.")
    else:
        print(f"ERROR: GCS_BUCKET_NAME in src/common/config.py ('{GCS_BUCKET_NAME}') might be incorrect for local test.")
        # Consider exiting if this is critical: exit(1)

    print(f"NotificationSpec availability: {_NOTIFICATION_SPEC_AVAILABLE}")
    if _NOTIFICATION_SPEC_AVAILABLE and os.environ.get("VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME"):
        print(f"Notifications will be attempted to topic: {os.environ['VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME']}")
    elif _NOTIFICATION_SPEC_AVAILABLE:
        print("NotificationSpec is available, but VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME is not set. Notifications will be skipped.")
    else:
        print("NotificationSpec is NOT available (due to import error). Notifications will be skipped.")


    print(f"Ensure the Vertex AI model '{os.environ['VERTEX_AI_MODEL_NAME']}' exists in project "
          f"'{os.environ['GCP_PROJECT_ID']}' region '{os.environ['GCP_REGION']}'.")
    if _NOTIFICATION_SPEC_AVAILABLE and os.environ.get("VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME"):
        print(f"Ensure the Pub/Sub topic '{os.environ['VERTEX_NOTIFICATION_PUBSUB_TOPIC_NAME']}' exists in project '{os.environ['GCP_PROJECT_ID']}'.")
    print("Ensure your local environment is authenticated to GCP (e.g., `gcloud auth application-default login`).")
    print("Ensure 'fire-app-vm-service-account' exists and has necessary roles (Vertex AI User, Storage Object Admin, Service Usage Consumer etc.).")
    print("Ensure Google Earth Engine API is enabled for the project and the service account has 'Earth Engine Data Writer'.")


    mock_event_data_str = json.dumps({"message": "Daily trigger from Cloud Scheduler - Local Test with optional NotificationSpec"})
    mock_event_data_b64 = base64.b64encode(mock_event_data_str.encode('utf-8')).decode('utf-8')
    mock_event = {"data": mock_event_data_b64}

    class MockContext:
        def __init__(self, event_id="test-event-local-initiator-optional-ns", timestamp=datetime.now(dt_timezone.utc).isoformat()):
            self.event_id = event_id
            self.timestamp = timestamp
            self.event_type = "google.cloud.pubsub.topic.v1.messagePublished"
            self.resource = {"name": f"projects/{os.environ['GCP_PROJECT_ID']}/topics/your-scheduler-topic", "service": "pubsub.googleapis.com"}

    mock_context_obj = MockContext()

    try:
        pipeline_initiator_cloud_function(mock_event, mock_context_obj)
        print("--- Local test of Pipeline Initiator completed. Check logs, GCS, and Vertex AI for batch job. ---")
    except ValueError as ve:
        print(f"--- Local test failed due to ValueError: {ve} ---")
    except Exception as e:
        print(f"--- Local test failed with an unexpected error: {type(e).__name__} - {e} ---")
        import traceback
        traceback.print_exc()
