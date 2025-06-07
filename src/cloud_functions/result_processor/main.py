# src/cloud_functions/result_processor/main.py

import os
import logging # Standard logging
import json
import io # For byte streams (map image)
import pandas as pd # For FIRMS data, though AI results are JSONL
from datetime import datetime, timedelta # Used for timestamps, date parsing
from google.cloud import storage # Correct GCS client
import base64 # Required for decoding Pub/Sub messages from Vertex AI
from typing import Dict, Any, List, Tuple, Optional # Good type hinting

# Import components from our project structure
from src.common.config import MONITORED_REGIONS, GCS_BUCKET_NAME
from src.firms_data_retriever.retriever import FirmsDataRetriever, FIRMS_API_BASE_URL, FIRMS_SENSORS
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration for Cloud Function ---
GCS_FINAL_MAPS_DIR = "final_outputs/maps/"
GCS_METADATA_DIR = "final_outputs/metadata/"
FINAL_STATUS_FILENAME = "wildfire_status_latest.json"

# --- Logging Setup ---
logger = logging.getLogger(__name__)

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "ResultProcessorCF",
        **kwargs
    }
    print(json.dumps(log_entry))


def _download_gcs_blob_as_bytes(storage_client: storage.Client, bucket_name: str, blob_name: str) -> Optional[bytes]:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        if not blob.exists():
            _log_json("WARNING", "GCS blob does not exist, cannot download.", bucket=bucket_name, blob_name=blob_name)
            return None
        contents = blob.download_as_bytes(timeout=30)
        _log_json("INFO", "Successfully downloaded GCS blob.", bucket=bucket_name, blob_name=blob_name, size_bytes=len(contents))
        return contents
    except Exception as e:
        _log_json("ERROR", f"Failed to download GCS blob '{blob_name}' from bucket '{bucket_name}'.",
                   error=str(e), error_type=type(e).__name__, bucket=bucket_name, blob_name=blob_name)
        return None

def _upload_gcs_blob_from_bytes(storage_client: storage.Client, bucket_name: str, blob_name: str, data_bytes: bytes, content_type: str = "application/octet-stream"):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        blob.upload_from_string(data_bytes, content_type=content_type, timeout=60)
        _log_json("INFO", "Successfully uploaded GCS blob.", bucket=bucket_name, blob_name=blob_name, size_bytes=len(data_bytes))
    except Exception as e:
        _log_json("ERROR", f"Failed to upload GCS blob to '{blob_name}' in bucket '{bucket_name}'.",
                   error=str(e), error_type=type(e).__name__, bucket=bucket_name, blob_name=blob_name)
        raise

def _parse_vertex_ai_batch_output(storage_client: storage.Client, output_gcs_uri_prefix: str) -> Dict[str, Dict[str, Any]]:
    prediction_results: Dict[str, Dict[str, Any]] = {}
    
    if not output_gcs_uri_prefix.startswith("gs://"):
        _log_json("ERROR", "Invalid GCS output URI prefix format.", uri=output_gcs_uri_prefix)
        return {}
    
    path_parts = output_gcs_uri_prefix.replace("gs://", "").split("/", 1)
    if len(path_parts) < 2:
        _log_json("ERROR", "Invalid GCS output URI prefix, missing path.", uri=output_gcs_uri_prefix)
        return {}

    bucket_name = path_parts[0]
    prefix = path_parts[1] 
    
    bucket = storage_client.bucket(bucket_name)
    _log_json("INFO", "Searching for Vertex AI batch prediction output files.", bucket=bucket_name, prefix=prefix)
    
    blobs = bucket.list_blobs(prefix=prefix)
    
    found_predictions_file = False
    for blob in blobs:
        # *** THE FIX: This check is now less strict to match the actual file name ***
        if "prediction.results" in blob.name: # CORRECTED, more robust check
            found_predictions_file = True
            _log_json("INFO", "Found Vertex AI predictions file.", blob_name=blob.name)
            
            predictions_bytes = _download_gcs_blob_as_bytes(storage_client, bucket_name, blob.name)
            if predictions_bytes:
                try:
                    for line_number, line in enumerate(predictions_bytes.decode('utf-8').splitlines()):
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            instance_data = entry.get("instance", {}) 
                            prediction_data_from_handler = entry.get("prediction", {})
                            instance_id_from_vertex_instance_field = instance_data.get("instance_id")

                            if instance_id_from_vertex_instance_field:
                                prediction_results[instance_id_from_vertex_instance_field] = {
                                    **instance_data,
                                    "ai_model_output": prediction_data_from_handler
                                }
                            else:
                                _log_json("WARNING", "Skipping batch output line: 'instance_id' missing.", line_number=line_number + 1)
                        except json.JSONDecodeError as jde:
                            _log_json("ERROR", f"Failed to parse JSONL line: {jde}", line_number=line_number + 1)
                        except Exception as e_line:
                            _log_json("ERROR", f"Unexpected error processing line: {e_line}", line_number=line_number + 1)
                except UnicodeDecodeError as ude:
                     _log_json("ERROR", f"Failed to decode predictions file as UTF-8: {ude}", blob_name=blob.name)
    
    if not found_predictions_file:
        _log_json("WARNING", "No 'prediction.results' files found in batch output directory.", prefix=prefix)

    _log_json("INFO", "Finished parsing Vertex AI batch prediction output.", total_results_parsed=len(prediction_results))
    return prediction_results


# Main Cloud Function entry point
def result_processor_cloud_function(event: Dict, context: Dict):
    _log_json("INFO", "Result Processor Cloud Function triggered.")

    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    firms_api_key = os.environ.get("FIRMS_API_KEY")

    if not all([gcp_project_id, firms_api_key, GCS_BUCKET_NAME]):
        missing_vars = [var for var, val in {"GCP_PROJECT_ID": gcp_project_id, "FIRMS_API_KEY": firms_api_key, "GCS_BUCKET_NAME": GCS_BUCKET_NAME}.items() if not val]
        _log_json("CRITICAL", "Missing required configurations.", missing_variables=missing_vars)
        raise ValueError(f"Missing required configurations: {', '.join(missing_vars)}")

    storage_client = storage.Client(project=gcp_project_id)

    if 'data' not in event:
        _log_json("ERROR", "Pub/Sub message 'data' field is missing.")
        raise ValueError("Invalid Pub/Sub message: 'data' field missing.")
    
    try:
        message_data_str = base64.b64decode(event['data']).decode('utf-8')
        message_data = json.loads(message_data_str)
        payload = message_data.get("payload", message_data)
        job_resource_name = payload.get("batchPredictionJob", {}).get("name", "UnknownJob")
        job_state = payload.get("jobState") or payload.get("batchPredictionJob", {}).get("state", "JOB_STATE_UNSPECIFIED")
        output_info = payload.get("batchPredictionJob", {}).get("outputInfo", {})
        output_gcs_uri_prefix = output_info.get("gcsOutputDirectory") or output_info.get("outputUriPrefix")

        _log_json("INFO", "Vertex AI Batch Job notification received.", job_name=job_resource_name, job_state=job_state, output_gcs_prefix=output_gcs_uri_prefix)

        if job_state != 'JOB_STATE_SUCCEEDED':
            _log_json("WARNING", f"Job '{job_resource_name}' did not succeed. State: {job_state}. Skipping.", job_name=job_resource_name)
            return

        if not output_gcs_uri_prefix:
            _log_json("ERROR", "GCS URI prefix missing from notification.", job_name=job_resource_name)
            raise ValueError("Missing GCS URI prefix.")

        report_acquisition_date_str = "UnknownDate"

        all_ai_predictions_by_instance_id = _parse_vertex_ai_batch_output(storage_client, output_gcs_uri_prefix)
        
        if not all_ai_predictions_by_instance_id:
            _log_json("WARNING", "No AI predictions found or parsed.")
        else:
            first_instance_id = next(iter(all_ai_predictions_by_instance_id.keys()), None)
            if first_instance_id and '_' in first_instance_id:
                try:
                    date_part = first_instance_id.split('_')[-1]
                    report_acquisition_date_str = datetime.strptime(date_part, '%Y%m%d').strftime('%Y-%m-%d')
                    _log_json("INFO", f"Derived report date from AI results: {report_acquisition_date_str}")
                except (ValueError, IndexError):
                    _log_json("WARNING", "Could not parse date from first AI instance_id.", first_instance_id=first_instance_id)

        firms_retriever = FirmsDataRetriever(api_key=firms_api_key, base_url=FIRMS_API_BASE_URL, sensors=FIRMS_SENSORS)
        relevant_firms_df = firms_retriever.get_and_filter_firms_data(MONITORED_REGIONS)
        _log_json("INFO", "FIRMS hotspots retrieved.", count=len(relevant_firms_df))

        map_visualizer = MapVisualizer()
        final_monitored_areas_status: List[Dict[str, Any]] = []
        overall_fires_detected_in_report = False

        for region_config in MONITORED_REGIONS:
            region_id = region_config["id"]
            current_region_acquisition_date_str = report_acquisition_date_str
            # ... [rest of the loop is the same, no changes needed there]
            instance_id_for_region_lookup = f"{region_id}_{current_region_acquisition_date_str.replace('-', '')}" if current_region_acquisition_date_str != "UnknownDate" else None

            area_status: Dict[str, Any] = {
                "area_id": region_id,
                "area_name": region_config.get("name", "N/A"),
                "status": "DATA_UNAVAILABLE",
                "acquisition_date": current_region_acquisition_date_str,
                "gcs_map_image_path": None,
                "gcs_source_image_path": None,
                "ai_prediction_summary": {"detected": False, "confidence": 0.0, "details": "N/A"},
                "firms_hotspot_count_in_area": 0,
                "last_updated_utc": datetime.utcnow().isoformat() + "Z",
                "error_details": None
            }

            ai_prediction_for_this_region = None
            if instance_id_for_region_lookup:
                ai_prediction_for_this_region = all_ai_predictions_by_instance_id.get(instance_id_for_region_lookup)

            if ai_prediction_for_this_region:
                original_instance_data = ai_prediction_for_this_region
                model_output = original_instance_data.get("ai_model_output", {})
                area_status["gcs_source_image_path"] = original_instance_data.get("gcs_image_uri")
                
                if model_output.get("error_message"):
                    area_status["status"] = "PROCESSING_ERROR"
                    area_status["error_details"] = f"AI Model Error: {model_output['error_message']}"
                else:
                    area_status["ai_prediction_summary"] = {
                        "detected": model_output.get("detected", False),
                        "confidence": model_output.get("confidence", 0.0),
                        "details": model_output.get("detection_details", "AI processed")
                    }
                    area_status["status"] = "NO_FIRE_DETECTED"
                    if area_status["ai_prediction_summary"]["detected"]:
                        area_status["status"] = "FIRE_DETECTED"
                        overall_fires_detected_in_report = True
                
                if area_status["gcs_source_image_path"]:
                    source_img_bucket_name, source_img_blob_name = area_status["gcs_source_image_path"].replace("gs://", "").split("/", 1)
                    source_image_bytes = _download_gcs_blob_as_bytes(storage_client, source_img_bucket_name, source_img_blob_name)
                    
                    if source_image_bytes:
                        min_lon, min_lat, max_lon, max_lat = region_config["bbox"]
                        region_firms_for_map_df = relevant_firms_df[(relevant_firms_df['latitude'] >= min_lat) & (relevant_firms_df['latitude'] <= max_lat) & (relevant_firms_df['longitude'] >= min_lon) & (relevant_firms_df['longitude'] <= max_lon)].copy()
                        area_status["firms_hotspot_count_in_area"] = len(region_firms_for_map_df)

                        try:
                            map_image_pil = map_visualizer.generate_fire_map(
                                base_image_bytes=source_image_bytes,
                                image_bbox=region_config["bbox"],
                                ai_detections=[area_status["ai_prediction_summary"]],
                                firms_hotspots_df=region_firms_for_map_df,
                                acquisition_date_str=current_region_acquisition_date_str
                            )
                            map_image_buffer = io.BytesIO()
                            map_image_pil.save(map_image_buffer, format="PNG")
                            map_image_bytes = map_image_buffer.getvalue()

                            map_filename = f"map_{region_id}_{current_region_acquisition_date_str.replace('-', '')}.png"
                            gcs_map_blob_name = f"{GCS_FINAL_MAPS_DIR}{map_filename}"
                            _upload_gcs_blob_from_bytes(storage_client, GCS_BUCKET_NAME, gcs_map_blob_name, map_image_bytes, content_type="image/png")
                            area_status["gcs_map_image_path"] = f"gs://{GCS_BUCKET_NAME}/{gcs_map_blob_name}"
                        except Exception as map_e:
                            _log_json("ERROR", f"Error generating map for {region_id}: {map_e}", region_id=region_id)
                            area_status["status"] = "PROCESSING_ERROR"
                            area_status["error_details"] = f"Map generation failed: {str(map_e)}"
                    else:
                        area_status["status"] = "DATA_UNAVAILABLE"
                        area_status["error_details"] = "Source satellite image not found."
                else:
                    area_status["status"] = "DATA_UNAVAILABLE"
                    area_status["error_details"] = "Source image URI missing from AI results."
            else:
                area_status["status"] = "DATA_UNAVAILABLE"
                area_status["error_details"] = f"AI prediction result not found for instance ID '{instance_id_for_region_lookup}'."
            
            final_monitored_areas_status.append(area_status)

        if overall_fires_detected_in_report:
            overall_status_summary_message = "Fire activity detected by AI in one or more monitored areas."
        else:
            overall_status_summary_message = "No significant fire activity detected by AI in processed areas."
        
        final_status_report = {
            "report_generated_utc": datetime.utcnow().isoformat() + "Z",
            "report_for_acquisition_date": report_acquisition_date_str,
            "overall_status_summary": overall_status_summary_message,
            "monitored_areas": final_monitored_areas_status,
            "data_sources_info": {
                "firms_last_checked_utc": datetime.utcnow().isoformat() + "Z",
                "satellite_imagery_acquisition_date": report_acquisition_date_str
            }
        }

        final_status_json_output = json.dumps(final_status_report, indent=2)
        final_status_gcs_blob_name = f"{GCS_METADATA_DIR}{FINAL_STATUS_FILENAME}"
        _upload_gcs_blob_from_bytes(storage_client, GCS_BUCKET_NAME, final_status_gcs_blob_name,
                                    final_status_json_output.encode('utf-8'), content_type="application/json")
        _log_json("INFO", "Final wildfire status report uploaded to GCS.", path=f"gs://{GCS_BUCKET_NAME}/{final_status_gcs_blob_name}")

    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred.", error=str(e), error_type=type(e).__name__)
        raise

    _log_json("INFO", "Result Processor Cloud Function execution finished.")


# --- Local Testing Entrypoint ---
if __name__ == "__main__":
    print("--- Running local test for Result Processor Cloud Function ---")

    os.environ["GCP_PROJECT_ID"] = "haryo-kebakaran"
    os.environ["FIRMS_API_KEY"] = "0331973a7ee830ca7f026493faaa367a"

    if GCS_BUCKET_NAME == "fire-app-bucket":
        print(f"Using configured GCS_BUCKET_NAME: {GCS_BUCKET_NAME} for local test.")
    else:
        print(f"ERROR: GCS_BUCKET_NAME in src/common/config.py ('{GCS_BUCKET_NAME}') seems incorrect.")
        exit(1)

    # --- Use the REAL batch prediction output from your successful job ---
    print("--- Pointing local test to REAL batch prediction output ---")
    
    # This is the path to the directory containing your prediction.results-*.jsonl file
    mock_vertex_output_dir_prefix = "gs://fire-app-bucket/vertex_ai_batch_outputs/prediction-wildfire-cpr-predictor-model-2025_06_06T22_59_08_659Z/"
    
    # We also need a mock job name for the Pub/Sub message payload
    mock_job_id = "job-that-ran-for-real-8387" # Can be anything

    # Mock Pub/Sub event for Vertex AI Batch Job completion
    mock_event_payload = {
        "payload": {
            "batchPredictionJob": {
                "name": f"projects/{os.environ['GCP_PROJECT_ID']}/locations/us-central1/batchPredictionJobs/{mock_job_id}",
                "state": "JOB_STATE_SUCCEEDED",
                "outputInfo": {
                    "gcsOutputDirectory": mock_vertex_output_dir_prefix
                }
            },
            "jobState": "JOB_STATE_SUCCEEDED"
        }
    }
    mock_event_data_str = json.dumps(mock_event_payload)
    mock_event_data_b64 = base64.b64encode(mock_event_data_str.encode('utf-8')).decode('utf-8')
    mock_event = {"data": mock_event_data_b64}

    class MockContext:
        def __init__(self, event_id="test-event-local-proc-final", timestamp=datetime.utcnow().isoformat() + "Z"):
            self.event_id = event_id
            self.timestamp = timestamp
            self.event_type = "google.cloud.pubsub.topic.v1.messagePublished"
            self.resource = {"name": "projects/your-gcp-project-id/topics/your-vertex-notifications-topic", "service": "pubsub.googleapis.com"}
    mock_context_obj = MockContext()

    try:
        result_processor_cloud_function(mock_event, mock_context_obj)
        print("--- Local test of Result Processor completed. Check logs and GCS for 'final_outputs'. ---")
    except ValueError as ve:
        print(f"--- Local test failed due to ValueError: {ve} ---")
    except Exception as e:
        print(f"--- Local test failed with an unexpected error: {e} ---")
        import traceback
        traceback.print_exc()
