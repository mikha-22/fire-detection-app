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
# Environment variables expected in Cloud Functions:
# GCP_PROJECT_ID: Your Google Cloud Project ID
# FIRMS_API_KEY: Your NASA FIRMS API key (TODO: Fetch from Secret Manager in Phase 3)
# GCP_REGION: (Not strictly used by this CF but good for consistency if other components need it)

# GCS paths for outputs, consistent with Pipeline Initiator
# GCS_BATCH_INPUT_DIR is not used here.
# GCS_BATCH_OUTPUT_DIR_PREFIX is where Vertex AI writes predictions.
GCS_FINAL_MAPS_DIR = "final_outputs/maps/" # Grouped final outputs
GCS_SOURCE_IMAGERY_COPY_DIR = "final_outputs/source_imagery/" # Optional: if we want to keep a copy of source images with final report
GCS_METADATA_DIR = "final_outputs/metadata/" # Grouped final outputs
FINAL_STATUS_FILENAME = "wildfire_status_latest.json" # More descriptive name

# --- Logging Setup ---
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # Can be removed
logger = logging.getLogger(__name__) # Standard practice

def _log_json(severity: str, message: str, **kwargs):
    """
    Helper to log structured JSON messages to stdout for GCP Cloud Logging.
    """
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "ResultProcessorCF",
        **kwargs
    }
    print(json.dumps(log_entry))


def _download_gcs_blob_as_bytes(storage_client: storage.Client, bucket_name: str, blob_name: str) -> Optional[bytes]:
    """Downloads a blob from GCS as bytes. Pass storage_client."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        if not blob.exists(): # Check existence before download attempt
            _log_json("WARNING", "GCS blob does not exist, cannot download.", bucket=bucket_name, blob_name=blob_name)
            return None
        contents = blob.download_as_bytes(timeout=30) # Add timeout
        _log_json("INFO", "Successfully downloaded GCS blob.", bucket=bucket_name, blob_name=blob_name, size_bytes=len(contents))
        return contents
    except Exception as e: # Catch any exception during download
        _log_json("ERROR", f"Failed to download GCS blob '{blob_name}' from bucket '{bucket_name}'.",
                   error=str(e), error_type=type(e).__name__, bucket=bucket_name, blob_name=blob_name)
        return None

def _upload_gcs_blob_from_bytes(storage_client: storage.Client, bucket_name: str, blob_name: str, data_bytes: bytes, content_type: str = "application/octet-stream"):
    """Uploads bytes to a GCS blob. Pass storage_client."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        blob.upload_from_string(data_bytes, content_type=content_type, timeout=60) # Add timeout
        _log_json("INFO", "Successfully uploaded GCS blob.", bucket=bucket_name, blob_name=blob_name, size_bytes=len(data_bytes))
    except Exception as e: # Catch any exception during upload
        _log_json("ERROR", f"Failed to upload GCS blob to '{blob_name}' in bucket '{bucket_name}'.",
                   error=str(e), error_type=type(e).__name__, bucket=bucket_name, blob_name=blob_name)
        raise # Re-raise to signal failure of this critical step

def _parse_vertex_ai_batch_output(storage_client: storage.Client, output_gcs_uri_prefix: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses Vertex AI Batch Prediction output files (JSONL) from GCS.
    The output is in a directory, usually containing one or more prediction-*.jsonl files.
    
    Returns a dictionary mapping instance_id to its prediction result (merged instance + prediction).
    """
    prediction_results: Dict[str, Dict[str, Any]] = {}
    
    if not output_gcs_uri_prefix.startswith("gs://"):
        _log_json("ERROR", "Invalid GCS output URI prefix format. Must start with 'gs://'.", uri=output_gcs_uri_prefix)
        return {}
    
    path_parts = output_gcs_uri_prefix.replace("gs://", "").split("/", 1)
    if len(path_parts) < 2: # Should be at least bucket/path/
        _log_json("ERROR", "Invalid GCS output URI prefix, missing path component.", uri=output_gcs_uri_prefix)
        return {}

    bucket_name = path_parts[0]
    # The prefix from Vertex AI notification already points to the job's output directory.
    # Files are typically named like "prediction.results-00000-of-00001"
    prefix = path_parts[1] 
    
    bucket = storage_client.bucket(bucket_name)
    _log_json("INFO", "Searching for Vertex AI batch prediction output files.", bucket=bucket_name, prefix=prefix)
    
    # List all blobs matching the prefix (Vertex AI output directory)
    # Vertex AI Batch Prediction output files are typically named `prediction.results-xxxxx-of-yyyyy`
    # or `predictions_001.jsonl` etc. if sharded.
    blobs = bucket.list_blobs(prefix=prefix)
    
    found_predictions_file = False
    for blob in blobs:
        # Check for common prediction output file patterns
        if "prediction.results" in blob.name and blob.name.endswith(".jsonl"): # More robust check
            found_predictions_file = True
            _log_json("INFO", "Found Vertex AI predictions file.", blob_name=blob.name)
            
            predictions_bytes = _download_gcs_blob_as_bytes(storage_client, bucket_name, blob.name)
            if predictions_bytes:
                try:
                    for line_number, line in enumerate(predictions_bytes.decode('utf-8').splitlines()):
                        if not line.strip(): continue # Skip empty lines
                        try:
                            # Vertex AI batch prediction output format can vary slightly.
                            # Common: {"instance": {original_instance_fields...}, "prediction": {handler_output...}}
                            # Or sometimes the handler output is directly merged if the handler returns a single JSON.
                            # The handler currently returns: {"instance_id": ..., "detected": ..., ...}
                            # Vertex AI wraps this in a "prediction" field if the handler output is a simple dict.
                            # If the handler output *is* the instance_id, then it might be direct.
                            # Let's assume the handler output is under "prediction" key from Vertex.
                            entry = json.loads(line)
                            
                            # Extract instance data (from original input) and prediction data (from model handler)
                            # The `instance` field contains the original JSONL line sent to batch prediction.
                            # The `prediction` field contains the output from your TorchServe handler.
                            instance_data = entry.get("instance", {}) 
                            prediction_data_from_handler = entry.get("prediction", {})

                            # The handler itself includes "instance_id" in its output.
                            # We need to ensure we link correctly. The "instance" field from Vertex
                            # is the most reliable source for the original instance_id.
                            instance_id_from_vertex_instance_field = instance_data.get("instance_id")

                            if instance_id_from_vertex_instance_field:
                                # Merge original instance data with the model's prediction output
                                # Prediction_data_from_handler is what our custom handler returned.
                                prediction_results[instance_id_from_vertex_instance_field] = {
                                    **instance_data, # Contains original gcs_image_uri, region_metadata, etc.
                                    "ai_model_output": prediction_data_from_handler # Contains detected, confidence, etc.
                                }
                            else:
                                _log_json("WARNING", "Skipping batch output line: 'instance_id' missing in 'instance' field.",
                                          line_number=line_number + 1, line_snippet=line[:200])
                        except json.JSONDecodeError as jde:
                            _log_json("ERROR", f"Failed to parse JSONL line from predictions file: {jde}",
                                      line_number=line_number + 1, line_snippet=line[:200], blob_name=blob.name)
                        except Exception as e_line: # Catch other errors processing a line
                            _log_json("ERROR", f"Unexpected error processing predictions line: {e_line}",
                                      line_number=line_number + 1, line_snippet=line[:200], blob_name=blob.name, error_type=type(e_line).__name__)
                except UnicodeDecodeError as ude:
                     _log_json("ERROR", f"Failed to decode predictions file as UTF-8: {ude}", blob_name=blob.name)
            # else: # Already logged by _download_gcs_blob_as_bytes if download fails
            # No break here, process all prediction files in the directory if sharded.
    
    if not found_predictions_file:
        _log_json("WARNING", "No 'prediction.results*.jsonl' files found in batch output directory.", prefix=prefix)

    _log_json("INFO", "Finished parsing Vertex AI batch prediction output.", total_results_parsed=len(prediction_results))
    return prediction_results


# Main Cloud Function entry point
def result_processor_cloud_function(event: Dict, context: Dict):
    """
    Google Cloud Function that processes Vertex AI Batch Prediction results,
    generates maps, and finalizes the fire status report.
    Triggered by a Pub/Sub notification from Vertex AI Batch Job completion.
    """
    _log_json("INFO", "Result Processor Cloud Function triggered.",
               event_id=context.event_id, event_type=context.event_type,
               trigger_resource=context.resource.get("name") if isinstance(context.resource, dict) else str(context.resource),
               timestamp=context.timestamp)

    # --- 0. Configuration & Validation ---
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")
    firms_api_key = os.environ.get("FIRMS_API_KEY") # TODO: Fetch from Secret Manager

    if not all([gcp_project_id, firms_api_key, GCS_BUCKET_NAME]):
        missing_vars = [
            var for var, val in {
                "GCP_PROJECT_ID": gcp_project_id, "FIRMS_API_KEY": firms_api_key,
                "GCS_BUCKET_NAME (from config)": GCS_BUCKET_NAME
            }.items() if not val
        ]
        _log_json("CRITICAL", "Missing one or more required environment variables or configurations.",
                   missing_variables=missing_vars)
        raise ValueError(f"Missing required configurations: {', '.join(missing_vars)}")

    _log_json("INFO", "Environment variables and configurations loaded.", project_id=gcp_project_id)

    # Initialize GCS client once
    storage_client = storage.Client(project=gcp_project_id)

    # Decode the Pub/Sub message data (Vertex AI Batch Job completion notification)
    # https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1.JobState
    # https://cloud.google.com/vertex-ai/docs/predictions/get-notifications#pubsub
    if 'data' not in event:
        _log_json("ERROR", "Pub/Sub message 'data' field is missing. Expected Vertex AI job completion notification.")
        raise ValueError("Invalid Pub/Sub message: 'data' field missing.")
    
    try:
        message_data_str = base64.b64decode(event['data']).decode('utf-8')
        message_data = json.loads(message_data_str)
        
        # Extract relevant fields from the notification payload
        # The payload structure for BatchPredictionJob notifications:
        # `payload.batchPredictionJob` contains the job details.
        # `payload.jobState` contains the state.
        # For older notifications, it might be flatter. Let's try to be robust.
        
        payload = message_data.get("payload", message_data) # Handle direct or nested payload
        
        job_resource_name = payload.get("batchPredictionJob", {}).get("name") # Full job resource name
        if not job_resource_name: # Fallback for older/different notification structures
            job_resource_name = payload.get("resourceName", "UnknownJob") # From your previous version

        job_state = payload.get("jobState") # e.g., "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED"
        if not job_state: # Fallback
             job_state_from_payload = payload.get("batchPredictionJob", {}).get("state") # e.g., "JOB_STATE_SUCCEEDED"
             if job_state_from_payload:
                 job_state = job_state_from_payload
             else: # Try the flatter structure
                 job_state = payload.get("state", "JOB_STATE_UNSPECIFIED")


        # Output info is typically nested
        output_info = payload.get("batchPredictionJob", {}).get("outputInfo", {})
        if not output_info: # Fallback
            output_info = payload.get("metadata", {}).get("outputInfo", {})
            
        output_gcs_uri_prefix = output_info.get("gcsOutputDirectory") # Correct field name
        if not output_gcs_uri_prefix: # Fallback for "outputUriPrefix"
            output_gcs_uri_prefix = output_info.get("outputUriPrefix")

        _log_json("INFO", "Vertex AI Batch Job notification received.",
                   job_name=job_resource_name, job_state=job_state,
                   output_gcs_prefix=output_gcs_uri_prefix,
                   raw_notification_snippet=message_data_str[:500]) # Log snippet for debugging

        if job_state != 'JOB_STATE_SUCCEEDED': # Official state string
            _log_json("WARNING", f"Vertex AI Batch Job '{job_resource_name}' did not succeed. State: {job_state}. Skipping processing.",
                       job_name=job_resource_name, job_state=job_state)
            return # Graceful exit

        if not output_gcs_uri_prefix:
            _log_json("ERROR", "Batch prediction output GCS URI prefix is missing from job notification. Cannot proceed.",
                       job_name=job_resource_name)
            raise ValueError("Missing batch prediction output GCS URI prefix.")

        # Derive acquisition_date_str from instance_id in AI predictions later, if possible,
        # as it's more directly tied to the data processed by the AI.
        # Fallback: try to parse from output_gcs_uri_prefix (less reliable).
        # Example output_gcs_uri_prefix: gs://<bucket>/vertex_ai_batch_outputs/1234567890123456789/
        # The last part is the job ID, not a timestamp.
        # The actual prediction files are inside a subfolder like:
        # gs://<bucket>/vertex_ai_batch_outputs/<job_id>/prediction-<model_id>-<timestamp>/
        # For now, let's assume acquisition_date will come from the AI results' instance_id.
        # A placeholder for now, will be updated per region.
        report_acquisition_date_str = "UnknownDate"


        # --- 1. Fetch AI Prediction Results (from GCS) ---
        _log_json("INFO", "Fetching and parsing Vertex AI batch prediction results.")
        all_ai_predictions_by_instance_id = _parse_vertex_ai_batch_output(storage_client, output_gcs_uri_prefix)
        
        if not all_ai_predictions_by_instance_id:
            _log_json("WARNING", "No AI predictions found or parsed. Finalizing report with limited insights.")
            # Proceed to generate overall status with data unavailable or no fires.
            # We still want to generate a status file even if AI failed.
        else:
            _log_json("INFO", "Successfully parsed AI predictions.", num_predictions=len(all_ai_predictions_by_instance_id))
            # Try to get a common acquisition date from the first valid instance_id
            first_instance_id = next(iter(all_ai_predictions_by_instance_id.keys()), None)
            if first_instance_id and '_' in first_instance_id:
                try:
                    date_part = first_instance_id.split('_')[-1] # Expects format like "regionid_YYYYMMDD"
                    report_acquisition_date_str = datetime.strptime(date_part, '%Y%m%d').strftime('%Y-%m-%d')
                    _log_json("INFO", f"Derived report acquisition date from AI results: {report_acquisition_date_str}")
                except (ValueError, IndexError):
                    _log_json("WARNING", "Could not parse date from first AI instance_id. Report date will be 'UnknownDate'.",
                                       first_instance_id=first_instance_id)


        # --- 2. Re-fetch FIRMS Data (for map visualization and summary) ---
        _log_json("INFO", "Re-fetching FIRMS data for map visualization and summary.")
        firms_retriever = FirmsDataRetriever(
            api_key=firms_api_key,
            base_url=FIRMS_API_BASE_URL,
            sensors=FIRMS_SENSORS
        )
        # This gets FIRMS for the last 24h from now.
        # For map overlays, it's generally fine. For strict historical alignment with imagery,
        # FIRMS API would need to be queried for a specific date range.
        relevant_firms_df = firms_retriever.get_and_filter_firms_data(MONITORED_REGIONS)
        _log_json("INFO", "Relevant FIRMS hotspots retrieved for summary.", count=len(relevant_firms_df))


        # --- 3. Generate Maps and Assemble Final Metadata ---
        _log_json("INFO", "Generating maps and assembling final metadata for monitored areas.")
        map_visualizer = MapVisualizer()
        
        final_monitored_areas_status: List[Dict[str, Any]] = []
        overall_fires_detected_in_report = False # Renamed for clarity

        for region_config in MONITORED_REGIONS: # Iterate through configured regions
            region_id = region_config["id"]
            # Construct the expected instance_id format used by PipelineInitiatorCF
            # This requires knowing the acquisition_date used for that instance.
            # If report_acquisition_date_str is "UnknownDate", this lookup will likely fail.
            current_region_acquisition_date_str = report_acquisition_date_str # Assume same date for all for now
            if current_region_acquisition_date_str == "UnknownDate" and all_ai_predictions_by_instance_id:
                 # Try to find an instance_id specific to this region to get its date
                for inst_id in all_ai_predictions_by_instance_id.keys():
                    if inst_id.startswith(region_id + "_"):
                        try:
                            date_part = inst_id.split('_')[-1]
                            current_region_acquisition_date_str = datetime.strptime(date_part, '%Y%m%d').strftime('%Y-%m-%d')
                            _log_json("INFO", f"Derived acquisition date for region '{region_id}': {current_region_acquisition_date_str}")
                            break
                        except (ValueError, IndexError):
                            pass # Keep as UnknownDate if parsing fails

            instance_id_for_region_lookup = f"{region_id}_{current_region_acquisition_date_str.replace('-', '')}" if current_region_acquisition_date_str != "UnknownDate" else None

            area_status: Dict[str, Any] = {
                "area_id": region_id,
                "area_name": region_config.get("name", "N/A"),
                "status": "DATA_UNAVAILABLE", # Default status
                "acquisition_date": current_region_acquisition_date_str,
                "gcs_map_image_path": None,
                "gcs_source_image_path": None, # Will be populated from AI prediction instance data
                "ai_prediction_summary": {"detected": False, "confidence": 0.0, "details": "N/A"},
                "firms_hotspot_count_in_area": 0,
                "last_updated_utc": datetime.utcnow().isoformat() + "Z",
                "error_details": None
            }

            ai_prediction_for_this_region = None
            if instance_id_for_region_lookup:
                ai_prediction_for_this_region = all_ai_predictions_by_instance_id.get(instance_id_for_region_lookup)

            if ai_prediction_for_this_region:
                # `ai_prediction_for_this_region` contains:
                #   - original instance fields (gcs_image_uri, region_metadata, firms_hotspot_count_in_region)
                #   - "ai_model_output": {detected, confidence, detection_details, error_message from handler}
                
                original_instance_data = ai_prediction_for_this_region # Contains gcs_image_uri etc.
                model_output = original_instance_data.get("ai_model_output", {})

                area_status["gcs_source_image_path"] = original_instance_data.get("gcs_image_uri")
                
                if model_output.get("error_message"):
                    area_status["status"] = "PROCESSING_ERROR"
                    area_status["error_details"] = f"AI Model Handler Error: {model_output['error_message']}"
                    area_status["ai_prediction_summary"]["details"] = "AI processing error"
                    _log_json("WARNING", f"AI model handler reported an error for instance {instance_id_for_region_lookup}.",
                                       error=model_output['error_message'])
                else:
                    area_status["ai_prediction_summary"] = {
                        "detected": model_output.get("detected", False),
                        "confidence": model_output.get("confidence", 0.0),
                        "details": model_output.get("detection_details", "AI processed")
                    }
                    area_status["status"] = "NO_FIRE_DETECTED" # Default if AI processed successfully
                    if area_status["ai_prediction_summary"]["detected"]:
                        area_status["status"] = "FIRE_DETECTED"
                        overall_fires_detected_in_report = True
                        _log_json("INFO", f"AI detected fire in region.", region_id=region_id,
                                           confidence=area_status["ai_prediction_summary"]["confidence"])
                    # else: # No need to log "no fire" explicitly unless for debug
                    #     _log_json("DEBUG", f"AI detected no fire in region.", region_id=region_id)


                # Map Generation
                if area_status["gcs_source_image_path"]:
                    source_img_bucket_name, source_img_blob_name = area_status["gcs_source_image_path"].replace("gs://", "").split("/", 1)
                    source_image_bytes = _download_gcs_blob_as_bytes(storage_client, source_img_bucket_name, source_img_blob_name)
                    
                    if source_image_bytes:
                        # Filter FIRMS hotspots for the current region's bbox for map overlay
                        region_firms_for_map_df = pd.DataFrame() # Empty if no FIRMS data
                        if not relevant_firms_df.empty:
                            min_lon, min_lat, max_lon, max_lat = region_config["bbox"]
                            region_firms_for_map_df = relevant_firms_df[
                                (pd.to_numeric(relevant_firms_df['latitude'], errors='coerce') >= min_lat) &
                                (pd.to_numeric(relevant_firms_df['latitude'], errors='coerce') <= max_lat) &
                                (pd.to_numeric(relevant_firms_df['longitude'], errors='coerce') >= min_lon) &
                                (pd.to_numeric(relevant_firms_df['longitude'], errors='coerce') <= max_lon)
                            ].copy()
                        
                        area_status["firms_hotspot_count_in_area"] = len(region_firms_for_map_df)

                        try:
                            map_image_pil = map_visualizer.generate_fire_map(
                                base_image_bytes=source_image_bytes,
                                image_bbox=region_config["bbox"], # Bbox of the source image/region
                                ai_detections=[area_status["ai_prediction_summary"]], # Pass this region's AI summary
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
                            _log_json("INFO", "Map image generated and uploaded.", region_id=region_id, map_path=area_status["gcs_map_image_path"])

                        except Exception as map_e:
                            _log_json("ERROR", f"Error generating or uploading map for region {region_id}: {map_e}", region_id=region_id)
                            if area_status["status"] not in ["PROCESSING_ERROR", "DATA_UNAVAILABLE"]: # Don't overwrite more specific error
                                area_status["status"] = "PROCESSING_ERROR"
                            area_status["error_details"] = (area_status["error_details"] + "; " if area_status["error_details"] else "") + f"Map generation/upload failed: {str(map_e)}"
                    else:
                        _log_json("WARNING", f"Could not download source image for map generation: {area_status['gcs_source_image_path']}.", region_id=region_id)
                        area_status["status"] = "DATA_UNAVAILABLE"
                        area_status["error_details"] = (area_status["error_details"] + "; " if area_status["error_details"] else "") + "Source satellite image for map was not found or unreadable."
                else: # No gcs_source_image_path from AI result
                    _log_json("WARNING", f"No GCS source image URI found in AI result for region {region_id}. Cannot generate map.", region_id=region_id)
                    area_status["status"] = "DATA_UNAVAILABLE"
                    area_status["error_details"] = (area_status["error_details"] + "; " if area_status["error_details"] else "") + "Source image URI missing from AI prediction results, map not generated."
            else: # No AI prediction result for this region_id and derived date
                _log_json("WARNING", f"No AI prediction result found for instance_id '{instance_id_for_region_lookup}'.", region_id=region_id)
                area_status["status"] = "DATA_UNAVAILABLE"
                area_status["error_details"] = f"AI prediction result not found for instance ID '{instance_id_for_region_lookup}'."
            
            final_monitored_areas_status.append(area_status)

        # Determine overall status message for the report
        if overall_fires_detected_in_report:
            overall_status_summary_message = "Fire activity detected by AI in one or more monitored areas."
        elif any(area['status'] == 'FIRE_DETECTED' for area in final_monitored_areas_status): # Fallback if AI didn't run but FIRMS might indicate
            overall_status_summary_message = "Potential fire activity indicated (check FIRMS data or map details)."
        elif all(area['status'] in ['NO_FIRE_DETECTED', 'DATA_UNAVAILABLE'] for area in final_monitored_areas_status):
             overall_status_summary_message = "No significant fire activity detected by AI in processed areas."
        else: # Mix of errors, unavailable, etc.
            overall_status_summary_message = "Wildfire status processed; check individual area details."
        
        # --- 4. Assemble and Store Final wildfire_status_latest.json ---
        final_status_report = {
            "report_generated_utc": datetime.utcnow().isoformat() + "Z",
            "report_for_acquisition_date": report_acquisition_date_str, # Date of the source imagery/data
            "overall_status_summary": overall_status_summary_message,
            "monitored_areas": final_monitored_areas_status,
            "data_sources_info": { # Renamed for clarity
                "firms_last_checked_utc": datetime.utcnow().isoformat() + "Z", # When FIRMS was queried for this report
                "satellite_imagery_acquisition_date": report_acquisition_date_str
            }
        }

        final_status_json_output = json.dumps(final_status_report, indent=2)
        final_status_gcs_blob_name = f"{GCS_METADATA_DIR}{FINAL_STATUS_FILENAME}"
        _upload_gcs_blob_from_bytes(storage_client, GCS_BUCKET_NAME, final_status_gcs_blob_name,
                                    final_status_json_output.encode('utf-8'), content_type="application/json")
        _log_json("INFO", "Final wildfire status report uploaded to GCS.",
                   path=f"gs://{GCS_BUCKET_NAME}/{final_status_gcs_blob_name}")

    except ValueError as ve: # Catch config errors
        _log_json("CRITICAL", f"Configuration or Pub/Sub message parsing error: {str(ve)}", error_type=type(ve).__name__)
        raise
    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during result processing.",
                   error=str(e), error_type=type(e).__name__)
        raise e # Re-raise to signal failure

    _log_json("INFO", "Result Processor Cloud Function execution finished.")


# --- Local Testing Entrypoint ---
if __name__ == "__main__":
    print("--- Running local test for Result Processor Cloud Function ---")

    os.environ["GCP_PROJECT_ID"] = "your-gcp-project-id" # REPLACE
    os.environ["FIRMS_API_KEY"] = "YOUR_FIRMS_API_KEY" # REPLACE
    # GCS_BUCKET_NAME is from common.config

    if GCS_BUCKET_NAME == "fire-app-bucket": # Or your actual one
        print(f"Using configured GCS_BUCKET_NAME: {GCS_BUCKET_NAME} for local test.")
    else:
        print(f"ERROR: GCS_BUCKET_NAME in src/common/config.py ('{GCS_BUCKET_NAME}') seems incorrect.")
        exit(1)

    # For local testing, you need to simulate:
    # 1. A Pub/Sub message (Vertex AI job completion).
    # 2. Vertex AI batch prediction output files in GCS (e.g., prediction.results-xxxxx-of-yyyyy.jsonl).
    # 3. The source satellite images in GCS that are referenced in the batch prediction output.
    # 4. FIRMS API key needs to be valid for FIRMS data fetching.

    # --- Simulate GCS environment for local test ---
    mock_storage_client = storage.Client(project=os.environ["GCP_PROJECT_ID"])
    
    # Example: Create dummy source image and upload
    dummy_region_id = MONITORED_REGIONS[0]["id"] # e.g., "california_central"
    dummy_acq_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
    dummy_source_image_blob_name = f"raw_satellite_imagery/wildfire_imagery_{dummy_region_id}_{dummy_acq_date}.tif"
    dummy_source_image_gcs_uri = f"gs://{GCS_BUCKET_NAME}/{dummy_source_image_blob_name}"

    if not mock_storage_client.bucket(GCS_BUCKET_NAME).blob(dummy_source_image_blob_name).exists():
        try:
            from PIL import Image as PILImage # Alias to avoid conflict with module name
            dummy_pil_image = PILImage.new('RGB', (200, 200), color = (100, 150, 100))
            img_byte_arr = io.BytesIO()
            dummy_pil_image.save(img_byte_arr, format='TIFF')
            _upload_gcs_blob_from_bytes(mock_storage_client, GCS_BUCKET_NAME, dummy_source_image_blob_name,
                                        img_byte_arr.getvalue(), content_type="image/tiff")
            print(f"Uploaded dummy source image to: {dummy_source_image_gcs_uri}")
        except Exception as upload_err:
            print(f"WARNING: Could not upload dummy source image for local test: {upload_err}")


    # Example: Create dummy Vertex AI prediction output file
    dummy_instance_id = f"{dummy_region_id}_{dummy_acq_date}"
    dummy_predictions_content = json.dumps({
        "instance": {
            "instance_id": dummy_instance_id,
            "gcs_image_uri": dummy_source_image_gcs_uri,
            "region_metadata": MONITORED_REGIONS[0],
            "firms_hotspot_count_in_region": 3
        },
        "prediction": { # This is what your handler returns
            "instance_id": dummy_instance_id, # Handler includes it
            "detected": True,
            "confidence": 0.92,
            "detection_details": "Simulated fire detected by AI (local test)",
            "error_message": None
        }
    }) + "\n" # JSONL format

    # Simulate Vertex AI output path structure
    # gs://<bucket>/<prefix_from_notification>/prediction.results-00000-of-00001
    mock_job_id = "local_test_job_12345"
    mock_vertex_output_dir_prefix = f"gs://{GCS_BUCKET_NAME}/vertex_ai_batch_outputs_test/{mock_job_id}/"
    # The actual file would be inside a sub-folder like prediction-<model_id>-<timestamp>
    # For simplicity, let's put it directly under job_id for local test.
    # Or, more accurately, the notification gives the prefix *up to* the job_id.
    # The files are then inside that.
    mock_predictions_blob_name = f"vertex_ai_batch_outputs_test/{mock_job_id}/prediction.results-00000-of-00001.jsonl"

    try:
        _upload_gcs_blob_from_bytes(mock_storage_client, GCS_BUCKET_NAME, mock_predictions_blob_name,
                                    dummy_predictions_content.encode('utf-8'), content_type="application/jsonl")
        print(f"Uploaded dummy predictions file to: gs://{GCS_BUCKET_NAME}/{mock_predictions_blob_name}")
    except Exception as upload_err:
        print(f"WARNING: Could not upload dummy predictions file for local test: {upload_err}")


    # Mock Pub/Sub event for Vertex AI Batch Job completion
    mock_event_payload = {
        "payload": { # Simulating the richer payload structure
            "batchPredictionJob": {
                "name": f"projects/{os.environ['GCP_PROJECT_ID']}/locations/us-central1/batchPredictionJobs/{mock_job_id}",
                "state": "JOB_STATE_SUCCEEDED", # Critical field
                "outputInfo": {
                    "gcsOutputDirectory": mock_vertex_output_dir_prefix # Path to the directory
                }
            },
            "jobState": "JOB_STATE_SUCCEEDED" # Redundant but sometimes present
        }
    }
    mock_event_data_str = json.dumps(mock_event_payload)
    mock_event_data_b64 = base64.b64encode(mock_event_data_str.encode('utf-8')).decode('utf-8')
    mock_event = {"data": mock_event_data_b64}

    class MockContext: # Simplified context
        def __init__(self, event_id="test-event-local-proc-456", timestamp=datetime.utcnow().isoformat() + "Z"):
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
