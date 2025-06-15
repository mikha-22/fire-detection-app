import os
import json
import base64
import io
import logging
from datetime import datetime

import folium
import pandas as pd
from google.cloud import storage
from PIL import Image

# Assuming the 'src' directory is in the deployment package, which is set up by cloudbuild.
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
INCIDENTS_GCS_PREFIX = "incidents"
BATCH_INPUT_GCS_PREFIX = "incident_inputs"
BATCH_OUTPUT_GCS_PREFIX = "incident_outputs"
FINAL_RESULTS_GCS_PREFIX = "final_reports"

# --- Logging Setup ---
# Standard logging is good, but structured JSON is better for GCP.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
def _log_json(severity: str, message: str, **kwargs):
    """
    Prints a structured JSON log message to stdout for GCP Cloud Logging.
    """
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "ResultProcessor",
        **kwargs
    }
    # This print statement is what sends the log to Cloud Logging.
    print(json.dumps(log_entry))


def _get_gcs_blob_content(storage_client, bucket_name, blob_name):
    """Downloads and returns the content of a blob from GCS with logging."""
    gcs_path = f"gs://{bucket_name}/{blob_name}"
    _log_json("INFO", "Attempting to download file from GCS.", gcs_path=gcs_path)
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            _log_json("ERROR", "File not found at specified GCS path.", gcs_path=gcs_path)
            return None
        content = blob.download_as_string()
        _log_json("INFO", "File download successful.", gcs_path=gcs_path, bytes_downloaded=len(content))
        return content
    except Exception as e:
        _log_json("ERROR", "Failed to download file from GCS.", gcs_path=gcs_path, error=str(e), error_type=type(e).__name__)
        return None

def result_processor_cloud_function(event, context):
    """
    Cloud Function triggered by Vertex AI Batch Prediction completion.
    Consolidates data to produce a final interactive map report.
    """
    # Use context to get a unique ID for this function execution for better log tracking.
    execution_id = context.event_id if context else 'local_run'
    _log_json("INFO", "Result Processor function triggered.", trigger_event_id=execution_id)

    if not GCS_BUCKET_NAME:
        _log_json("CRITICAL", "GCS_BUCKET_NAME environment variable not set. Exiting.", execution_id=execution_id, status="failure")
        return

    storage_client = storage.Client()
    run_date_str = datetime.utcnow().strftime('%Y-%m-%d')
    _log_json("INFO", "Processing results based on current system date.", run_date=run_date_str, execution_id=execution_id)

    # 1. --- Load Original Incident Data ---
    _log_json("INFO", "Starting Step 1: Loading Original Incident Data.", execution_id=execution_id)
    incidents_blob_name = f"{INCIDENTS_GCS_PREFIX}/{run_date_str}/detected_incidents.jsonl"
    incidents_content = _get_gcs_blob_content(storage_client, GCS_BUCKET_NAME, incidents_blob_name)
    if not incidents_content:
        _log_json("CRITICAL", "Could not load incident data. Aborting process.", execution_id=execution_id, status="failure")
        return
    incidents_data = [json.loads(line) for line in incidents_content.strip().split(b'\n')]
    incidents_df = pd.DataFrame(incidents_data)
    incidents_df.rename(columns={'cluster_id': 'instance_id'}, inplace=True)
    _log_json("INFO", "Step 1 Complete: Loaded original incidents.", incident_count=len(incidents_df), execution_id=execution_id)


    # 2. --- Load Batch Prediction Input Data ---
    _log_json("INFO", "Starting Step 2: Loading Batch Prediction Input Data.", execution_id=execution_id)
    batch_input_blob_name = f"{BATCH_INPUT_GCS_PREFIX}/{run_date_str}.jsonl"
    batch_input_content = _get_gcs_blob_content(storage_client, GCS_BUCKET_NAME, batch_input_blob_name)
    if not batch_input_content:
        _log_json("CRITICAL", "Could not load batch input data. Aborting process.", execution_id=execution_id, status="failure")
        return
    batch_input_data = json.loads(batch_input_content).get("clusters", [])
    batch_input_df = pd.DataFrame(batch_input_data)
    batch_input_df.rename(columns={'cluster_id': 'instance_id'}, inplace=True)
    _log_json("INFO", "Step 2 Complete: Loaded batch prediction inputs.", input_count=len(batch_input_df), execution_id=execution_id)


    # 3. --- Load AI Prediction Results ---
    _log_json("INFO", "Starting Step 3: Loading AI Prediction Results.", execution_id=execution_id)
    prediction_results_list = []
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    prediction_prefix = f"{BATCH_OUTPUT_GCS_PREFIX}/{run_date_str}/"
    blobs_found = bucket.list_blobs(prefix=prediction_prefix)
    
    # Filter for the actual results file, which is often nested
    prediction_files = [b for b in blobs_found if "prediction.results" in b.name]

    if not prediction_files:
        _log_json("ERROR", "No prediction result files found in GCS.", search_prefix=prediction_prefix, execution_id=execution_id)
        return
        
    for blob in prediction_files:
        _log_json("INFO", "Processing prediction result file.", gcs_path=f"gs://{GCS_BUCKET_NAME}/{blob.name}", execution_id=execution_id)
        content = blob.download_as_string()
        predictions = json.loads(content).get("predictions", [])
        prediction_results_list.extend(predictions)

    if not prediction_results_list:
        _log_json("ERROR", "Prediction result files were found but contained no predictions.", search_prefix=prediction_prefix, execution_id=execution_id)
        return
    predictions_df = pd.DataFrame(prediction_results_list)
    _log_json("INFO", "Step 3 Complete: Loaded AI prediction results.", result_count=len(predictions_df), execution_id=execution_id)

    # 4. --- Merge All Data Sources ---
    _log_json("INFO", "Starting Step 4: Merging all data sources.", execution_id=execution_id)
    merged_df = pd.merge(incidents_df, batch_input_df, on='instance_id', how='inner')
    final_df = pd.merge(merged_df, predictions_df, on='instance_id', how='inner')
    _log_json("INFO", "Step 4 Complete: Successfully merged data.", record_count=len(final_df), execution_id=execution_id,
              data_columns=final_df.columns.tolist())

    if final_df.empty:
        _log_json("WARNING", "No matching records found after merging all data. Nothing to visualize. Exiting.", execution_id=execution_id, status="success_empty")
        return

    # 5. --- Generate and Upload Visuals ---
    _log_json("INFO", "Starting Step 5: Generating and uploading visuals for each incident.", execution_id=execution_id)
    visualizer = MapVisualizer()
    map_center_lat = final_df['centroid_latitude'].iloc[0]
    map_center_lon = final_df['centroid_longitude'].iloc[0]
    daily_report_map = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=7)

    for index, row in final_df.iterrows():
        incident_id = row['instance_id']
        _log_json("INFO", "Processing incident.", incident_id=incident_id, execution_id=execution_id)
        try:
            gcs_image_uri = row['gcs_image_uri']
            image_blob_name = gcs_image_uri.replace(f"gs://{GCS_BUCKET_NAME}/", "")
            image_bytes = _get_gcs_blob_content(storage_client, GCS_BUCKET_NAME, image_blob_name)
            if not image_bytes:
                _log_json("WARNING", "Skipping incident due to missing satellite image.", incident_id=incident_id, execution_id=execution_id)
                continue

            firms_hotspots_df = pd.DataFrame([f['properties'] for f in row['hotspots']])
            ai_detections = [{"detected": row['detected'], "confidence": row['confidence']}]

            map_image_pil = visualizer.generate_fire_map(
                base_image_bytes=image_bytes, image_bbox=row['image_bbox'],
                ai_detections=ai_detections, firms_hotspots_df=firms_hotspots_df,
                acquisition_date_str=run_date_str
            )

            buffer = io.BytesIO()
            map_image_pil.save(buffer, format="PNG")
            image_for_upload_bytes = buffer.getvalue()

            final_image_blob_name = f"{FINAL_RESULTS_GCS_PREFIX}/{run_date_str}/{incident_id}.png"
            bucket.blob(final_image_blob_name).upload_from_string(image_for_upload_bytes, content_type='image/png')
            _log_json("INFO", "Uploaded generated map image for incident.", incident_id=incident_id, gcs_path=f"gs://{GCS_BUCKET_NAME}/{final_image_blob_name}", execution_id=execution_id)

            encoded_image = base64.b64encode(image_for_upload_bytes).decode('utf-8')
            popup_html = f"""<h4>Incident: {incident_id}</h4><b>AI Detection:</b> {'Fire' if row['detected'] else 'No Fire'}<br><b>AI Confidence:</b> {row['confidence']:.2f}<br><b>FIRMS Hotspots:</b> {row['point_count']}<br><img src="data:image/png;base64,{encoded_image}" width="400">"""
            iframe = folium.IFrame(popup_html, width=430, height=480)
            popup = folium.Popup(iframe, max_width=430)
            
            folium.Marker(
                location=[row['centroid_latitude'], row['centroid_longitude']], popup=popup,
                tooltip=f"{incident_id} (AI: {'Fire' if row['detected'] else 'No Fire'})",
                icon=folium.Icon(color='red' if row['detected'] else 'green', icon='fire')
            ).add_to(daily_report_map)
            _log_json("INFO", "Added marker to Folium map.", incident_id=incident_id, execution_id=execution_id)

        except Exception as e:
            _log_json("ERROR", "Failed to process and visualize an incident.", incident_id=incident_id, error=str(e), error_type=type(e).__name__, execution_id=execution_id)

    _log_json("INFO", "Step 5 Complete: Finished processing all incidents.", execution_id=execution_id)

    # 6. --- Save and Upload Final Folium Map ---
    _log_json("INFO", "Starting Step 6: Saving and Uploading Final Folium Map.", execution_id=execution_id)
    final_map_html = daily_report_map.get_root().render()
    final_map_blob_name = f"{FINAL_RESULTS_GCS_PREFIX}/{run_date_str}/daily_summary_map.html"
    bucket.blob(final_map_blob_name).upload_from_string(final_map_html, content_type='text/html')

    _log_json("INFO", "Step 6 Complete: Successfully generated and uploaded the final daily summary map.",
              final_map_path=f"gs://{GCS_BUCKET_NAME}/{final_map_blob_name}", execution_id=execution_id)

    _log_json("INFO", "Result Processor function finished successfully.", execution_id=execution_id, status="success")
