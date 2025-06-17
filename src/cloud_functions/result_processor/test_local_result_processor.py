import os
import json
import base64
import logging
import time
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
from google.cloud import firestore, storage, aiplatform
from PIL import Image
import folium
from src.map_visualizer.visualizer import MapVisualizer
from src.common.config import GCS_PATHS, FILE_NAMES

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
FIRESTORE_DATABASE_ID = "fire-app-firestore-db"
POLLING_TIMEOUT_SECONDS = 600
POLLING_INTERVAL_SECONDS = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db = firestore.Client(database=FIRESTORE_DATABASE_ID)
storage_client = storage.Client()

def wait_for_vertex_ai_output(bucket, prefix):
    logger.info(f"Waiting for prediction files with prefix: gs://{bucket.name}/{prefix}")
    start_time = time.time()
    while time.time() - start_time < POLLING_TIMEOUT_SECONDS:
        blobs = list(bucket.list_blobs(prefix=prefix))
        prediction_files = [b for b in blobs if "prediction.results" in b.name]
        if prediction_files:
            logger.info(f"Found {len(prediction_files)} prediction result files.")
            return prediction_files
        logger.info(f"No prediction files found yet. Waiting {POLLING_INTERVAL_SECONDS}s...")
        time.sleep(POLLING_INTERVAL_SECONDS)
    logger.error("Timed out waiting for Vertex AI prediction files.")
    return []

def process_vertex_ai_output(prediction_files, bucket, job_id, run_date):
    all_predictions = []
    logger.info("Processing prediction output files...")
    for blob in prediction_files:
        try:
            content = blob.download_as_string().decode('utf-8')
            for line in content.strip().split('\n'):
                if line:
                    prediction = json.loads(line)
                    all_predictions.append(prediction)
        except Exception as e:
            logger.error(f"Failed to process file {blob.name}: {e}")

    logger.info(f"Successfully processed {len(all_predictions)} predictions.")
    return bucket, all_predictions

def result_processor_cloud_function(event, context):
    logger.info("Result Processor function triggered.")

    if not os.environ.get("IS_LOCAL_TEST"):
        doc_ref = db.collection('processed_events').document(context.event_id)
        if doc_ref.get().exists:
            logger.warning(f"Duplicate result processing event: {context.event_id}. Skipping.")
            return
        else:
            doc_ref.set({'processed_at': datetime.now(timezone.utc)})

    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    job_id = None
    
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    manifest_path = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{FILE_NAMES['job_manifest']}"
    try:
        manifest = json.loads(bucket.blob(manifest_path).download_as_string())
        if manifest.get("jobs"):
            job_id = sorted(manifest["jobs"], key=lambda x: x["created_at"], reverse=True)[0]['job_id']
    except Exception as e:
        logger.error(f"Failed to load manifest or find jobs: {e}")
        return
    
    if not job_id:
        logger.error("Could not determine job_id to process.")
        return

    logger.info(f"Processing results for run_date: {run_date}, job_id: {job_id}")

    raw_output_prefix = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{GCS_PATHS['JOB_RAW_OUTPUT']}/"
    prediction_files = wait_for_vertex_ai_output(bucket, raw_output_prefix)
    if not prediction_files:
        logger.error("No prediction files found. Exiting.")
        return

    _, predictions = process_vertex_ai_output(prediction_files, bucket, job_id, run_date)
    if not predictions:
        logger.error("No predictions processed. Exiting.")
        return

    grouped_predictions = defaultdict(list)
    for pred in predictions:
        if pred.get("instance_id"):
            grouped_predictions[pred["instance_id"]].append(pred)
    
    # --- THIS SECTION IS NOW FIXED ---
    incidents_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date}/{FILE_NAMES['incident_data']}"
    input_path = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{FILE_NAMES['batch_input']}"
    try:
        incidents_content = bucket.blob(incidents_path).download_as_string().decode('utf-8')
        incidents_data = json.loads(incidents_content)
        
        # Standardize so we are always dealing with a list of incidents
        if isinstance(incidents_data, dict):
            incidents_data = [incidents_data]
            
        incidents_by_cluster = {inc['cluster_id']: inc for inc in incidents_data}
        
        # This part for input_metadata is optional for the new report but maintains consistency
        input_content = bucket.blob(input_path).download_as_string().decode('utf-8')
        input_data = [json.loads(line) for line in input_content.strip().split('\n')]
        input_metadata = {inst['cluster_id']: inst for inst in input_data}
    except Exception as e:
        logger.error(f"Failed to load supporting data: {e}")
        return
    # --- END OF FIX ---

    if not grouped_predictions:
        logger.error("No valid clusters to visualize.")
        return

    visualizer = MapVisualizer()
    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    for cluster_id, preds_for_cluster in grouped_predictions.items():
        # Use the prediction itself, which has the instance_id (cluster_id)
        incident_data = incidents_by_cluster.get(cluster_id)
        if not incident_data: 
            logger.warning(f"No incident data found for cluster {cluster_id}")
            continue

        # Use the first prediction for the main score display
        main_prediction = preds_for_cluster[0]
        score = main_prediction.get('confidence_score', 0.0)
        marker_color = 'red' if score > 0.75 else 'orange' if score > 0.5 else 'green'

        popup_html = f"<h4>Cluster ID: {cluster_id}</h4><h3>Risk Score: {score:.2f}</h3><hr>"
        popup_html += f"<b>Hotspot Count:</b> {incident_data.get('point_count', 'N/A')}<br>"
        # Add other context if available
        if incident_data.get('realtime_context'):
            popup_html += f"<b>Air Quality:</b> {incident_data['realtime_context']['air_quality']['air_quality_severity']}<br>"
            popup_html += f"<b>Humidity:</b> {incident_data['realtime_context']['weather']['relative_humidity_percent']}%<br>"

        iframe = folium.IFrame(popup_html, width=280, height=180)
        popup = folium.Popup(iframe, max_width=280)
        
        folium.Marker(
            location=[incident_data['centroid_latitude'], incident_data['centroid_longitude']],
            popup=popup, tooltip=f"{cluster_id} (Risk: {score:.2f})",
            icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
        ).add_to(m)

    report_filename = f"heuristic_wildfire_report_{run_date}_{job_id}.html"
    report_path = f"{GCS_PATHS['FINAL_REPORTS']}/{run_date}/{report_filename}"
    local_temp_path = f"/tmp/{report_filename}"
    m.save(local_temp_path)
    bucket.blob(report_path).upload_from_filename(local_temp_path, content_type='text/html')
    logger.info(f"Report saved to: gs://{GCS_BUCKET_NAME}/{report_path}")

    logger.info("Result Processor function finished successfully.")

if __name__ == "__main__":
    print("--- Running Result Processor locally ---")
    os.environ['IS_LOCAL_TEST'] = 'true'
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCP_REGION'] = 'asia-southeast2'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    result_processor_cloud_function(event=None, context=None)
    print("--- Local run of Result Processor finished ---")
