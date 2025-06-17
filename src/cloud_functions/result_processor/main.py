# src/cloud_functions/result_processor/main.py

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
POLLING_TIMEOUT_SECONDS = 600  # 10 minutes
POLLING_INTERVAL_SECONDS = 30  # 30 seconds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db = firestore.Client(database=FIRESTORE_DATABASE_ID)
storage_client = storage.Client()

# *** START OF MISSING HELPER FUNCTIONS ***

def wait_for_vertex_ai_output(bucket, prefix):
    """Polls GCS until the Vertex AI prediction output files are found."""
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
    """Downloads, parses, and combines results from Vertex AI output files."""
    all_predictions = []
    logger.info("Processing prediction output files...")
    for blob in prediction_files:
        try:
            content = blob.download_as_string().decode('utf-8')
            for line in content.strip().split('\n'):
                if line:
                    result = json.loads(line)
                    # Extract the prediction part from the Vertex AI output
                    prediction = result.get("prediction", {})
                    # The prediction already has the correct cluster_id (without source suffix)
                    cluster_id = prediction.get("instance_id", "")
                    prediction['cluster_id'] = cluster_id
                    all_predictions.append(prediction)
        except Exception as e:
            logger.error(f"Failed to process file {blob.name}: {e}")

    logger.info(f"Successfully processed {len(all_predictions)} predictions.")
    return bucket, all_predictions

# *** END OF MISSING HELPER FUNCTIONS ***

def result_processor_cloud_function(event, context):
    logger.info("Result Processor function triggered.")

    # --- Idempotency Check ---
    if not os.environ.get("IS_LOCAL_TEST") and (not context or not context.event_id):
        logger.error("Cannot ensure idempotency: context.event_id is missing.")
        return
    if not os.environ.get("IS_LOCAL_TEST"):
        doc_ref = db.collection('processed_events').document(context.event_id)
        if doc_ref.get().exists:
            logger.warning(f"Duplicate result processing event: {context.event_id}. Skipping.")
            return
        else:
            doc_ref.set({'processed_at': datetime.now(timezone.utc)})

    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    job_id = None
    
    # Logic to find the latest job_id from the manifest
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

    # --- Wait for and process prediction files ---
    raw_output_prefix = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{GCS_PATHS['JOB_RAW_OUTPUT']}/"
    prediction_files = wait_for_vertex_ai_output(bucket, raw_output_prefix)
    if not prediction_files:
        logger.error("No prediction files found. Exiting.")
        return

    _, predictions = process_vertex_ai_output(prediction_files, bucket, job_id, run_date)
    if not predictions:
        logger.error("No predictions processed. Exiting.")
        return

    # --- Group predictions by the original cluster_id ---
    grouped_predictions = defaultdict(list)
    for pred in predictions:
        if pred.get("cluster_id"):
            grouped_predictions[pred["cluster_id"]].append(pred)
    
    # --- Load supporting data ---
    incidents_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date}/{FILE_NAMES['incident_data']}"
    input_path = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{GCS_PATHS['JOB_INPUT']}/{FILE_NAMES['batch_input']}"
    try:
        incidents_content = bucket.blob(incidents_path).download_as_string().decode('utf-8')
        hotspots_by_cluster = {inc['cluster_id']: inc['hotspots'] for inc in [json.loads(line) for line in incidents_content.strip().split('\n')]}
        
        input_content = bucket.blob(input_path).download_as_string().decode('utf-8')
        input_metadata = {}
        for line in input_content.strip().split('\n'):
            if line:
                inst = json.loads(line)
                # Map both the full instance_id and the cluster_id to the metadata
                cluster_data = inst['clusters'][0]
                input_metadata[inst['instance_id']] = cluster_data
                # Also map by cluster_id for easier lookup
                input_metadata[cluster_data['cluster_id']] = cluster_data
    except Exception as e:
        logger.error(f"Failed to load supporting data: {e}")
        return

    # --- Generate Visualizations and Report ---
    if not grouped_predictions:
        logger.error("No valid clusters to visualize.")
        return

    visualizer = MapVisualizer()
    m = folium.Map(location=[-2.5, 118], zoom_start=5)

    for cluster_id, preds_for_cluster in grouped_predictions.items():
        # Get cluster metadata - it should be stored by cluster_id now
        cluster_metadata = input_metadata.get(cluster_id)
        if not cluster_metadata: 
            logger.warning(f"No metadata found for cluster {cluster_id}")
            continue

        overall_detected = any(p['detected'] for p in preds_for_cluster)
        marker_color = 'red' if overall_detected else 'green'

        popup_html = f"<h4>Cluster ID: {cluster_id}</h4>"
        
        # Get all images for this cluster from input metadata
        cluster_images = []
        for key, meta in input_metadata.items():
            if meta.get('cluster_id') == cluster_id and key != cluster_id:
                cluster_images.append((key, meta))
        
        # Sort by source for consistent ordering
        cluster_images.sort(key=lambda x: x[1].get('source', ''))
        
        for instance_id, inst_meta in cluster_images:
            # Find the corresponding prediction
            pred = next((p for p in preds_for_cluster if p.get('instance_id') == cluster_id), None)
            if not pred:
                continue

            try:
                source = inst_meta.get('source', 'unknown')
                img_bucket, img_blob = inst_meta['gcs_image_uri'].replace("gs://", "").split("/", 1)
                image_bytes = storage_client.bucket(img_bucket).blob(img_blob).download_as_bytes()
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                
                popup_html += f"""
                <hr>
                <p><b>Source:</b> {source.capitalize()}<br>
                   <b>AI Detection:</b> {'FIRE DETECTED' if pred['detected'] else 'No Fire Detected'}<br>
                   <b>Confidence:</b> {pred['confidence']:.2%}</p>
                <img src='data:image/png;base64,{encoded_image}' width='400'>
                """
            except Exception as e:
                logger.error(f"Failed to process image for instance {instance_id}: {e}")
        
        iframe = folium.IFrame(popup_html, width=430, height=500)
        popup = folium.Popup(iframe, max_width=430)
        
        # Use the centroid of the bounding box for marker placement
        bbox = cluster_metadata['image_bbox']
        marker_lat = (bbox[1] + bbox[3]) / 2
        marker_lon = (bbox[0] + bbox[2]) / 2
        
        folium.Marker(
            location=[marker_lat, marker_lon],
            popup=popup, tooltip=cluster_id,
            icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
        ).add_to(m)

    report_filename = f"wildfire_report_{run_date}_{job_id}.html"
    report_path = f"{GCS_PATHS['FINAL_REPORTS']}/{run_date}/{report_filename}"
    local_temp_path = f"/tmp/{report_filename}"
    m.save(local_temp_path)
    bucket.blob(report_path).upload_from_filename(local_temp_path, content_type='text/html')
    logger.info(f"Report saved to: gs://{GCS_BUCKET_NAME}/{report_path}")

    logger.info("Result Processor function finished successfully.")

if __name__ == "__main__":
    print("--- Running Result Processor locally ---")
    os.environ['IS_LOCAL_TEST'] = 'true'
    # The environment variables are now set within this block for local execution
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCP_REGION'] = 'asia-southeast2'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    result_processor_cloud_function(event=None, context=None)
    print("--- Local run of Result Processor finished ---")
