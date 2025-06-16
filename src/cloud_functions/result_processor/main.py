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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db = firestore.Client(database=FIRESTORE_DATABASE_ID)
storage_client = storage.Client()

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
    manifest_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{FILE_NAMES['job_manifest']}"
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
    raw_output_prefix = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_raw_output']}/"
    prediction_files = wait_for_vertex_ai_output(bucket, raw_output_prefix)
    if not prediction_files:
        logger.error("No prediction files found. Exiting.")
        return

    _, predictions = process_vertex_ai_output(prediction_files, bucket, job_id, run_date)
    if not predictions:
        logger.error("No predictions processed. Exiting.")
        return

    # --- NEW: Group predictions by the original cluster_id ---
    grouped_predictions = defaultdict(list)
    for pred in predictions:
        if pred.get("cluster_id"):
            grouped_predictions[pred["cluster_id"]].append(pred)
    
    # --- Load supporting data ---
    incidents_path = f"{GCS_PATHS['incidents']}/{run_date}/{FILE_NAMES['incident_data']}"
    input_path = f"{GCS_PATHS['batch_jobs']}/{run_date}/{job_id}/{GCS_PATHS['batch_input']}/{FILE_NAMES['batch_input']}"
    try:
        incidents_content = bucket.blob(incidents_path).download_as_string().decode('utf-8')
        hotspots_by_cluster = {inc['cluster_id']: inc['hotspots'] for inc in [json.loads(line) for line in incidents_content.strip().split('\n')]}
        
        input_content = bucket.blob(input_path).download_as_string().decode('utf-8')
        input_metadata = {inst['instance_id']: inst['clusters'][0] for inst in [json.loads(line) for line in input_content.strip().split('\n')]}
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
        # Use the first prediction's metadata to get location info
        first_instance_id = preds_for_cluster[0]['instance_id']
        cluster_metadata = input_metadata.get(first_instance_id)
        if not cluster_metadata: continue

        overall_detected = any(p['detected'] for p in preds_for_cluster)
        marker_color = 'red' if overall_detected else 'green'

        popup_html = f"<h4>Cluster ID: {cluster_id}</h4>"
        for pred in sorted(preds_for_cluster, key=lambda p: p['instance_id']):
            instance_id = pred['instance_id']
            inst_meta = input_metadata.get(instance_id)
            if not inst_meta: continue

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
        
        folium.Marker(
            location=[cluster_metadata['image_bbox'][1], cluster_metadata['image_bbox'][0]],
            popup=popup, tooltip=cluster_id,
            icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
        ).add_to(m)

    report_filename = f"wildfire_report_{run_date}_{job_id}.html"
    report_path = f"{GCS_PATHS['reports']}/{run_date}/{report_filename}"
    local_temp_path = f"/tmp/{report_filename}"
    m.save(local_temp_path)
    bucket.blob(report_path).upload_from_filename(local_temp_path, content_type='text/html')
    logger.info(f"Report saved to: gs://{GCS_BUCKET_NAME}/{report_path}")

    logger.info("Result Processor function finished successfully.")

# Helper functions wait_for_vertex_ai_output and process_vertex_ai_output go here
# They are unchanged from the previous version.

if __name__ == "__main__":
    print("--- Running Result Processor locally ---")
    os.environ['IS_LOCAL_TEST'] = 'true'
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCP_REGION'] = 'asia-southeast2'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    result_processor_cloud_function(event=None, context=None)
    print("--- Local run of Result Processor finished ---")
