# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
import time
from io import BytesIO

from google.cloud import storage, aiplatform
from PIL import Image
import folium
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
FINAL_OUTPUT_GCS_PREFIX = "final_outputs/"
RETRY_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 20

# --- Initializations ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
storage_client = storage.Client()

try:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logging.info(f"Vertex AI client initialized for project '{GCP_PROJECT_ID}' and region '{GCP_REGION}'.")
except Exception as e:
    logging.critical(f"Failed to initialize Vertex AI client. Error: {e}")

def result_processor_cloud_function(event, context):
    logging.info("Result Processor function triggered.")

    if not all([GCS_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION]):
        logging.critical("Missing required environment variables.")
        return

    try:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)
        job_id = log_entry['resource']['labels']['job_id']
        logging.info(f"Extracted Vertex AI Batch Prediction job ID: {job_id}")

        job_resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/batchPredictionJobs/{job_id}"
        batch_job = aiplatform.BatchPredictionJob(job_resource_name)
        run_date = batch_job.display_name.replace('batch_prediction_', '')
        
        if not run_date:
            raise ValueError(f"Could not extract run_date from job display name: '{batch_job.display_name}'")

        gcs_output_prefix = f"incident_outputs/{run_date}/"
        logging.info(f"Processing prediction results for run_date '{run_date}' from GCS prefix: {gcs_output_prefix}")

    except Exception as e:
        logging.error(f"Could not parse trigger event or determine run details. Error: {e}", exc_info=True)
        return

    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    prediction_blobs = []
    for attempt in range(RETRY_ATTEMPTS):
        logging.info(f"Checking for prediction files... (Attempt {attempt + 1}/{RETRY_ATTEMPTS})")
        prediction_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
        if prediction_blobs:
            logging.info(f"Found {len(prediction_blobs)} prediction file(s).")
            break
        if attempt < RETRY_ATTEMPTS - 1:
            logging.warning(f"No prediction files found. Retrying in {RETRY_DELAY_SECONDS} seconds.")
            time.sleep(RETRY_DELAY_SECONDS)

    if not prediction_blobs:
        logging.error(f"No prediction result files found at prefix: {gcs_output_prefix} after {RETRY_ATTEMPTS} attempts. Exiting.")
        return

    master_input_blob_path = f"incident_inputs/{run_date}.jsonl"
    try:
        input_instance_str = bucket.blob(master_input_blob_path).download_as_string().decode('utf-8')
        input_instance = json.loads(input_instance_str)
        input_metadata = {cluster['cluster_id']: cluster for cluster in input_instance['clusters']}
    except Exception as e:
        logging.error(f"Could not read or parse the master input file: {master_input_blob_path}. Error: {e}", exc_info=True)
        return

    folium_map_data = []

    for prediction_blob in prediction_blobs:
        if 'prediction' not in prediction_blob.name:
            continue
            
        logging.info(f"Processing results file: {prediction_blob.name}")
        prediction_content_str = prediction_blob.download_as_string().decode('utf-8')
        
        for line in prediction_content_str.strip().split('\n'):
            if not line.strip(): continue

            try:
                prediction_data = json.loads(line)
                ai_detections_list = prediction_data.get('predictions')
                if not ai_detections_list: continue

                for prediction in ai_detections_list:
                    cluster_id = prediction.get("instance_id")
                    input_data = input_metadata.get(cluster_id)
                    if not input_data:
                        logging.error(f"Could not find input metadata for cluster_id '{cluster_id}'")
                        continue

                    original_image_uri = input_data['gcs_image_uri']
                    image_bbox = input_data['image_bbox']
                    
                    img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
                    image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
                    
                    visualizer = MapVisualizer()
                    final_map_image = visualizer.generate_fire_map(
                        base_image_bytes=image_bytes, 
                        image_bbox=image_bbox, 
                        ai_detections=[prediction],
                        acquisition_date_str=run_date
                    )
                    
                    img_byte_arr = BytesIO()
                    final_map_image.save(img_byte_arr, format='PNG')
                    
                    # --- MODIFICATION: Base64 encode the image for direct embedding ---
                    encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                    folium_map_data.append({
                        "cluster_id": cluster_id,
                        "latitude": (image_bbox[1] + image_bbox[3]) / 2,
                        "longitude": (image_bbox[0] + image_bbox[2]) / 2,
                        "detected": prediction.get("detected"),
                        "confidence": prediction.get("confidence", 0),
                        "encoded_png": encoded_image # Store the encoded image data
                    })

            except Exception as e:
                logging.error(f"Failed to process a prediction line: '{line}'. Error: {e}", exc_info=True)

    if folium_map_data:
        logging.info(f"Generating final interactive report for {len(folium_map_data)} clusters.")
        m = folium.Map(location=[-2.5, 118], zoom_start=5)

        for item in folium_map_data:
            # --- MODIFICATION: Use a Data URI to embed the image in the HTML ---
            image_uri = f"data:image/png;base64,{item['encoded_png']}"
            
            popup_html = f"""
            <h4>Cluster ID: {item['cluster_id']}</h4>
            <p>
              <b>AI Detection:</b> {'FIRE DETECTED' if item['detected'] else 'No Fire Detected'}<br>
              <b>Confidence:</b> {item['confidence']:.2%}
            </p>
            <img src='{image_uri}' width='400'>
            """
            
            iframe = folium.IFrame(popup_html, width=430, height=450)
            popup = folium.Popup(iframe, max_width=430)
            marker_color = 'red' if item['detected'] else 'green'
            
            folium.Marker(
                location=[item['latitude'], item['longitude']],
                popup=popup,
                tooltip=item['cluster_id'],
                icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
            ).add_to(m)

        report_filename = f"daily_interactive_report_{run_date}.html"
        local_temp_path = f"/tmp/{report_filename}"
        m.save(local_temp_path)
        
        report_blob_path = f"{FINAL_OUTPUT_GCS_PREFIX}{run_date}/{report_filename}"
        bucket.blob(report_blob_path).upload_from_filename(local_temp_path, content_type='text/html')
        
        logging.info(f"Successfully generated and uploaded interactive report to: {report_blob_path}")

    logging.info("Result Processor function finished successfully.")
