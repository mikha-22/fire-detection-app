# src/cloud_functions/result_processor/main.py

import os
import json
import base64
import logging
from io import BytesIO

from google.cloud import storage
from PIL import Image

# Import the visualizer from the shared source
from src.map_visualizer.visualizer import MapVisualizer

# --- Configuration ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "fire-app-bucket")
# This is the GCS folder where final, user-facing artifacts will be stored.
FINAL_OUTPUT_GCS_PREFIX = "final_outputs/"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
storage_client = storage.Client()

def result_processor_cloud_function(event, context):
    """
    Triggered by a Pub/Sub message from a Cloud Logging Sink when a Vertex AI
    Batch Prediction job completes. This function:
    1. Fetches the AI prediction results.
    2. Retrieves the original satellite image and its metadata.
    3. Invokes the MapVisualizer to create a final map.
    4. Saves the final map and metadata to a public-facing GCS location.
    """
    logging.info("Result Processor function triggered.")

    # 1. Parse the incoming message from the Cloud Logging sink
    try:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        log_entry = json.loads(message_data)
        
        # Extract the GCS path for the prediction output. This path is standard
        # in Vertex AI job completion logs.
        gcs_output_uri = log_entry['protoPayload']['metadata']['batchPredictionJob']['outputInfo']['gcsOutputDirectory']
        logging.info(f"Processing prediction results from: {gcs_output_uri}")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Could not parse trigger event payload or find GCS path. Error: {e}", extra={"payload": event.get('data')})
        return

    # 2. Fetch the prediction results file from GCS
    bucket_name, blob_prefix = gcs_output_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    
    # The result file is typically named 'predictions_00001.jsonl' or similar.
    # We find the first blob with 'prediction' in its name.
    prediction_blob = next(
        (b for b in bucket.list_blobs(prefix=blob_prefix) if 'prediction' in b.name),
        None
    )
    if not prediction_blob:
        logging.error(f"No prediction result file found at prefix: {gcs_output_uri}")
        return

    prediction_content = json.loads(prediction_blob.download_as_string())
    # The predictor returns a dictionary like {"predictions": [...]}, so we extract the list.
    ai_detections = prediction_content.get('predictions')

    if not ai_detections:
        logging.warning("Prediction result file was empty or malformed.")
        return
        
    # We assume one image per job, so we process the first prediction.
    prediction = ai_detections[0]
    cluster_id = prediction.get("instance_id")
    logging.info(f"Successfully parsed prediction for cluster: {cluster_id}")

    # 3. Retrieve the original image and its metadata
    # The 'ImageProcessorCF' saved the input data in a predictable location.
    input_data_blob_path = f"incident_inputs/{cluster_id}.jsonl"
    input_blob = bucket.blob(input_data_blob_path)
    
    try:
        input_data_str = input_blob.download_as_string()
        input_data = json.loads(input_data_str)
        
        # Get the image GCS path and its bounding box
        original_image_uri = input_data['gcs_image_uri']
        image_bbox = input_data['image_bbox'] # This is why the previous step was critical
        
        # Download the actual image bytes
        img_bucket_name, img_blob_name = original_image_uri.replace("gs://", "").split("/", 1)
        image_bytes = storage_client.bucket(img_bucket_name).blob(img_blob_name).download_as_bytes()
        
    except Exception as e:
        logging.error(f"Failed to retrieve original image or metadata for {cluster_id}. Error: {e}", exc_info=True)
        return

    # 4. Generate and save the final map
    try:
        visualizer = MapVisualizer()
        acquisition_date = cluster_id.split('_')[2] # Infer date from the ID, e.g., "fire_cluster_20240115_0"
        
        final_map_image = visualizer.generate_fire_map(
            base_image_bytes=image_bytes,
            image_bbox=image_bbox,
            ai_detections=ai_detections,
            firms_hotspots_df=None, # FIRMS data is not passed through; omitting for simplicity
            acquisition_date_str=acquisition_date
        )
        
        # Save map image to the final output location in GCS
        map_output_path = f"{FINAL_OUTPUT_GCS_PREFIX}{cluster_id}_map.png"
        map_blob = bucket.blob(map_output_path)
        
        # Convert PIL Image to bytes to upload
        img_byte_arr = BytesIO()
        final_map_image.save(img_byte_arr, format='PNG')
        map_blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
        
        logging.info(f"Successfully saved final map to gs://{GCS_BUCKET_NAME}/{map_output_path}")

    except Exception as e:
        logging.error(f"Failed to generate or save map for {cluster_id}. Error: {e}", exc_info=True)
    
    logging.info("Result Processor function finished successfully.")
