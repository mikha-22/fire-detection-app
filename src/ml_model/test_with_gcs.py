import os
import json
from datetime import datetime
from google.cloud import aiplatform, storage

# --- Configuration (remains the same) ---
PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
GCS_BUCKET_NAME = "fire-app-bucket"
MODEL_RESOURCE_NAME = "projects/216616339006/locations/asia-southeast2/models/4863210298397425664" 
GCS_INPUT_URI = "gs://fire-app-bucket/01_incidents_detected/2019-09-21/incidents.jsonl"


def main():
    """Runs a full end-to-end test against the deployed Vertex AI model using a real GCS file."""

    aiplatform.init(project=PROJECT_ID, location=REGION)
    storage_client = storage.Client()
    
    print(f"--- 1. Verifying input file exists ---")
    bucket_name, blob_name = GCS_INPUT_URI.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    input_blob = bucket.blob(blob_name)
    
    if not input_blob.exists():
        print(f"ERROR: The specified input file does not exist: {GCS_INPUT_URI}")
        return
    print(f"Input file verified: {GCS_INPUT_URI}")

    print("\n--- 2. Triggering Vertex AI Batch Prediction Job ---")
    
    model = aiplatform.Model(MODEL_RESOURCE_NAME)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    job_display_name = f"real_data_test_job_{timestamp}"
    gcs_output_uri_prefix = f"gs://{GCS_BUCKET_NAME}/gcs_testing_output/{job_display_name}/"

    # --- THE FIX IS HERE ---
    # We must specify the machine type for custom models.
    job = model.batch_predict(
        job_display_name=job_display_name,
        gcs_source=GCS_INPUT_URI,
        gcs_destination_prefix=gcs_output_uri_prefix,
        sync=True,
        machine_type="n1-standard-2",  # Added this required parameter
    )
    # -----------------------
    
    print(f"Batch prediction job '{job.display_name}' completed.")
    print(f"Job state: {job.state}")

    print("\n--- 3. Downloading and verifying results ---")
    
    output_location = job.output_info.gcs_output_directory
    print(f"Results are located in: {output_location}")

    prefix = output_location.replace(f"gs://{GCS_BUCKET_NAME}/", "")
    blobs = storage_client.list_blobs(GCS_BUCKET_NAME, prefix=prefix)
    result_blob = next((b for b in blobs if "prediction.results" in b.name), None)

    if not result_blob:
        print("ERROR: Could not find prediction results file in the output directory.")
        return

    result_content = result_blob.download_as_string().decode('utf-8')

    print("\n--- Parsed Predictions ---")
    for line in result_content.strip().split('\n'):
        prediction = json.loads(line)
        instance_id = prediction['instance_id']
        score = prediction['confidence_score']
        print(f"  Cluster ID: {instance_id}, Predicted Risk Score: {score:.4f}")

    print("\n--- Test complete. ---")

if __name__ == "__main__":
    main()
