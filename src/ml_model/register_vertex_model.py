# src/ml_model/register_vertex_model.py
import os
from google.cloud import aiplatform
from google.cloud import storage

# --- Configuration ---
PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
BUCKET_NAME = "fire-app-bucket"

# The GCS directory where the MAR file will be stored.
# NOTE: The directory should ONLY contain the model.mar file.
GCS_ARTIFACT_DIRECTORY_NAME = "models/wildfire_predictor_v2_mar_only/" 
GCS_ARTIFACT_DIRECTORY_URI = f"gs://{BUCKET_NAME}/{GCS_ARTIFACT_DIRECTORY_NAME}"
LOCAL_MODEL_MAR_PATH = "src/ml_model/model.mar"

# --- Model Registration Details ---
PREBUILT_PYTORCH_CPU_URI = "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-4:latest"
MODEL_DISPLAY_NAME = "wildfire-predictor-from-mar" # New name to reflect new method

# --- Main Script Logic ---
try:
    # --- 1. UPLOAD THE SINGLE MAR FILE TO GCS ---
    print("--- Starting Artifact Upload Step ---")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)

    # Following the official guide: upload ONLY the MAR file to the artifact URI.
    print(f"Uploading {LOCAL_MODEL_MAR_PATH} to {GCS_ARTIFACT_DIRECTORY_URI}...")
    blob_mar = bucket.blob(f"{GCS_ARTIFACT_DIRECTORY_NAME}model.mar")
    blob_mar.upload_from_filename(LOCAL_MODEL_MAR_PATH)
    print("--- Model archive uploaded successfully ---")

    # --- 2. REGISTER THE MODEL WITH VERTEX AI ---
    print(f"\n--- Starting Model Registration Step ---")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"Registering model '{MODEL_DISPLAY_NAME}' using artifacts from '{GCS_ARTIFACT_DIRECTORY_URI}'...")

    # The simplest upload method, with NO serving_container_args.
    # The pre-built container will read the manifest inside the MAR file.
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=GCS_ARTIFACT_DIRECTORY_URI,
        serving_container_image_uri=PREBUILT_PYTORCH_CPU_URI,
        serving_container_predict_route="/predictions/model",
        serving_container_health_route="/ping",
    )

    print(f"\nModel registration for '{MODEL_DISPLAY_NAME}' submitted successfully.")
    print(f"Resource name: {model.resource_name}")
    
    # Manually construct the console link to avoid deprecated attribute error.
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/models/"
        f"{model.name}?project={PROJECT_ID}"
    )
    print(f"View in console: {console_url}")

except FileNotFoundError as e:
    print(f"\n[ERROR] File not found: {e}. Did you run archive_model.sh first?")
except Exception as e:
    print(f"\n[ERROR] An error occurred during the process: {e}")
    import traceback
    traceback.print_exc()
