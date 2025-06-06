# src/ml_model/register_vertex_model.py
import os
from google.cloud import aiplatform
from google.cloud import storage

# --- Configuration ---
PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
BUCKET_NAME = "fire-app-bucket"

# --- Define local and GCS paths ---
GCS_ARTIFACT_DIRECTORY_NAME = "models/wildfire_predictor_v1/" 
GCS_ARTIFACT_DIRECTORY_URI = f"gs://{BUCKET_NAME}/{GCS_ARTIFACT_DIRECTORY_NAME}"

# Local paths to the files that need to be uploaded
LOCAL_MODEL_MAR_PATH = "src/ml_model/model.mar"
LOCAL_HANDLER_PATH = "src/ml_model/handler.py"
LOCAL_REQUIREMENTS_PATH = "src/ml_model/requirements.txt"
LOCAL_CONFIG_PROPERTIES_PATH = "src/ml_model/config.properties" 

# --- Model Registration Details ---
PREBUILT_PYTORCH_CPU_URI = "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-4:latest"
MODEL_DISPLAY_NAME = "wildfire-predictor-with-custom-handler"

# --- Main Script Logic ---
try:
    # --- 1. UPLOAD ARTIFACTS TO GCS ---
    print("--- Starting Artifact Upload Step ---")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)

    # Upload all four required files
    print(f"Uploading {LOCAL_MODEL_MAR_PATH}...")
    bucket.blob(f"{GCS_ARTIFACT_DIRECTORY_NAME}model.mar").upload_from_filename(LOCAL_MODEL_MAR_PATH)

    print(f"Uploading {LOCAL_HANDLER_PATH}...")
    bucket.blob(f"{GCS_ARTIFACT_DIRECTORY_NAME}handler.py").upload_from_filename(LOCAL_HANDLER_PATH)

    print(f"Uploading {LOCAL_REQUIREMENTS_PATH}...")
    bucket.blob(f"{GCS_ARTIFACT_DIRECTORY_NAME}requirements.txt").upload_from_filename(LOCAL_REQUIREMENTS_PATH)
    
    print(f"Uploading {LOCAL_CONFIG_PROPERTIES_PATH}...")
    bucket.blob(f"{GCS_ARTIFACT_DIRECTORY_NAME}config.properties").upload_from_filename(LOCAL_CONFIG_PROPERTIES_PATH)

    print("--- All artifacts uploaded successfully ---")

    # --- 2. REGISTER THE MODEL WITH VERTEX AI ---
    print(f"\n--- Starting Model Registration Step ---")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"Registering model '{MODEL_DISPLAY_NAME}' using artifacts from '{GCS_ARTIFACT_DIRECTORY_URI}'...")

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=GCS_ARTIFACT_DIRECTORY_URI,
        serving_container_image_uri=PREBUILT_PYTORCH_CPU_URI,
        serving_container_predict_route="/predictions/model",
        serving_container_health_route="/ping",
        # CORRECTED: Only override the ts-config. The container finds handler.py automatically.
        serving_container_args=[
            "--ts-config=config.properties"
        ]
    )

    print(f"\nModel registration for '{MODEL_DISPLAY_NAME}' submitted successfully.")
    print(f"Resource name: {model.resource_name}")
    
    # CORRECTED: Manually construct the console link to avoid deprecated attribute error.
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/models/"
        f"{model.name}?project={PROJECT_ID}"
    )
    print(f"View in console: {console_url}")

except FileNotFoundError as e:
    print(f"\n[ERROR] File not found: {e}. Check local file paths.")
except Exception as e:
    print(f"\n[ERROR] An error occurred during the process: {e}")
    import traceback
    traceback.print_exc()
