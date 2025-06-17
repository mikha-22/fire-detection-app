# src/ml_model/register_vertex_model.py
import os
from google.cloud import aiplatform
# --- ADDED: Import the storage library ---
from google.cloud import storage
from google.cloud.aiplatform.prediction import LocalModel

from predictor import WildfireHeuristicPredictor

# --- Configuration (remains the same) ---
PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
DOCKER_REPO_NAME = "wildfire-detector-repo"
IMAGE_NAME = "wildfire-heuristic-predictor"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:latest"

GCS_ARTIFACT_DIRECTORY_NAME = "models/wildfire_heuristic_predictor_v1/"
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'fire-app-bucket')
GCS_ARTIFACT_URI = f"gs://{GCS_BUCKET_NAME}/{GCS_ARTIFACT_DIRECTORY_NAME}"

MODEL_DISPLAY_NAME = "wildfire-heuristic-predictor-model"

def main():
    print("--- Starting HEURISTIC model registration process ---")

    # ... (verification and build steps are the same)
    print("Verifying 'config.json' exists for the build context...")
    local_config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(local_config_path):
        print(f"CRITICAL: 'config.json' not found at {local_config_path}. Cannot build predictor.")
        return
    print("'config.json' found.")

    print(f"--- Building Custom Prediction Routine container ---")
    local_model = LocalModel.build_cpr_model(
        ".",
        IMAGE_URI,
        predictor=WildfireHeuristicPredictor,
        requirements_path="requirements.txt"
    )
    local_model.get_serving_container_spec()

    print("\n--- Pushing container image to Artifact Registry ---")
    local_model.push_image()
    print("--- Image pushed successfully ---")

    # --- NEW STEP: Manually upload the config.json artifact before registering the model ---
    print("\n--- Manually uploading artifacts to GCS ---")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    destination_blob_name = os.path.join(GCS_ARTIFACT_DIRECTORY_NAME, "config.json")
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_config_path)
    print(f"Successfully uploaded {local_config_path} to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
    # ------------------------------------------------------------------------------------

    print("\n--- Uploading model to Vertex AI Model Registry ---")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    model = aiplatform.Model.upload(
        local_model=local_model,
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=GCS_ARTIFACT_URI,
    )

    print(f"\nModel registration for '{MODEL_DISPLAY_NAME}' submitted successfully.")
    print(f"Resource name: {model.resource_name}")
    
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/models/"
        f"{model.name}?project={PROJECT_ID}"
    )
    print(f"View in console: {console_url}")

if __name__ == "__main__":
    main()
