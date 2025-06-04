# src/ml_model/register_vertex_model.py
from google.cloud import aiplatform
import os

PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
BUCKET_NAME = "fire-app-bucket" # Your bucket

aiplatform.init(project=PROJECT_ID, location=REGION)

# Official Prebuilt Container URI for PyTorch "2.4" CPU in Asia
PREBUILT_PYTORCH_CPU_URI = "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-4:latest" 

MODEL_DISPLAY_NAME = "dummy_wildfire_prebuilt_pt24_overwrite_v1" # New distinct display name
# This is the GCS DIRECTORY where your model.mar is located.
GCS_ARTIFACT_DIRECTORY = f"gs://{BUCKET_NAME}/models/dummy_fire_detection/" 

try:
    print(f"Attempting to upload model '{MODEL_DISPLAY_NAME}' using prebuilt container in '{REGION}'...")
    print(f"Using prebuilt container URI: {PREBUILT_PYTORCH_CPU_URI}")
    print(f"Expecting model.mar in GCS artifact directory: {GCS_ARTIFACT_DIRECTORY}")

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=GCS_ARTIFACT_DIRECTORY, 
        serving_container_image_uri=PREBUILT_PYTORCH_CPU_URI,
        serving_container_predict_route="/predictions/model", 
        serving_container_health_route="/ping"
    )
    print(f"Model registration for '{MODEL_DISPLAY_NAME}' submitted successfully.")
    print(f"Resource name: {model.resource_name}")
    print(f"View in console: {model.gca_resource.url if hasattr(model, 'gca_resource') else 'Check Vertex AI Models UI'}")

except Exception as e:
    print(f"Error uploading model: {e}")
    import traceback
    traceback.print_exc()
