# register_vertex_model.py
from google.cloud import aiplatform
import os

PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
BUCKET_NAME = "fire-app-bucket"

aiplatform.init(project=PROJECT_ID, location=REGION)

serving_container_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/wildfire-detector-ts/wildfire-detector-ts:latest"
artifact_uri = f"gs://{BUCKET_NAME}/models/dummy_fire_detection/"

try:
    print(f"Attempting to upload model '{PROJECT_ID}' in '{REGION}'...")
    model = aiplatform.Model.upload(
        display_name="dummy_wildfire_detector_v1",
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route="/predictions/model",
        serving_container_health_route="/ping"
    )
    model.wait()
    print(f"Model uploaded successfully. Resource name: {model.resource_name}")
    # IMPORTANT: Copy the NEW MODEL ID from the end of the 'resource_name' output.
except Exception as e:
    print(f"Error uploading model: {e}")
    print("This might happen if the model with this display_name already exists.")
    print("If it exists, check Vertex AI -> Models to get its ID, or increment version.")
