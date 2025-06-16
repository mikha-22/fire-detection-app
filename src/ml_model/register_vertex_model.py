# src/ml_model/register_vertex_model.py
import os
from google.cloud import aiplatform
from google.cloud.aiplatform.prediction import LocalModel

# *** CORRECTED: IMPORT THE PREDICTOR CLASS ***
from predictor import WildfirePredictor

# --- Configuration ---
PROJECT_ID = "haryo-kebakaran"
REGION = "asia-southeast2"
# The name of your Artifact Registry repository for Docker images
DOCKER_REPO_NAME = "wildfire-detector-repo"
# The name for the Docker image we are building
IMAGE_NAME = "wildfire-cpr-predictor"
# The full URI for the image
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{DOCKER_REPO_NAME}/{IMAGE_NAME}:latest"

# GCS URI where the model.pth file will be stored.
GCS_ARTIFACT_DIRECTORY_NAME = "models/wildfire_predictor_cpr_v1/"
GCS_ARTIFACT_URI = f"gs://{os.environ.get('GCS_BUCKET_NAME', 'fire-app-bucket')}/{GCS_ARTIFACT_DIRECTORY_NAME}"

# Model display name in Vertex AI
MODEL_DISPLAY_NAME = "wildfire-cpr-predictor-model"

# --- Main Script Logic ---
def main():
    print(f"--- Starting model registration process ---")

    # --- FIX: Ensure model.pth exists BEFORE building the local_model ---
    # Check in current directory first
    model_path = "model.pth"
    if not os.path.exists(model_path):
        # Try in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model.pth")
        
        if not os.path.exists(model_path):
            print(f"model.pth not found. Running fire_detection_model.py to generate it...")
            # FIX: Use correct path to fire_detection_model.py
            fire_detection_script = os.path.join(script_dir, "fire_detection_model.py")
            if os.path.exists(fire_detection_script):
                os.system(f"python {fire_detection_script}")
            else:
                print(f"CRITICAL: Cannot find fire_detection_model.py at {fire_detection_script}")
                return
                
            if not os.path.exists(model_path):
                print(f"CRITICAL: Failed to generate model.pth. Exiting.")
                return
            print(f"Successfully generated {model_path}.")
    
    print(f"Found model at: {model_path}")
    # --------------------------------------------------------------------

    print(f"--- Building Custom Prediction Routine container ---")
    print(f"Image URI: {IMAGE_URI}")
    
    # 1. Build the custom container using CPR. Now it will include model.pth.
    # Use the current directory as the source directory
    local_model = LocalModel.build_cpr_model(
        ".",  # Current directory
        IMAGE_URI,
        # *** CORRECTED: PASS THE CLASS OBJECT, NOT A STRING ***
        predictor=WildfirePredictor,
        requirements_path="requirements.txt",
    )
    
    local_model.get_serving_container_spec()

    # 2. Push the built container to Artifact Registry
    print("\n--- Pushing container image to Artifact Registry ---")
    local_model.push_image()
    print("--- Image pushed successfully ---")

    # 3. Upload the model to Vertex AI Model Registry
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
