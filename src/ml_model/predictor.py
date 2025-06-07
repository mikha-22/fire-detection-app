# src/ml_model/predictor.py
import os
import io
import json
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from google.cloud import storage
from google.cloud.aiplatform.prediction.predictor import Predictor

# Import the model definition and transforms from our model file
from fire_detection_model import DummyFireDetectionModel, MODEL_INPUT_TRANSFORMS

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(), "message": message, "component": "CPR_Predictor", **kwargs
    }
    print(json.dumps(log_entry))

class WildfirePredictor(Predictor):
    """
    Custom Predictor for the Wildfire Detection model.
    This class handles loading the model and processing prediction requests.
    """
    
    def __init__(self):
        """Initializes the predictor instance."""
        self._storage_client = None

    def load(self, artifacts_uri: str) -> None:
        """
        Loads all model artifacts required for prediction.
        """
        _log_json("INFO", "Starting model artifact loading.", artifacts_uri=artifacts_uri)
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log_json("INFO", f"Using device: {self._device}")

        self._storage_client = storage.Client()
        _log_json("INFO", "Google Cloud Storage client initialized successfully.")

        model_pt_path_gcs = os.path.join(artifacts_uri, "model.pth")
        _log_json("INFO", "Attempting to download model from GCS.", path=model_pt_path_gcs)

        try:
            local_model_path = "/tmp/model.pth"
            bucket_name, blob_name = model_pt_path_gcs.replace("gs://", "").split("/", 1)
            bucket = self._storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_model_path)
            
            _log_json("INFO", "Model downloaded successfully, now loading into memory.")
            
            self._model = DummyFireDetectionModel()
            self._model.load_state_dict(torch.load(local_model_path, map_location=self._device, weights_only=True))
            self._model.to(self._device)
            self._model.eval()

            _log_json("INFO", "Model loaded and ready for prediction.")

        except Exception as e:
            _log_json("CRITICAL", "Failed to load model artifacts.", error=str(e), error_type=type(e).__name__)
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess(self, prediction_input: Dict[str, Any]) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Preprocesses a single prediction instance.
        This is robust to handle the wrapped `{\"instances\": [...]}` format from batch jobs.
        """
        _log_json("DEBUG", "Received instance for preprocessing.", instance_data=prediction_input)

        # --- THE FIX: Increase Pillow's image size limit ---
        # Set a new limit of 150 million pixels to handle large satellite images.
        Image.MAX_IMAGE_PIXELS = 2_000_000_000
        # ---------------------------------------------------

        if "instances" in prediction_input and isinstance(prediction_input["instances"], list) and prediction_input["instances"]:
            actual_instance = prediction_input["instances"][0]
        else:
            actual_instance = prediction_input

        gcs_image_uri = actual_instance.get("gcs_image_uri")
        instance_id = actual_instance.get("instance_id", "unknown")
        
        if not gcs_image_uri:
            _log_json("ERROR", "Missing 'gcs_image_uri' after unwrapping instance.", instance_id=instance_id, received_keys=list(actual_instance.keys()))
            raise ValueError(f"Missing 'gcs_image_uri' for instance {instance_id}")

        try:
            bucket_name, blob_name = gcs_image_uri.replace("gs://", "").split("/", 1)
            blob = self._storage_client.bucket(bucket_name).blob(blob_name)
            image_bytes = blob.download_as_bytes()
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            transformed_image = MODEL_INPUT_TRANSFORMS(image)
            
            return (actual_instance, transformed_image)
            
        except Exception as e:
            _log_json("ERROR", "Preprocessing failed for instance.", instance_id=instance_id, error=str(e))
            raise ValueError(f"Failed to preprocess instance {instance_id}: {e}")

    def predict(self, instances: Tuple[Dict[str, Any], torch.Tensor]) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
        """
        Performs prediction on a single preprocessed instance.
        The `instances` argument is now a TUPLE, not a list.
        """
        original_input, tensor = instances
        
        batch_to_infer = tensor.unsqueeze(0).to(self._device)
        _log_json("INFO", "Performing inference on a single instance.", shape=str(batch_to_infer.shape))
        
        with torch.no_grad():
            prediction_output = self._model(batch_to_infer)

        return [(original_input, prediction_output[0])]

    def postprocess(self, prediction_results: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Postprocesses the model's prediction results.
        It receives a list containing a single tuple from the predict method.
        """
        final_predictions = []
        for original_input, inference_output in prediction_results:
            probabilities = F.softmax(inference_output.unsqueeze(0), dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            is_detected = (predicted_class == 1)
            confidence = probabilities[0][predicted_class].item()

            final_predictions.append({
                "instance_id": original_input.get("instance_id", "unknown"),
                "detected": bool(is_detected),
                "confidence": float(confidence),
                "detection_details": "Fire detected by AI model" if is_detected else "No fire detected by AI model",
                "error_message": None
            })
            
        return {"predictions": final_predictions}
