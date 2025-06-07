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
# Setup structured logging for easy parsing in Cloud Logging
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
        This method is called once by Vertex AI when the container starts.
        Args:
            artifacts_uri (str): The GCS path to the directory containing model artifacts.
        """
        _log_json("INFO", "Starting model artifact loading.", artifacts_uri=artifacts_uri)
        
        # Determine device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log_json("INFO", f"Using device: {self._device}")

        # Initialize GCS client (guaranteed to be available as it's in the container)
        self._storage_client = storage.Client()
        _log_json("INFO", "Google Cloud Storage client initialized successfully.")

        # The artifacts_uri is a GCS directory. We need the model file within it.
        model_pt_path_gcs = os.path.join(artifacts_uri, "model.pth")
        _log_json("INFO", "Attempting to download model from GCS.", path=model_pt_path_gcs)

        try:
            # Download model weights to a temporary local file
            local_model_path = "/tmp/model.pth"
            bucket_name, blob_name = model_pt_path_gcs.replace("gs://", "").split("/", 1)
            bucket = self._storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_model_path)
            
            _log_json("INFO", "Model downloaded successfully, now loading into memory.")
            
            # Load the model
            self._model = DummyFireDetectionModel()
            # FIX 1: Added weights_only=True for security and to resolve the FutureWarning.
            self._model.load_state_dict(torch.load(local_model_path, map_location=self._device, weights_only=True))
            self._model.to(self._device)
            self._model.eval()

            _log_json("INFO", "Model loaded and ready for prediction.")

        except Exception as e:
            _log_json("CRITICAL", "Failed to load model artifacts.", error=str(e), error_type=type(e).__name__)
            raise RuntimeError(f"Failed to load model: {e}")

    # FIX 2: Correct the data flow to pass instance context through the prediction chain.

    def preprocess(self, prediction_input: Dict[str, Any]) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Preprocesses a single prediction instance.
        Now returns the original input along with the tensor to preserve context.
        """
        # Add a debug log to see exactly what is being received.
        _log_json("DEBUG", "Received instance for preprocessing.", instance_data=prediction_input)

        gcs_image_uri = prediction_input.get("gcs_image_uri")
        instance_id = prediction_input.get("instance_id", "unknown")
        
        if not gcs_image_uri:
            _log_json("ERROR", "Missing 'gcs_image_uri' in prediction instance.", instance_id=instance_id, received_keys=list(prediction_input.keys()))
            raise ValueError(f"Missing 'gcs_image_uri' for instance {instance_id}")

        try:
            bucket_name, blob_name = gcs_image_uri.replace("gs://", "").split("/", 1)
            blob = self._storage_client.bucket(bucket_name).blob(blob_name)
            image_bytes = blob.download_as_bytes()
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            transformed_image = MODEL_INPUT_TRANSFORMS(image)
            
            # Return the original input alongside the processed tensor
            return (prediction_input, transformed_image)
            
        except Exception as e:
            _log_json("ERROR", "Preprocessing failed for instance.", instance_id=instance_id, error=str(e))
            raise ValueError(f"Failed to preprocess instance {instance_id}: {e}")

    def predict(self, instances: List[Tuple[Dict[str, Any], torch.Tensor]]) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
        """
        Performs prediction on a batch of preprocessed instances.
        It now expects tuples, processes the tensors, and re-packs the results.
        """
        # Unpack the original inputs and the tensors
        original_inputs, tensors = zip(*instances)
        
        batch_to_infer = torch.stack(tensors).to(self._device)
        _log_json("INFO", "Performing inference on a batch.", count=len(tensors), shape=str(batch_to_infer.shape))
        
        with torch.no_grad():
            prediction_outputs = self._model(batch_to_infer)

        # Re-pack each original input with its corresponding prediction output tensor
        return list(zip(original_inputs, prediction_outputs))

    def postprocess(self, prediction_results: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Postprocesses the model's prediction results.
        It now has access to the original instance data for each result.
        """
        final_predictions = []
        for original_input, inference_output in prediction_results:
            probabilities = F.softmax(inference_output.unsqueeze(0), dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            is_detected = (predicted_class == 1)
            confidence = probabilities[0][predicted_class].item()

            # Now we can include the instance_id in the output!
            final_predictions.append({
                "instance_id": original_input.get("instance_id", "unknown"),
                "detected": bool(is_detected),
                "confidence": float(confidence),
                "detection_details": "Fire detected by AI model" if is_detected else "No fire detected by AI model",
                "error_message": None # This is what the ResultProcessor expects
            })
            
        return {"predictions": final_predictions}
