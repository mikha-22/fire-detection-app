# src/ml_model/predictor.py

import os
import io
import json
import logging
import traceback # <--- ADD THIS IMPORT
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from google.cloud import storage
from google.cloud.aiplatform.prediction.predictor import Predictor

from fire_detection_model import DummyFireDetectionModel, MODEL_INPUT_TRANSFORMS

logging.basicConfig(level=logging.INFO)

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(), "message": message, "component": "CPR_Predictor", **kwargs
    }
    print(json.dumps(log_entry))


class WildfirePredictor(Predictor):
    def __init__(self):
        self._storage_client = None

    def load(self, artifacts_uri: str) -> None:
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
            # Ensure weights_only=False if your model's state_dict might contain metadata
            # For a dummy model, weights_only=True is usually fine, but keep in mind.
            self._model.load_state_dict(torch.load(local_model_path, map_location=self._device, weights_only=True))
            self._model.to(self._device)
            self._model.eval()

            _log_json("INFO", "Model loaded and ready for prediction.")

        except Exception as e:
            _log_json("CRITICAL", "Failed to load model artifacts.", error=str(e), exc_info=True) # Added exc_info=True
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess(self, prediction_input: Dict[str, Any]) -> Tuple[Dict[str, Any], torch.Tensor]:
        _log_json("DEBUG", "Received instance for preprocessing.", instance_data=prediction_input)
        Image.MAX_IMAGE_PIXELS = 2_000_000_000

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

            _log_json("INFO", f"Preprocessing successful for instance {instance_id}.", image_uri=gcs_image_uri)
            return (actual_instance, transformed_image)

        except Exception as e:
            _log_json("CRITICAL", f"Preprocessing failed for instance {instance_id}.", error=str(e), exc_info=True) # Added exc_info=True
            raise ValueError(f"Failed to preprocess instance {instance_id}: {e}")

    def predict(self, instances: Tuple[Dict[str, Any], torch.Tensor]) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
        original_input, tensor = instances

        _log_json("INFO", "Performing inference on a single instance.", shape=str(tensor.shape), instance_id=original_input.get('instance_id', 'unknown')) # Added instance_id
        batch_to_infer = tensor.unsqueeze(0).to(self._device)

        try:
            with torch.no_grad():
                prediction_output = self._model(batch_to_infer)
            _log_json("INFO", f"Inference successful for instance {original_input.get('instance_id', 'unknown')}.", output_shape=str(prediction_output.shape))
            return [(original_input, prediction_output[0])]
        except Exception as e:
            _log_json("CRITICAL", f"Prediction failed for instance {original_input.get('instance_id', 'unknown')}.", error=str(e), exc_info=True) # Added exc_info=True
            raise RuntimeError(f"Prediction failed for instance {original_input.get('instance_id', 'unknown')}: {e}")


    def postprocess(self, prediction_results: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Dict[str, List[Dict[str, Any]]]:
        final_predictions = []
        for original_input, inference_output in prediction_results:
            try:
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
                _log_json("INFO", f"Postprocessing successful for instance {original_input.get('instance_id', 'unknown')}.", detected=is_detected, confidence=confidence)
            except Exception as e:
                _log_json("CRITICAL", f"Postprocessing failed for instance {original_input.get('instance_id', 'unknown')}.", error=str(e), exc_info=True) # Added exc_info=True
                # Append a failed prediction entry so the batch job can still report results
                final_predictions.append({
                    "instance_id": original_input.get("instance_id", "unknown"),
                    "detected": False,
                    "confidence": 0.0,
                    "detection_details": "Error during post-processing",
                    "error_message": str(e)
                })
        return {"predictions": final_predictions}
