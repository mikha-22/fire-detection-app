# src/ml_model/predictor.py

import os
import io
import json
import logging
import traceback
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
            self._model.load_state_dict(torch.load(local_model_path, map_location=self._device, weights_only=True))
            self._model.to(self._device)
            self._model.eval()
            _log_json("INFO", "Model loaded and ready for prediction.")
        except Exception as e:
            _log_json("CRITICAL", "Failed to load model artifacts.", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess(self, prediction_input: Dict[str, Any]) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
        """
        Handles both single JSONL lines and batch JSON requests.
        """
        _log_json("DEBUG", "Received payload for preprocessing.", payload=prediction_input)
        Image.MAX_IMAGE_PIXELS = 2_000_000_000

        instances = prediction_input.get("instances", [])
        if not instances:
            # If "instances" key is not present, assume the entire payload is a single instance.
            # This handles the case for a single line from a JSONL file.
            instances = [prediction_input]

        preprocessed_data = []
        for instance in instances:
            # Each instance from our image_processor contains a 'clusters' list (usually with one item)
            clusters_to_process = instance.get("clusters", [])
            
            for cluster_data in clusters_to_process:
                gcs_image_uri = cluster_data.get("gcs_image_uri")
                cluster_id = cluster_data.get("cluster_id", "unknown_cluster")
                
                if not gcs_image_uri:
                    _log_json("WARNING", f"Skipping cluster {cluster_id} due to missing 'gcs_image_uri'.")
                    continue
                
                try:
                    bucket_name, blob_name = gcs_image_uri.replace("gs://", "").split("/", 1)
                    blob = self._storage_client.bucket(bucket_name).blob(blob_name)
                    image_bytes = blob.download_as_bytes()

                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    transformed_image = MODEL_INPUT_TRANSFORMS(image)

                    # Pass the original cluster data along with the tensor
                    preprocessed_data.append((cluster_data, transformed_image))
                    _log_json("INFO", f"Preprocessing successful for cluster {cluster_id}.", image_uri=gcs_image_uri)

                except Exception as e:
                    _log_json("ERROR", f"Preprocessing failed for cluster {cluster_id}.", error=str(e), exc_info=True)
        
        _log_json("INFO", f"Successfully preprocessed {len(preprocessed_data)} total items for the model.")
        return preprocessed_data

    def predict(self, instances: List[Tuple[Dict[str, Any], torch.Tensor]]) -> List[Tuple[Dict[str, Any], torch.Tensor]]:
        if not instances:
            _log_json("WARNING", "Predict received an empty list of instances. Nothing to do.")
            return []

        original_inputs = [item[0] for item in instances]
        tensors_to_infer = [item[1] for item in instances]
        
        batch_to_infer = torch.stack(tensors_to_infer).to(self._device)
        _log_json("INFO", f"Performing inference on a batch of {len(tensors_to_infer)} clusters.", shape=str(batch_to_infer.shape))
        
        try:
            with torch.no_grad():
                prediction_outputs = self._model(batch_to_infer)
            _log_json("INFO", "Inference successful for batch.", output_shape=str(prediction_outputs.shape))
            
            results = [(original_inputs[i], prediction_outputs[i]) for i in range(len(original_inputs))]
            return results
        except Exception as e:
            _log_json("CRITICAL", "Prediction failed for the entire batch.", error=str(e), exc_info=True)
            raise RuntimeError(f"Prediction failed for batch: {e}")

    def postprocess(self, prediction_results: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        FIXED: Return predictions wrapped in a dictionary with 'predictions' key for Vertex AI.
        """
        final_predictions = []
        _log_json("INFO", f"Postprocessing {len(prediction_results)} individual cluster results.")
        
        for original_input, inference_output in prediction_results:
            try:
                probabilities = F.softmax(inference_output, dim=0)
                predicted_class = torch.argmax(probabilities).item()
                is_detected = (predicted_class == 1)
                confidence = probabilities[predicted_class].item()

                final_predictions.append({
                    "instance_id": original_input.get("cluster_id", "unknown"),
                    "detected": bool(is_detected),
                    "confidence": float(confidence),
                    "detection_details": "Fire detected by AI model" if is_detected else "No fire detected by AI model",
                    "error_message": None
                })
                _log_json("INFO", f"Postprocessing successful for cluster {original_input.get('cluster_id', 'unknown')}.", detected=is_detected, confidence=confidence)
            except Exception as e:
                _log_json("CRITICAL", f"Postprocessing failed for instance {original_input.get('cluster_id', 'unknown')}.", error=str(e), exc_info=True)
                final_predictions.append({
                    "instance_id": original_input.get("cluster_id", "unknown"),
                    "detected": False,
                    "confidence": 0.0,
                    "detection_details": "Error during post-processing",
                    "error_message": str(e)
                })
        
        # FIXED: Wrap predictions in a dictionary with 'predictions' key
        return {"predictions": final_predictions}
