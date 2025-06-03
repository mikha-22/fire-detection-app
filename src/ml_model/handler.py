# src/ml_model/handler.py

import os
import io
import logging
import json
import torch
import torch.nn.functional as F
from PIL import Image
from google.cloud import storage
from ts.torch_handler.base_handler import BaseHandler
from datetime import datetime

# Import the model definition and transforms
from fire_detection_model import DummyFireDetectionModel, MODEL_INPUT_TRANSFORMS

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "TorchServeHandler",
        **kwargs
    }
    print(json.dumps(log_entry))


class FireDetectionHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.storage_client = None
        _log_json("INFO", "FireDetectionHandler instance created.")

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu")

        _log_json("INFO", f"Initializing handler on device: {self.device}")

        self.model = DummyFireDetectionModel()
        model_pt_path = os.path.join(model_dir, "model.pth")
        if not os.path.exists(model_pt_path):
            _log_json("ERROR", f"Model state_dict not found at: {model_pt_path}")
            raise FileNotFoundError(f"Missing model state_dict: {model_pt_path}")

        try:
            self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            _log_json("INFO", "Model loaded successfully.", model_path=model_pt_path, device=str(self.device))
        except Exception as e:
            _log_json("CRITICAL", f"Failed to load model state_dict: {e}", model_path=model_pt_path)
            raise RuntimeError(f"Failed to load model: {e}")

        try:
            self.storage_client = storage.Client()
            _log_json("INFO", "Google Cloud Storage client initialized.")
        except Exception as e:
            _log_json("CRITICAL", f"Failed to initialize GCS client: {e}")
            raise RuntimeError(f"Failed to initialize GCS client: {e}")

        self.initialized = True
        _log_json("INFO", "FireDetectionHandler initialization complete.")

    def preprocess(self, data: list) -> tuple[list[torch.Tensor | None], list[dict]]:
        _log_json("INFO", f"Starting preprocessing for {len(data)} instances.")
        
        processed_tensors = []
        instance_metadata_list = [] 

        for i, row in enumerate(data):
            original_input_data = {} 
            current_instance_id_for_log = f"instance_{i}_parsing_failed" # Default if parsing row fails

            try:
                if isinstance(row, dict) and 'data' in row and isinstance(row['data'], bytes):
                    input_data = json.loads(row['data'].decode('utf-8'))
                    original_input_data = input_data 
                elif isinstance(row, dict):
                    input_data = row
                    original_input_data = row 
                else:
                    err_msg = f"Unexpected input data format for instance {i}."
                    _log_json("ERROR", err_msg, input_type=type(row), input_snippet=str(row)[:100])
                    processed_tensors.append(None) 
                    instance_metadata_list.append({"instance_id": current_instance_id_for_log, "preprocess_error": err_msg})
                    continue
                
                # Successfully parsed row, now store metadata and get instance_id
                instance_metadata_list.append(original_input_data)
                current_instance_id_for_log = original_input_data.get("instance_id", f"instance_{i}")

                gcs_image_uri = input_data.get("gcs_image_uri")
                if not gcs_image_uri:
                    err_msg = f"Missing 'gcs_image_uri' for instance {current_instance_id_for_log}."
                    _log_json("ERROR", err_msg, instance_id=current_instance_id_for_log)
                    processed_tensors.append(None) 
                    instance_metadata_list[-1]["preprocess_error"] = err_msg # Add error to last appended metadata
                    continue

                _log_json("INFO", f"Downloading image from GCS.", instance_id=current_instance_id_for_log, uri=gcs_image_uri)
                parts = gcs_image_uri.replace("gs://", "").split("/", 1)
                if len(parts) < 2:
                    err_msg = f"Invalid GCS URI format: {gcs_image_uri}."
                    _log_json("ERROR", err_msg, instance_id=current_instance_id_for_log)
                    processed_tensors.append(None)
                    instance_metadata_list[-1]["preprocess_error"] = err_msg
                    continue
                
                bucket_name, blob_name = parts[0], parts[1]
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                if not blob.exists():
                    err_msg = f"GCS image blob does not exist: {gcs_image_uri}."
                    _log_json("ERROR", err_msg, instance_id=current_instance_id_for_log)
                    processed_tensors.append(None)
                    instance_metadata_list[-1]["preprocess_error"] = err_msg
                    continue

                image_bytes = blob.download_as_bytes()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                transformed_image = MODEL_INPUT_TRANSFORMS(image)
                processed_tensors.append(transformed_image)
                _log_json("INFO", "Image preprocessed successfully.", instance_id=current_instance_id_for_log, uri=gcs_image_uri)

            except Exception as e:
                # This catches errors during parsing row, GCS ops, or image transformation
                err_msg = f"Error preprocessing instance {current_instance_id_for_log}: {str(e)}"
                _log_json("ERROR", err_msg, original_input_snippet=str(row)[:200], error_type=type(e).__name__)
                processed_tensors.append(None)
                # If metadata wasn't appended due to early failure (e.g. row parsing), append it now with error
                if len(instance_metadata_list) == len(processed_tensors) -1: # Check if metadata for this item is missing
                    instance_metadata_list.append({"instance_id": current_instance_id_for_log, "preprocess_error": err_msg})
                else: # Metadata was already appended, just add/update the error
                    instance_metadata_list[-1]["preprocess_error"] = err_msg
        
        # Ensure lists are of same length before returning (should be by construction)
        if len(processed_tensors) != len(instance_metadata_list):
            _log_json("CRITICAL", "Mismatch in lengths of processed_tensors and instance_metadata_list.",
                      len_tensors=len(processed_tensors), len_metadata=len(instance_metadata_list))
            # Fallback: Pad metadata if necessary, though this indicates a logic error above.
            while len(instance_metadata_list) < len(processed_tensors):
                instance_metadata_list.append({"instance_id": "fallback_meta_error", "preprocess_error": "Metadata list padding"})


        return processed_tensors, instance_metadata_list

    def inference(self, handler_input: tuple[list[torch.Tensor | None], list[dict]]) -> tuple[list[torch.Tensor | None], list[dict]]:
        processed_tensors, instance_metadata_list = handler_input
        _log_json("INFO", f"Starting inference for {len(processed_tensors)} total instances.")

        valid_tensors = []
        valid_indices = []
        for i, tensor_data in enumerate(processed_tensors):
            if tensor_data is not None:
                valid_tensors.append(tensor_data)
                valid_indices.append(i)

        inference_results_for_valid_inputs = []
        if valid_tensors:
            batch_to_infer = torch.stack(valid_tensors).to(self.device)
            _log_json("INFO", f"Performing inference on a batch of {batch_to_infer.shape[0]} valid instances.")
            with torch.no_grad():
                inference_results_for_valid_inputs = self.model(batch_to_infer)
            _log_json("INFO", "Inference complete for valid instances.")
        else:
            _log_json("INFO", "No valid instances to perform inference on.")

        all_inference_outputs = [None] * len(processed_tensors)
        for i, result_tensor in enumerate(inference_results_for_valid_inputs):
            original_index = valid_indices[i]
            all_inference_outputs[original_index] = result_tensor
        
        return all_inference_outputs, instance_metadata_list

    def postprocess(self, handler_output: tuple[list[torch.Tensor | None], list[dict]]) -> list[dict[str, any]]:
        all_inference_outputs, instance_metadata_list = handler_output
        _log_json("INFO", f"Starting postprocessing for {len(all_inference_outputs)} total instances.")
        
        results = []

        for i in range(len(all_inference_outputs)):
            inference_output_tensor = all_inference_outputs[i]
            # Ensure metadata list is not shorter than output list (should not happen with current logic)
            if i >= len(instance_metadata_list):
                _log_json("ERROR", "Metadata list shorter than inference output list during postprocessing.",
                          index=i, len_metadata=len(instance_metadata_list), len_outputs=len(all_inference_outputs))
                # Create a fallback metadata to avoid crashing
                metadata = {"instance_id": f"missing_metadata_instance_{i}", 
                            "error": "Metadata missing for this instance in postprocess"}
            else:
                metadata = instance_metadata_list[i]
            
            instance_id = metadata.get("instance_id", f"processed_instance_{i}") 

            if inference_output_tensor is None:
                error_message = metadata.get("preprocess_error", "Processing failed (reason unknown or occurred during inference step for valid preprocess)")
                
                results.append({
                    "instance_id": instance_id,
                    "detected": False, 
                    "confidence": 0.0,
                    "detection_details": "Error during processing.",
                    "error_message": error_message 
                })
                _log_json("WARNING", "Instance failed processing.", instance_id=instance_id, error=error_message)
            else:
                probabilities = F.softmax(inference_output_tensor.unsqueeze(0), dim=1) 
                predicted_class = torch.argmax(probabilities, dim=1).item()
                
                is_detected = (predicted_class == 1)
                confidence = probabilities[0][predicted_class].item()

                results.append({
                    "instance_id": instance_id,
                    "detected": bool(is_detected),
                    "confidence": float(confidence),
                    "detection_details": "Fire detected by AI model" if is_detected else "No fire detected by AI model",
                    "error_message": None
                })
        
        _log_json("INFO", "Postprocessing complete.", total_results_generated=len(results))
        return results

# --- Example Usage (for local testing of handler methods, not full TorchServe) ---
if __name__ == '__main__':
    from src.common.config import GCS_BUCKET_NAME 

    _log_json("INFO", "Running local handler test (partial - requires GCS setup and model file).")

    class MockContext:
        def __init__(self, model_dir, gpu_id=None):
            self.system_properties = {"model_dir": model_dir, "gpu_id": gpu_id}
            self.manifest = {"model": {"modelName": "model"}}

    if not os.path.exists("model_artifacts"):
        os.makedirs("model_artifacts")
    
    dummy_model_for_test = DummyFireDetectionModel()
    torch.save(dummy_model_for_test.state_dict(), os.path.join("model_artifacts", "model.pth"))
    _log_json("INFO", "Dummy model file 'model.pth' created for local handler test.")

    mock_context = MockContext("model_artifacts")

    handler = FireDetectionHandler()
    try:
        handler.initialize(mock_context)

        valid_gcs_uri = f"gs://{GCS_BUCKET_NAME}/raw_satellite_imagery/test_image_valid.png" 
        invalid_gcs_uri = f"gs://{GCS_BUCKET_NAME}/raw_satellite_imagery/non_existent_image.png"
        malformed_gcs_uri = "gs:/malformed/uri"
        missing_uri_input = {"instance_id": "test_instance_4_missing_uri", "region_metadata": {"id": "test_region_4"}}
        
        dummy_local_image_path = "dummy_handler_test_image.png"
        if not os.path.exists(dummy_local_image_path):
            Image.new('RGB', (224, 224), color = 'green').save(dummy_local_image_path)
            _log_json("INFO", f"Created dummy local image for GCS upload test: {dummy_local_image_path}")
        
        try:
            if GCS_BUCKET_NAME != "fire-app-bucket":
                 _log_json("WARNING", f"GCS_BUCKET_NAME '{GCS_BUCKET_NAME}' might not be intended for this test. Skipping GCS upload.")
            else:
                storage_client_test = storage.Client()
                bucket_test = storage_client_test.bucket(GCS_BUCKET_NAME)
                blob_test = bucket_test.blob(valid_gcs_uri.replace(f"gs://{GCS_BUCKET_NAME}/", ""))
                if not blob_test.exists():
                    blob_test.upload_from_filename(dummy_local_image_path)
                    _log_json("INFO", f"Uploaded {dummy_local_image_path} to {valid_gcs_uri} for testing.")
                else:
                    _log_json("INFO", f"Test image {valid_gcs_uri} already exists in GCS.")
        except Exception as e:
            _log_json("WARNING", f"Could not ensure test image exists at {valid_gcs_uri} due to: {e}. GCS test case might fail.")

        input_batch_for_preprocess = [
            {"instance_id": "test_instance_1_valid", "gcs_image_uri": valid_gcs_uri, "region_metadata": {"id": "test_region_1"}},
            {"instance_id": "test_instance_2_invalid_uri", "gcs_image_uri": invalid_gcs_uri, "region_metadata": {"id": "test_region_2"}},
            {"instance_id": "test_instance_3_malformed_uri", "gcs_image_uri": malformed_gcs_uri, "region_metadata": {"id": "test_region_3"}},
            missing_uri_input,
            {"instance_id": "test_instance_5_another_valid", "gcs_image_uri": valid_gcs_uri, "region_metadata": {"id": "test_region_5"}} 
        ]
        
        _log_json("INFO", "--- Testing Preprocess ---")
        processed_tensors, metadata_list = handler.preprocess(input_batch_for_preprocess)
        
        for i, tensor in enumerate(processed_tensors):
            meta = metadata_list[i] if i < len(metadata_list) else {"instance_id": "error_meta_missing", "preprocess_error": "Metadata missing"}
            status = "Success" if tensor is not None else f"Failure ({meta.get('preprocess_error', 'Unknown preprocess error')})"
            _log_json("INFO", f"Preprocess instance {meta.get('instance_id')}: {status}")

        _log_json("INFO", "--- Testing Inference ---")
        inference_outputs, final_metadata_list = handler.inference((processed_tensors, metadata_list))

        _log_json("INFO", "--- Testing Postprocess ---")
        final_results = handler.postprocess((inference_outputs, final_metadata_list))
        
        _log_json("INFO", "Final batch results (should include errors for failed instances):")
        for res_idx, res_item in enumerate(final_results): # Changed variable name from res to res_item
            print(f"Result for instance {res_idx}: {json.dumps(res_item, indent=2)}") # Added index for clarity

    except Exception as e:
        _log_json("CRITICAL", f"Unhandled error during handler local test: {e}", error_details=str(e))
        import traceback
        traceback.print_exc()
