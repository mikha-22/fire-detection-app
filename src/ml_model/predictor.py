import os
import json
import numpy as np
from typing import Any, Dict, List

from google.cloud.aiplatform.prediction.predictor import Predictor

# --- Heuristic Model Logic (This class remains correct) ---
class FireRiskHeuristicModel:
    def __init__(self, config: Dict[str, Any]):
        if 'criteria' not in config or not isinstance(config['criteria'], list):
            raise ValueError("Configuration must contain a 'criteria' list.")
        self.criteria: List[Dict[str, Any]] = config['criteria']
        self._validate_config()

    def _validate_config(self):
        total_weight = sum(c['weight'] for c in self.criteria)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Sum of all weights must be 1.0, but is {total_weight}")

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def _get_value_from_dot_key(self, data: Dict[str, Any], key: str) -> Any:
        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else: return None
        return value

    def _get_criterion_score(self, criterion: Dict[str, Any], data: Dict[str, Any]) -> float:
        data_key = criterion['data_key']
        value = self._get_value_from_dot_key(data, data_key)

        if value is None: return 5.0

        if criterion['type'] == 'categorical':
            return float(criterion['mapping'].get(str(value).lower(), 5.0))
        
        elif criterion['type'] == 'numerical_threshold':
            direction = criterion.get("direction", "ascending")
            sorted_thresholds = sorted(criterion['mapping'].items(), key=lambda item: float(item[0]))
            for threshold, score in sorted_thresholds:
                if float(value) <= float(threshold): return float(score)
            if direction == "ascending": return float(max(criterion['mapping'].values()))
            else: return float(min(criterion['mapping'].values()))
        else:
            raise ValueError(f"Unsupported criterion type: {criterion['type']}")

    def _calculate_raw_score(self, data: Dict[str, Any]) -> float:
        total_raw_score = sum(self._get_criterion_score(c, data) * c['weight'] for c in self.criteria)
        return total_raw_score - 5.0

    def predict_confidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_score = self._calculate_raw_score(data)
        confidence_score = self._sigmoid(raw_score)
        return {"confidence_score": confidence_score, "raw_score": raw_score}


# --- CPR Predictor Class ---
class WildfireHeuristicPredictor(Predictor):
    def __init__(self):
        self._model = None

    def load(self, artifacts_uri: str) -> None:
        config_path = os.path.join(artifacts_uri, "config.json")
        if not os.path.exists(config_path):
             config_path = "config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        self._model = FireRiskHeuristicModel(config=model_config)

    # --- THIS IS THE CORRECTED PREPROCESS METHOD ---
    def preprocess(self, prediction_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Pre-processes the input by calculating derived values needed by the model.
        This is where all data preparation logic should live.
        """
        processed_instances = []
        for instance in prediction_input["instances"]:
            # Calculate the average Fire Radiative Power (FRP)
            hotspots = instance.get("hotspots", [])
            if hotspots:
                frp_values = [h['properties']['frp_mean'] for h in hotspots if 'frp_mean' in h.get('properties', {})]
                avg_frp = np.mean(frp_values) if frp_values else 0
            else:
                avg_frp = 0
            
            # Add the newly calculated value to the instance for the model to use
            instance['avg_frp'] = avg_frp
            processed_instances.append(instance)
            
        return processed_instances
    # --- END OF CORRECTION ---

    def predict(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = []
        for instance in instances:
            cluster_id = instance.get("cluster_id", "unknown_instance")
            result = self._model.predict_confidence(instance)
            result["instance_id"] = cluster_id
            predictions.append(result)
        return predictions

    def postprocess(self, prediction_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        return {"predictions": prediction_results}
