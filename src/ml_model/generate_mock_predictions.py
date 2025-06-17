import json
import numpy as np
from predictor import FireRiskHeuristicModel

print("--- Generating Mock Prediction Output File ---")

# 1. Load the model and its config
try:
    with open("config.json", 'r') as f:
        model_config = json.load(f)
    risk_model = FireRiskHeuristicModel(config=model_config)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# 2. Read the source incident data
try:
    with open("incidents.jsonl", "r") as f:
        incident_data = json.load(f)
    print("Source incidents.jsonl loaded successfully.")
except Exception as e:
    print(f"Failed to load incidents.jsonl: {e}")
    exit()

# 3. Pre-process the data (calculate avg_frp)
hotspots = incident_data.get("hotspots", [])
if hotspots:
    frp_values = [h['properties']['frp_mean'] for h in hotspots if 'frp_mean' in h.get('properties', {})]
    avg_frp = np.mean(frp_values) if frp_values else 0
else:
    avg_frp = 0
incident_data['avg_frp'] = avg_frp

# 4. Run prediction
result = risk_model.predict_confidence(incident_data)
cluster_id = incident_data.get("cluster_id")
result['instance_id'] = cluster_id # Add the instance_id for matching

# 5. Write the output to a new file in the correct format
output_filename = "mock_prediction_results.jsonl"
with open(output_filename, "w") as f:
    # Write the result as a single line, just like Vertex AI does
    f.write(json.dumps(result) + '\n')

print(f"\n--- ✅ SUCCESS ---")
print(f"Created '{output_filename}' with prediction score: {result['confidence_score']:.4f}")
