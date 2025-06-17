import json
import numpy as np
from predictor import FireRiskHeuristicModel 

print("--- Starting Enhanced Offline Logic Test ---")

# 1. Load the model configuration
try:
    with open("config.json", 'r') as f:
        model_config = json.load(f)
    print("Successfully loaded config.json")
except Exception as e:
    print(f"Failed to load config.json: {e}")
    exit()

# 2. Instantiate the model
risk_model = FireRiskHeuristicModel(config=model_config)
print("Heuristic model instantiated.")

# 3. Open and process the incidents file
print("\n--- Processing incidents.jsonl with pre-processing ---")
try:
    with open("incidents.jsonl", "r") as f:
        file_content = f.read()
        incident_data = json.loads(file_content)
        
        # --- NEW PRE-PROCESSING STEP ---
        # Calculate the average Fire Radiative Power (FRP)
        hotspots = incident_data.get("hotspots", [])
        if hotspots:
            frp_values = [h['properties']['frp_mean'] for h in hotspots if 'frp_mean' in h.get('properties', {})]
            avg_frp = np.mean(frp_values) if frp_values else 0
        else:
            avg_frp = 0
        
        # Add the newly calculated value to the data for the model to use
        incident_data['avg_frp'] = avg_frp
        print(f"Pre-processing complete. Calculated Average FRP: {avg_frp:.2f}")
        # --- END OF PRE-PROCESSING ---

        cluster_id = incident_data.get("cluster_id", "N/A")
        result = risk_model.predict_confidence(incident_data)
        score = result['confidence_score']
        
        print(f"\n  Cluster ID: {cluster_id}, Predicted Risk Score: {score:.4f}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Test complete. ---")
