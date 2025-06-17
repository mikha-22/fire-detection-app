import json
import numpy as np
# Import the core logic class directly
from predictor import FireRiskHeuristicModel 

print("--- Starting Pure Python Offline Logic Test ---")

# 1. Load the model configuration
try:
    with open("config.json", 'r') as f:
        model_config = json.load(f)
    print("Successfully loaded config.json")
except Exception as e:
    print(f"Failed to load config.json: {e}")
    exit()

# 2. Instantiate the core logic model
risk_model = FireRiskHeuristicModel(config=model_config)
print("Heuristic model instantiated.")

# 3. Open the local incidents file and process it
print("\n--- Processing incidents.jsonl with manual pre-processing ---")
try:
    with open("incidents.jsonl", "r") as f:
        file_content = f.read()
        incident_data = json.loads(file_content)
        
        # --- MANUAL PRE-PROCESSING STEP ---
        # This step simulates what the WildfireHeuristicPredictor.preprocess
        # method does before calling the model.
        hotspots = incident_data.get("hotspots", [])
        if hotspots:
            frp_values = [h['properties']['frp_mean'] for h in hotspots if 'frp_mean' in h.get('properties', {})]
            avg_frp = np.mean(frp_values) if frp_values else 0
        else:
            avg_frp = 0
        
        # Add the new value to the data dictionary before scoring
        incident_data['avg_frp'] = avg_frp
        print(f"Pre-processing complete. Calculated Average FRP: {avg_frp:.2f}")
        # --- END OF MANUAL PRE-PROCESSING ---

        cluster_id = incident_data.get("cluster_id", "N/A")
        
        # Pass the pre-processed data to the model
        result = risk_model.predict_confidence(incident_data)
        score = result['confidence_score']
        
        print(f"\n  Cluster ID: {cluster_id}, Predicted Risk Score: {score:.4f}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Test complete. ---")
