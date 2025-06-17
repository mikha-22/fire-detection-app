import json
from google.cloud.aiplatform.prediction import LocalModel
from predictor import WildfireHeuristicPredictor

print("--- Starting Final Offline Container Test ---")

# This builds the container with your latest, correct code.
print("Building the CPR container...")
local_model = LocalModel.build_cpr_model(
    ".",
    "asia-southeast2-docker.pkg.dev/haryo-kebakaran/wildfire-detector-repo/wildfire-heuristic-predictor:local-test",
    predictor=WildfireHeuristicPredictor,
    requirements_path="requirements.txt"
)

print("\n--- Deploying container to local Docker endpoint... ---")

with local_model.deploy_to_local_endpoint(host_port=8080) as local_endpoint:
    print("Endpoint is live. Sending prediction request...")

    try:
        # Prepare the request body in the correct format
        with open("incidents.jsonl", "r") as f:
            incident_data = json.load(f)
        request_body = {"instances": [incident_data]}
        temp_request_file = "temp_request.json"
        with open(temp_request_file, "w") as f:
            json.dump(request_body, f)

        # Send the request
        predict_response = local_endpoint.predict(
            request_file=temp_request_file,
            headers={"Content-Type": "application/json"},
        )
        
        if predict_response.status_code == 200:
            print("\n--- ✅ Prediction Successful (Status Code 200) ---")
            print("\n--- Parsed Predictions ---")
            
            # --- THIS IS THE CORRECT PARSING LOGIC ---
            response_content = predict_response.content.decode('utf-8')
            # 1. Parse the entire response as a JSON object (a dictionary)
            response_dict = json.loads(response_content)
            # 2. Extract the list of predictions from the 'predictions' key
            all_predictions = response_dict['predictions']
            # --- END OF CORRECTION ---
            
            for prediction in all_predictions:
                instance_id = prediction['instance_id']
                score = prediction['confidence_score']
                print(f"  Cluster ID: {instance_id}, Predicted Risk Score: {score:.4f}")
        else:
            print(f"\n--- ❌ Prediction Failed with Status Code: {predict_response.status_code} ---")
            print("--- Server Response ---")
            print(predict_response.text)

    except Exception as e:
        print(f"\n--- ❌ Test Script Crashed ---")
        print(f"Error: {e}")

print("\n--- Local endpoint has been shut down. Test complete. ---")
