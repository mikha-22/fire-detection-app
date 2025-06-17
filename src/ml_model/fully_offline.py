import json
from google.cloud.aiplatform.prediction import LocalModel
from predictor import WildfireHeuristicPredictor

print("--- Building local model for OFFLINE container testing ---")

# 1. Build the local model object. This inspects your code and prepares it.
#    We use a 'local-test' tag to avoid confusion with the real deployed image.
local_model = LocalModel.build_cpr_model(
    ".",
    "asia-southeast2-docker.pkg.dev/haryo-kebakaran/wildfire-detector-repo/wildfire-heuristic-predictor:local-test",
    predictor=WildfireHeuristicPredictor,
    requirements_path="requirements.txt"
)

print("\n--- Deploying to local Docker endpoint. This may take a moment... ---")

# 2. This 'with' block starts the Docker container, runs the test, and automatically cleans up.
with local_model.deploy_to_local_endpoint(
    # The container needs access to the data file you downloaded.
    # We mount the current directory into the container's /data directory.
    host_port=8080, # Expose the container's port to your machine's port 8080
) as local_endpoint:
    
    print("Endpoint is live on local Docker. Sending prediction request...")

    # 3. Send the local data file to the local container for prediction.
    try:
        predict_response = local_endpoint.predict(
            request_file="incidents.jsonl",  # The file we downloaded in Step 1
            headers={"Content-Type": "application/json"},
        )

        print("\n--- ✅ Prediction Successful ---")
        print("\n--- Parsed Predictions ---")
        
        # Decode the raw byte content of the response
        response_content = predict_response.content.decode('utf-8')
        
        # The result is a single JSON object with a 'predictions' key
        all_predictions = json.loads(response_content)['predictions']
        
        for prediction in all_predictions:
            instance_id = prediction['instance_id']
            score = prediction['confidence_score']
            print(f"  Cluster ID: {instance_id}, Predicted Risk Score: {score:.4f}")

    except Exception as e:
        print(f"\n--- ❌ Prediction Failed ---")
        print(f"Error: {e}")
        print("\n--- Printing container logs for debugging ---")
        local_endpoint.print_container_logs()


print("\n--- Local endpoint has been shut down. Test complete. ---")
