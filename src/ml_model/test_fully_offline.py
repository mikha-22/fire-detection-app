import json
from google.cloud.aiplatform.prediction import LocalModel

# --- THIS IS THE IMAGE WE JUST BUILT MANUALLY ---
PRE_BUILT_IMAGE_URI = "asia-southeast2-docker.pkg.dev/haryo-kebakaran/wildfire-detector-repo/wildfire-heuristic-predictor:local-test"

print("--- Skipping build and using pre-built local image ---")
print(f"Image URI: {PRE_BUILT_IMAGE_URI}")

# 1. Instantiate LocalModel directly from the image URI.
#    This completely bypasses the failing .build_cpr_model() step.
local_model = LocalModel(serving_container_image_uri=PRE_BUILT_IMAGE_URI)

print("\n--- Deploying to local Docker endpoint. This may take a moment... ---")

# 2. This 'with' block starts the Docker container, runs the test, and automatically cleans up.
with local_model.deploy_to_local_endpoint(host_port=8080) as local_endpoint:
    
    print("Endpoint is live on local Docker. Sending prediction request...")

    try:
        predict_response = local_endpoint.predict(
            request_file="incidents.jsonl",
            headers={"Content-Type": "application/json"},
        )
        
        if predict_response.status_code != 200:
            print(f"\n--- ❌ Prediction Failed with Status Code: {predict_response.status_code} ---")
            print("--- Server Response ---")
            print(predict_response.text)
            print("\n--- Printing container logs for debugging ---")
            local_endpoint.print_container_logs()
        else:
            print("\n--- ✅ Prediction Successful (Status Code 200) ---")
            print("\n--- Parsed Predictions ---")
            
            response_content = predict_response.content.decode('utf-8')
            # The local server returns a list directly
            all_predictions = json.loads(response_content)
            
            for prediction in all_predictions:
                instance_id = prediction['instance_id']
                score = prediction['confidence_score']
                print(f"  Cluster ID: {instance_id}, Predicted Risk Score: {score:.4f}")

    except Exception as e:
        print(f"\n--- ❌ Test Script Crashed ---")
        print(f"Error: {e}")

print("\n--- Local endpoint has been shut down. Test complete. ---")
