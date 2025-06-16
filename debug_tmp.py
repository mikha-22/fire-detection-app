#!/usr/bin/env python3
"""
Quick script to check the structure of predictions in the Vertex AI output
"""

import json
from google.cloud import storage

# Configuration
GCS_BUCKET_NAME = "fire-app-bucket"
RUN_DATE = "2025-06-16"
JOB_ID = "job_20250616_180313"

def main():
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    # Find prediction files
    prefix = f"02_prediction_jobs/{RUN_DATE}/{JOB_ID}/raw_vertex_output/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    prediction_files = [b for b in blobs if "prediction.results" in b.name]
    
    print(f"Found {len(prediction_files)} prediction files")
    
    for blob in prediction_files:
        print(f"\n--- File: {blob.name} ---")
        content = blob.download_as_string().decode('utf-8')
        lines = content.strip().split('\n')
        
        print(f"Number of predictions: {len(lines)}")
        
        # Show first 3 predictions
        for i, line in enumerate(lines[:3]):
            if line:
                try:
                    pred = json.loads(line)
                    print(f"\nPrediction {i}:")
                    print(json.dumps(pred, indent=2))
                except Exception as e:
                    print(f"Error parsing line {i}: {e}")
    
    # Also check the input file to see the instance_ids
    print("\n\n--- Checking Input File ---")
    input_path = f"02_prediction_jobs/{RUN_DATE}/{JOB_ID}/prediction_input.jsonl"
    try:
        input_blob = bucket.blob(input_path)
        input_content = input_blob.download_as_string().decode('utf-8')
        input_lines = input_content.strip().split('\n')
        
        print(f"Number of input instances: {len(input_lines)}")
        
        # Show first 3 instances
        for i, line in enumerate(input_lines[:3]):
            if line:
                inst = json.loads(line)
                print(f"\nInput instance {i}:")
                print(f"  instance_id: {inst.get('instance_id')}")
                print(f"  clusters: {inst.get('clusters', [])}")
    except Exception as e:
        print(f"Error reading input file: {e}")

if __name__ == "__main__":
    main()

