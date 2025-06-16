#!/usr/bin/env python3
"""
Debug script to check the contents of Vertex AI prediction output files
"""

import os
from google.cloud import storage
import json

# Configuration
GCS_BUCKET_NAME = "fire-app-bucket"
RUN_DATE = "2025-06-16"
JOB_ID = "job_20250616_164349"

def main():
    print("=== Debugging Vertex AI Output Files ===\n")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    # Look for prediction files
    prefix = f"02_prediction_jobs/{RUN_DATE}/{JOB_ID}/raw_vertex_output/"
    print(f"Searching for files with prefix: gs://{GCS_BUCKET_NAME}/{prefix}")
    
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"\nFound {len(blobs)} total files:")
    for blob in blobs:
        print(f"  - {blob.name}")
    
    # Filter for prediction result files
    prediction_files = [b for b in blobs if "prediction.results" in b.name]
    print(f"\nFound {len(prediction_files)} prediction result files:")
    
    for blob in prediction_files:
        print(f"\n--- File: {blob.name} ---")
        print(f"Size: {blob.size} bytes")
        
        try:
            # Download and display content
            content = blob.download_as_string().decode('utf-8')
            print(f"Raw content length: {len(content)} characters")
            print(f"Number of lines: {len(content.splitlines())}")
            
            # Show first few lines
            lines = content.splitlines()
            print(f"\nFirst 5 lines (raw):")
            for i, line in enumerate(lines[:5]):
                print(f"  Line {i}: {repr(line[:200])}")  # repr to show whitespace
            
            # Try to parse as JSON
            print(f"\nParsing JSON lines:")
            valid_predictions = 0
            errors = 0
            
            for i, line in enumerate(lines):
                if not line.strip():  # Skip empty lines
                    print(f"  Line {i}: [EMPTY]")
                    continue
                    
                try:
                    data = json.loads(line)
                    valid_predictions += 1
                    print(f"  Line {i}: Valid JSON - keys: {list(data.keys())}")
                    if i < 3:  # Show full content of first 3 predictions
                        print(f"    Full content: {json.dumps(data, indent=2)}")
                except json.JSONDecodeError as e:
                    errors += 1
                    print(f"  Line {i}: JSON Error - {e}")
                    print(f"    Content: {repr(line[:100])}")
            
            print(f"\nSummary: {valid_predictions} valid predictions, {errors} errors")
            
        except Exception as e:
            print(f"ERROR reading file: {e}")
    
    # Check for error files
    print(f"\n\n=== Checking Error Files ===")
    error_files = [b for b in blobs if "errors_stats" in b.name or "error" in b.name.lower()]
    print(f"Found {len(error_files)} error files:")
    
    for blob in error_files:
        print(f"\n--- Error File: {blob.name} ---")
        print(f"Size: {blob.size} bytes")
        
        try:
            error_content = blob.download_as_string().decode('utf-8')
            print(f"Error content:\n{error_content}")
        except Exception as e:
            print(f"Could not read error file: {e}")
    
    # Also check the input file to compare
    print(f"\n\n=== Checking Input File for Comparison ===")
    input_path = f"02_prediction_jobs/{RUN_DATE}/{JOB_ID}/prediction_input.jsonl"
    try:
        input_blob = bucket.blob(input_path)
        input_content = input_blob.download_as_string().decode('utf-8')
        input_lines = input_content.strip().split('\n')
        print(f"Input file has {len(input_lines)} instances")
        
        # Show first instance
        if input_lines:
            first_instance = json.loads(input_lines[0])
            print(f"First input instance:")
            print(json.dumps(first_instance, indent=2))
    except Exception as e:
        print(f"Could not read input file: {e}")
    
    # Check the Vertex AI job status
    print(f"\n\n=== Checking Job Metadata ===")
    metadata_path = f"02_prediction_jobs/{RUN_DATE}/{JOB_ID}/metadata.json"
    try:
        metadata_blob = bucket.blob(metadata_path)
        metadata = json.loads(metadata_blob.download_as_string())
        print(f"Job metadata:")
        print(json.dumps(metadata, indent=2))
    except Exception as e:
        print(f"Could not read metadata: {e}")

if __name__ == "__main__":
    main()
