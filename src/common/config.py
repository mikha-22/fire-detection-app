# src/common/config.py

"""
Global configuration settings for the Wildfire Detection System.
This file defines a clean, pipeline-oriented structure for all data artifacts.
"""

# Your Google Cloud Storage bucket name
GCS_BUCKET_NAME = "fire-app-bucket"

# --- NEW: Pipeline-Oriented GCS Path Structure ---
# This structure makes the data flow clear and easy to navigate.
GCS_PATHS = {
    # Stage 0: Static data and configuration (not used yet, but good practice)
    "CONFIG": "00_pipeline_config",
    
    # Stage 1: Initial detected incidents from FIRMS
    "INCIDENTS_DETECTED": "01_incidents_detected",
    
    # Stage 2: Data related to imagery and prediction jobs
    "SATELLITE_IMAGERY": "02_satellite_imagery",
    "PREDICTION_JOBS": "02_prediction_jobs", # Parent folder for jobs
    "JOB_INPUT": "input",                  # Subfolder for job's input file
    "JOB_RAW_OUTPUT": "raw_vertex_output", # Subfolder for messy Vertex AI output
    
    # Stage 3: Final, user-facing reports
    "FINAL_REPORTS": "03_final_reports",
    "REPORT_IMAGES": "images" # Subfolder for map images within a report
}

# --- NEW: Standardized File Naming ---
FILE_NAMES = {
    "incident_data": "incidents.jsonl",
    "batch_input": "prediction_input.jsonl",
    "job_metadata": "metadata.json",
    "job_manifest": "daily_manifest.json",
    "report_html": "report.html",
}
