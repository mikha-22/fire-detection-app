# src/common/config.py

"""
Global configuration settings for the Wildfire Detection System.
"""

# Define the geographic areas to monitor for wildfires.
MONITORED_REGIONS = [
    {
        "id": "sumatra_riau",
        "name": "Sumatra - Riau Province",
        "bbox": [100.0, -1.0, 104.0, 2.0],
        "description": "A primary hotspot. Covers the Kampar Peninsula and coastal peatlands frequently burned for agriculture."
    },
    {
        "id": "kalimantan_tengah",
        "name": "Central Kalimantan",
        "bbox": [111.0, -3.5, 115.5, -0.5],
        "description": "The highest density of peat fires in SE Asia, focused on the ex-Mega Rice Project area south of Palangkaraya."
    },
    {
        "id": "sumatra_selatan",
        "name": "South Sumatra",
        "bbox": [103.5, -4.5, 106.5, -2.0],
        "description": "Another major fire-prone area, targeting the extensive peatlands of the Ogan Komering Ilir (OKI) regency."
    }
]

# Your Google Cloud Storage bucket name
GCS_BUCKET_NAME = "fire-app-bucket"

# GCS Path Prefixes - Better organized structure
GCS_PATHS = {
    # Raw incident data from FIRMS processing
    "incidents": "incidents",
    
    # Batch prediction jobs
    "batch_jobs": "batch_jobs",
    "batch_input": "input",
    "batch_raw_output": "raw_output",  # Where Vertex AI writes
    "batch_processed_output": "processed_output",  # Our processed results
    
    # Satellite imagery
    "satellite_imagery": "satellite_imagery",
    
    # Final reports and visualizations
    "reports": "reports",
    "report_images": "report_images",
}

# File naming conventions
FILE_NAMES = {
    "incident_data": "detected_incidents.jsonl",
    "batch_input": "batch_input.jsonl",
    "batch_predictions": "predictions.jsonl",
    "job_metadata": "metadata.json",
    "job_manifest": "manifest.json",
    "job_summary": "summary.json",
    "report_metadata": "report_metadata.json",
}
