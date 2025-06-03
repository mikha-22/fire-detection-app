# src/common/config.py

"""
Global configuration settings for the Wildfire Detection System.
"""

# Define the geographic areas to monitor for wildfires.
# Each region is a dictionary with:
# - 'id': A unique identifier for the region.
# - 'name': A human-readable name for the region.
# - 'bbox': A bounding box defined as [min_longitude, min_latitude, max_longitude, max_latitude].
# - 'description': A brief description of the region.
MONITORED_REGIONS = [
    {
        "id": "california_central",
        "name": "Central California Wildfire Zone",
        "bbox": [-122.0, 36.0, -118.0, 38.0],  # [min_lon, min_lat, max_lon, max_lat]
        "description": "A test region in Central California prone to wildfires."
    },
    {
        "id": "australia_southeast",
        "name": "Southeast Australia Bushfire Zone",
        "bbox": [140.0, -38.0, 153.0, -28.0],
        "description": "A test region in Southeast Australia."
    }
    # Add more regions as needed
]

# Your Google Cloud Storage bucket name.
# This bucket will be used to store raw data, processed outputs (maps, metadata),
# and Vertex AI batch prediction inputs/outputs.
GCS_BUCKET_NAME = "fire-app-bucket" # Updated to your specified bucket name
