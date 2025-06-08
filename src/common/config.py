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

# Your Google Cloud Storage bucket name.
# This bucket will be used to store raw data, processed outputs (maps, metadata),
# and Vertex AI batch prediction inputs/outputs.
GCS_BUCKET_NAME = "fire-app-bucket" # Ensure this is your correct bucket name
