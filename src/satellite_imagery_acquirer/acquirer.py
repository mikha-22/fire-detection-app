# src/satellite_imagery_acquirer/acquirer.py

import os
import logging
import ee # Google Earth Engine Python API
import json # For structured logging of dicts
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Import MONITORED_REGIONS and GCS_BUCKET_NAME from the common config file
# from src.common.config import MONITORED_REGIONS # Not directly used, passed as arg
from src.common.config import GCS_BUCKET_NAME # Used in __main__

# --- Configuration ---
# GEE Project ID for export (Optional, if using custom projects for GEE assets)
# For most use cases, GEE is initialized with your GCP project's credentials.
# GEE_PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # If needed explicitly

# Output directory within GCS bucket for raw imagery
GCS_IMAGE_OUTPUT_PREFIX = "raw_satellite_imagery/"

# Default image collection and filters
# Sentinel-2 Level-2A is a good choice for visible light imagery
# with atmospheric correction.
# See: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED" # Surface Reflectance
CLOUD_COVER_THRESHOLD = 20 # Max percentage of cloud cover allowed

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def _log_json(severity: str, message: str, **kwargs):
    """
    Helper to log structured JSON messages to stdout, which GCP Cloud Logging
    can ingest as structured logs.
    """
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "SatelliteImageryAcquirer",
        **kwargs
    }
    print(json.dumps(log_entry))

class SatelliteImageryAcquirer:
    """
    Component 2: Obtains standard RGB satellite imagery for predefined monitored areas
    using the Google Earth Engine API and exports them to Google Cloud Storage.
    """

    def __init__(self, gcs_bucket_name: str):
        """
        Initializes the SatelliteImageryAcquirer.

        Args:
            gcs_bucket_name (str): The name of the GCS bucket to export images to.
        """
        if not gcs_bucket_name:
            _log_json("CRITICAL", "GCS_BUCKET_NAME is missing or empty. Cannot initialize SatelliteImageryAcquirer.")
            raise ValueError("GCS_BUCKET_NAME must be configured to a valid bucket.")
        self.gcs_bucket_name = gcs_bucket_name

        try:
            ee.Initialize(project=os.environ.get("GCP_PROJECT_ID"))
            _log_json("INFO", "Google Earth Engine initialized successfully.", gcp_project_id=os.environ.get("GCP_PROJECT_ID"))
        except ee.EEException as e:
            _log_json("CRITICAL", "Failed to initialize Google Earth Engine (EEException). "
                                   "Ensure 'earthengine-api' is installed, authentication is set up, "
                                   "and the GCP project has Earth Engine API enabled.", error=str(e))
            raise RuntimeError(f"GEE initialization failed (EEException): {e}")
        except Exception as e:
            _log_json("CRITICAL", "An unexpected error occurred during GEE initialization.", error=str(e))
            raise RuntimeError(f"GEE initialization failed (Unexpected): {e}")


    def _get_latest_composite_image(self, bbox: List[float], date_start: str, date_end: str) -> Optional[ee.Image]:
        """
        Retrieves the latest cloud-filtered Sentinel-2 composite image for a given bbox and date range.

        Args:
            bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat].
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format (exclusive).

        Returns:
            Optional[ee.Image]: An Earth Engine Image object or None if no suitable image is found.
        """
        try:
            geometry = ee.Geometry.Rectangle(bbox)

            collection = ee.ImageCollection(SENTINEL2_COLLECTION) \
                .filterDate(date_start, date_end) \
                .filterBounds(geometry) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_THRESHOLD))

            image = collection.sort('system:time_start', False).sort('CLOUDY_PIXEL_PERCENTAGE').first()

            if image is None or image.bandNames().size().getInfo() == 0:
                _log_json("WARNING", "No suitable image found for the specified region and date range after filtering.",
                          bbox=bbox, date_start=date_start, date_end=date_end, cloud_cover_threshold=CLOUD_COVER_THRESHOLD)
                return None

            _log_json("INFO", "Found a GEE image for processing.",
                      image_id=image.get('system:index').getInfo(),
                      cloud_percentage=image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo(),
                      image_date=datetime.fromtimestamp(image.get('system:time_start').getInfo() / 1000).strftime('%Y-%m-%d'))

            rgb_bands = ['B4', 'B3', 'B2']
            available_bands = image.bandNames().getInfo()
            if not all(band in available_bands for band in rgb_bands):
                _log_json("ERROR", "Selected GEE image is missing one or more required RGB bands (B4, B3, B2).",
                          image_id=image.get('system:index').getInfo(), available_bands=available_bands)
                return None

            rgb_image = image.select(rgb_bands)
            rgb_image_scaled = rgb_image.unitScale(0, 10000).multiply(255).toByte()

            return rgb_image_scaled
        except ee.EEException as e:
            _log_json("ERROR", "GEE error retrieving composite image.",
                      bbox=bbox, date_start=date_start, date_end=date_end, error=str(e))
            return None
        except Exception as e:
             _log_json("ERROR", "Unexpected error retrieving composite image from GEE.",
                      bbox=bbox, date_start=date_start, date_end=date_end, error=str(e), error_type=type(e).__name__)
             return None

    def acquire_and_export_imagery(self, monitored_regions: List[Dict[str, Any]], acquisition_date: Optional[str] = None) -> Dict[str, str]:
        """
        Acquires satellite imagery for each monitored region and exports it to GCS.

        Args:
            monitored_regions (List[Dict[str, Any]]): List of region dictionaries
                                                    with 'id' and 'bbox'.
            acquisition_date (Optional[str]): The target date for imagery in 'YYYY-MM-DD' format.
                                              If None, defaults to yesterday to ensure data availability.

        Returns:
            Dict[str, str]: A dictionary mapping monitored_region_id to its GCS image URI.
                            URIs are for *pending* exports. An empty dict if all fail.
        """
        if acquisition_date:
            try:
                target_date_obj = datetime.strptime(acquisition_date, '%Y-%m-%d')
            except ValueError:
                _log_json("ERROR", "Invalid acquisition_date format. Must be YYYY-MM-DD.", provided_date=acquisition_date)
                return {}
        else:
            target_date_obj = datetime.utcnow() - timedelta(days=1)
            acquisition_date = target_date_obj.strftime('%Y-%m-%d')

        date_start_str = target_date_obj.strftime('%Y-%m-%d')
        date_end_str = (target_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

        _log_json("INFO", "Starting satellite imagery acquisition.",
                  target_date_for_imagery=date_start_str, gee_date_filter_range=f"[{date_start_str}, {date_end_str})")

        exported_image_uris: Dict[str, str] = {}

        for region in monitored_regions:
            region_id = region["id"]
            region_bbox = region["bbox"]
            image_filename_stem = f"wildfire_imagery_{region_id}_{date_start_str.replace('-', '')}"
            gcs_file_prefix_for_export = f"{GCS_IMAGE_OUTPUT_PREFIX}{image_filename_stem}"
            expected_gcs_image_uri = f"gs://{self.gcs_bucket_name}/{gcs_file_prefix_for_export}.tif"

            _log_json("INFO", "Processing imagery for region.", region_id=region_id, bbox=region_bbox)

            image_to_export = self._get_latest_composite_image(region_bbox, date_start_str, date_end_str)

            if image_to_export is None:
                _log_json("WARNING", f"Skipping imagery export for region '{region_id}' as no suitable GEE image was found.",
                          region_id=region_id)
                continue

            try:
                export_geometry = ee.Geometry.Rectangle(region_bbox).getInfo()['coordinates']

                task = ee.batch.Export.image.toCloudStorage(
                    image=image_to_export,
                    description=f"Export_{image_filename_stem}",
                    bucket=self.gcs_bucket_name,
                    fileNamePrefix=gcs_file_prefix_for_export,
                    scale=10,
                    region=export_geometry,
                    fileFormat='GeoTIFF',
                    # --- MODIFICATION ---
                    # Increased maxPixels to allow larger exports like australia_southeast
                    # Original error indicated ~1.5e10 pixels. Setting to 2e10 for buffer.
                    maxPixels=2e10
                    # --- END MODIFICATION ---
                )
                task.start()
                _log_json("INFO", "GEE export task initiated.",
                          region_id=region_id, task_id=task.id, task_status=task.status(),
                          target_gcs_uri=expected_gcs_image_uri)

                exported_image_uris[region_id] = expected_gcs_image_uri

            except ee.EEException as e:
                _log_json("ERROR", f"GEE export task failed to start for region {region_id} (EEException).",
                          region_id=region_id, error=str(e))
            except Exception as e:
                _log_json("ERROR", f"An unexpected error occurred while initiating GEE export for region {region_id}.",
                          region_id=region_id, error=str(e), error_type=type(e).__name__)

        if not exported_image_uris:
            _log_json("WARNING", "No GEE export tasks were successfully initiated for any region.")
        else:
            _log_json("INFO", "Satellite imagery acquisition process complete. Review GEE Task Manager for export status.",
                      total_tasks_initiated=len(exported_image_uris),
                      initiated_uris=exported_image_uris)
        return exported_image_uris

# --- Example Usage (for local testing) ---
if __name__ == "__main__":
    from src.common.config import MONITORED_REGIONS

    _log_json("INFO", "Starting local test for SatelliteImageryAcquirer.")

    if not os.environ.get("GCP_PROJECT_ID"):
        _log_json("WARNING", "GCP_PROJECT_ID environment variable not set. GEE might use a default project. "
                             "Set it if you encounter permission issues or want to bill to a specific project.")

    if GCS_BUCKET_NAME == "fire-app-bucket": # Replace with your actual bucket if different for testing
        _log_json("INFO", f"Using configured GCS_BUCKET_NAME: {GCS_BUCKET_NAME} for local test.")
    else:
        _log_json("ERROR", f"GCS_BUCKET_NAME in src/common/config.py ('{GCS_BUCKET_NAME}') "
                           "does not match expected 'fire-app-bucket'. Please verify.")
        # exit(1) # Consider exiting if bucket name is critical for the test

    try:
        acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        test_acquisition_date = (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')

        _log_json("INFO", f"Attempting to acquire imagery for date: {test_acquisition_date} (local test)")
        image_uris = acquirer.acquire_and_export_imagery(MONITORED_REGIONS, acquisition_date=test_acquisition_date)

        if image_uris:
            _log_json("INFO", "GEE image export tasks initiated (local test). Check GEE Task Manager and GCS bucket.",
                      initiated_uris=image_uris)
        else:
            _log_json("WARNING", "No GEE image export tasks were initiated (local test). Check previous logs for reasons.")

    except ValueError as e:
        _log_json("ERROR", f"Configuration Error during local test: {e}")
    except RuntimeError as e:
        _log_json("ERROR", f"GEE Initialization Error during local test: {e}")
    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during satellite imagery acquisition example (local test).",
                  error=str(e), error_type=type(e).__name__)
