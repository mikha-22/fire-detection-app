# src/satellite_imagery_acquirer/acquirer.py

import os
import logging
import ee # Google Earth Engine Python API
import json # For structured logging of dicts
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.common.config import GCS_BUCKET_NAME

# --- Configuration ---
GCS_IMAGE_OUTPUT_PREFIX = "raw_satellite_imagery/"
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_COVER_THRESHOLD = 20 # Still useful for pre-filtering before the median

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
        Retrieves a median composite image for a given bbox and date range.
        This is more robust against gaps and clouds than taking the .first() image.
        """
        try:
            geometry = ee.Geometry.Rectangle(bbox)

            # 1. Filter the collection by date, location, and a reasonable cloud cover.
            collection = ee.ImageCollection(SENTINEL2_COLLECTION) \
                .filterDate(date_start, date_end) \
                .filterBounds(geometry) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_THRESHOLD))

            # 2. Check if the collection is empty *before* creating the median composite.
            image_count = collection.size().getInfo()
            if image_count == 0:
                _log_json("WARNING", "No suitable images found in the date range to create a composite.",
                          bbox=bbox, date_start=date_start, date_end=date_end, cloud_cover_threshold=CLOUD_COVER_THRESHOLD)
                return None
            _log_json("INFO", f"Found {image_count} images in date range to create a median composite.")

            # 3. Create the median composite. This operation is powerful.
            # It takes all images in the collection and computes the median value for each pixel.
            median_composite = collection.median()

            # Ensure the resulting composite has the bands we need.
            rgb_bands = ['B4', 'B3', 'B2']
            available_bands = median_composite.bandNames().getInfo()
            if not all(band in available_bands for band in rgb_bands):
                _log_json("ERROR", "Resulting median composite is missing RGB bands.", available_bands=available_bands)
                return None

            # Scale the image for export as an 8-bit visual.
            rgb_image = median_composite.select(rgb_bands)
            rgb_image_scaled = rgb_image.unitScale(0, 10000).multiply(255).toByte()

            return rgb_image_scaled

        except Exception as e:
            _log_json("ERROR", "An unexpected error occurred in GEE while creating composite image.",
                      bbox=bbox, date_start=date_start, date_end=date_end, error=str(e), error_type=type(e).__name__)
            return None

    def acquire_and_export_imagery(self, monitored_regions: List[Dict[str, Any]], acquisition_date: Optional[str] = None) -> Dict[str, str]:
        """
        Acquires satellite imagery for each monitored region and exports it to GCS.
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

        # --- THE COMPROMISE: A 3-DAY LOOKBACK WINDOW ---
        # This is a great balance between recency and data quality.
        date_end_obj = target_date_obj + timedelta(days=1) # End of window is still yesterday
        date_start_obj = target_date_obj - timedelta(days=2) # Start of window is 2 days before
        date_start_str = date_start_obj.strftime('%Y-%m-%d')
        date_end_str = date_end_obj.strftime('%Y-%m-%d')
        # --- END MODIFICATION ---

        _log_json("INFO", "Starting satellite imagery acquisition.",
                  target_date_for_imagery=acquisition_date,
                  gee_date_filter_range=f"[{date_start_str}, {date_end_str})") # Log the new 3-day range

        exported_image_uris: Dict[str, str] = {}

        for region in monitored_regions:
            region_id = region["id"]
            region_bbox = region["bbox"]
            # The filename still uses the single target date for consistency.
            image_filename_stem = f"wildfire_imagery_{region_id}_{acquisition_date.replace('-', '')}"
            gcs_file_prefix_for_export = f"{GCS_IMAGE_OUTPUT_PREFIX}{image_filename_stem}"
            expected_gcs_image_uri = f"gs://{self.gcs_bucket_name}/{gcs_file_prefix_for_export}.tif"

            _log_json("INFO", "Processing imagery for region.", region_id=region_id, bbox=region_bbox)

            image_to_export = self._get_latest_composite_image(region_bbox, date_start_str, date_end_str)

            if image_to_export is None:
                _log_json("WARNING", f"Skipping export for region '{region_id}': no suitable GEE image found to create composite.", region_id=region_id)
                continue

            try:
                export_geometry = ee.Geometry.Rectangle(region_bbox).getInfo()['coordinates']

                task = ee.batch.Export.image.toCloudStorage(
                    image=image_to_export,
                    description=f"Export_{image_filename_stem}",
                    bucket=self.gcs_bucket_name,
                    fileNamePrefix=gcs_file_prefix_for_export,
                    scale=100,
                    region=export_geometry,
                    fileFormat='GeoTIFF',
                    maxPixels=2e10
                )
                task.start()
                _log_json("INFO", "GEE export task initiated.",
                          region_id=region_id, task_id=task.id,
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
            _log_json("INFO", "Satellite imagery acquisition process complete. Review GEE Task Manager for status.",
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
