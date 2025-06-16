# src/satellite_imagery_acquirer/acquirer.py

import os
import logging
import ee
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.common.config import GCS_BUCKET_NAME, GCS_PATHS

# --- Configuration ---
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
CLOUD_COVER_THRESHOLD = 95 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def _log_json(severity: str, message: str, **kwargs):
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
    Component 2: Obtains satellite imagery using the Google Earth Engine API 
    and exports it to Google Cloud Storage.
    """

    def __init__(self, gcs_bucket_name: str):
        if not gcs_bucket_name:
            _log_json("CRITICAL", "GCS_BUCKET_NAME is missing or empty.")
            raise ValueError("GCS_BUCKET_NAME must be configured to a valid bucket.")
        self.gcs_bucket_name = gcs_bucket_name

        try:
            ee.Initialize(project=os.environ.get("GCP_PROJECT_ID"))
            _log_json("INFO", "Google Earth Engine initialized successfully.", 
                     gcp_project_id=os.environ.get("GCP_PROJECT_ID"))
        except Exception as e:
            _log_json("CRITICAL", "An unexpected error occurred during GEE initialization.", 
                     error=str(e))
            raise RuntimeError(f"GEE initialization failed: {e}")

    def _harmonize_band_names(self, image):
        landsat_bands = ['SR_B4', 'SR_B3', 'SR_B2']
        sentinel_bands = ['B4', 'B3', 'B2']
        return image.select(landsat_bands).rename(sentinel_bands)

    def _get_latest_composite_image(self, bbox: List[float], date_start: str, date_end: str) -> Optional[ee.Image]:
        try:
            geometry = ee.Geometry.Rectangle(bbox)

            s2_collection = ee.ImageCollection(SENTINEL2_COLLECTION) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_THRESHOLD))
            
            l8_collection = ee.ImageCollection(LANDSAT8_COLLECTION) \
                .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_THRESHOLD)) \
                .map(self._harmonize_band_names)
            
            l9_collection = ee.ImageCollection(LANDSAT9_COLLECTION) \
                .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_THRESHOLD)) \
                .map(self._harmonize_band_names)

            merged_collection = ee.ImageCollection(s2_collection.merge(l8_collection).merge(l9_collection)) \
                .filterDate(date_start, date_end) \
                .filterBounds(geometry)

            best_image = merged_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

            image_id = ee.String(best_image.id()).getInfo()
            if image_id is None:
                _log_json("WARNING", "No images found in the merged collection.", 
                         bbox=bbox, date_range=f"[{date_start}, {date_end})")
                return None
            
            cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo() or best_image.get('CLOUD_COVER').getInfo()
            _log_json("INFO", f"Found best available image ({image_id}) with approx {cloud_cover:.2f}% cloud cover.")

            rgb_image_scaled = best_image.select(['B4', 'B3', 'B2']).unitScale(0, 16000).multiply(255).toByte()

            return rgb_image_scaled

        except ee.EEException as e:
            _log_json("ERROR", "A Google Earth Engine error occurred.", error=str(e))
            return None
        except Exception as e:
            _log_json("ERROR", "An unexpected error occurred while acquiring the best image.", error=str(e))
            return None

    def acquire_and_export_imagery(self, monitored_regions: List[Dict[str, Any]], 
                                  acquisition_date: Optional[str] = None) -> Dict[str, str]:
        if acquisition_date:
            try:
                target_date_obj = datetime.strptime(acquisition_date, '%Y-%m-%d')
            except ValueError:
                _log_json("ERROR", "Invalid acquisition_date format.", provided_date=acquisition_date)
                return {}
        else:
            target_date_obj = datetime.utcnow() - timedelta(days=1)
            acquisition_date = target_date_obj.strftime('%Y-%m-%d')

        date_end_obj = target_date_obj + timedelta(days=1)
        date_start_obj = target_date_obj - timedelta(days=13)
        date_start_str = date_start_obj.strftime('%Y-%m-%d')
        date_end_str = date_end_obj.strftime('%Y-%m-%d')

        _log_json("INFO", "Starting satellite imagery acquisition from Sentinel-2 & Landsat.",
                  target_date_for_imagery=acquisition_date,
                  gee_date_filter_range=f"[{date_start_str}, {date_end_str})")

        exported_image_uris: Dict[str, str] = {}

        for region in monitored_regions:
            region_id = region["id"]
            region_bbox = region["bbox"]
            
            # Use new path structure - just the filename in the satellite_imagery/date/ folder
            image_filename = f"{region_id}.tif"
            gcs_file_prefix = f"{GCS_PATHS['satellite_imagery']}/{acquisition_date}/{region_id}"
            expected_gcs_image_uri = f"gs://{self.gcs_bucket_name}/{gcs_file_prefix}.tif"

            _log_json("INFO", "Processing imagery for region.", region_id=region_id, bbox=region_bbox)

            image_to_export = self._get_latest_composite_image(region_bbox, date_start_str, date_end_str)

            if image_to_export is None:
                _log_json("WARNING", f"Skipping export for region '{region_id}': no suitable image found.", 
                         region_id=region_id)
                continue

            try:
                export_geometry = ee.Geometry.Rectangle(region_bbox).getInfo()['coordinates']
                task = ee.batch.Export.image.toCloudStorage(
                    image=image_to_export,
                    description=f"Export_{region_id}_{acquisition_date.replace('-', '')}",
                    bucket=self.gcs_bucket_name,
                    fileNamePrefix=gcs_file_prefix,
                    scale=20,
                    region=export_geometry,
                    fileFormat='GeoTIFF',
                    maxPixels=2e10
                )
                task.start()
                _log_json("INFO", "GEE export task initiated.",
                          region_id=region_id, task_id=task.id,
                          target_gcs_uri=expected_gcs_image_uri)
                exported_image_uris[region_id] = expected_gcs_image_uri

            except Exception as e:
                _log_json("ERROR", f"An unexpected error occurred while initiating GEE export for region {region_id}.",
                          region_id=region_id, error=str(e), error_type=type(e).__name__)

        if not exported_image_uris:
            _log_json("WARNING", "No GEE export tasks were successfully initiated.")
        else:
            _log_json("INFO", "Satellite imagery acquisition process complete. Review GEE Task Manager for status.",
                      total_tasks_initiated=len(exported_image_uris),
                      initiated_uris=exported_image_uris)
        return exported_image_uris
