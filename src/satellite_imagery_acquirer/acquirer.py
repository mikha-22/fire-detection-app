# src/satellite_imagery_acquirer/acquirer.py

import os
import json
import logging
import time # Added for the main block
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import ee
from google.cloud import storage # Added for the main block
from src.common.config import GCS_PATHS, FILE_NAMES, GCS_BUCKET_NAME

# --- Module-level Constants ---
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
CLOUD_COVER_THRESHOLD = 95

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(), "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": "SatelliteImageryAcquirer", **kwargs
    }
    # Use indent for local testing readability
    indent = 2 if __name__ == "__main__" else None
    print(json.dumps(log_entry, indent=indent))

class SatelliteImageryAcquirer:
    # ... (The entire class definition remains exactly the same as before) ...
    def __init__(self, gcs_bucket_name: str):
        if not gcs_bucket_name:
            _log_json("CRITICAL", "GCS_BUCKET_NAME is missing or empty.")
            raise ValueError("GCS_BUCKET_NAME must be configured.")
        self.gcs_bucket_name = gcs_bucket_name

        try:
            # Check if GEE is already initialized to avoid errors on re-runs
            if not ee.data._credentials:
                 ee.Initialize(project=os.environ.get("GCP_PROJECT_ID"))
                 _log_json("INFO", "Google Earth Engine initialized successfully.",
                           gcp_project_id=os.environ.get("GCP_PROJECT_ID"))
            else:
                 _log_json("INFO", "Google Earth Engine already initialized.")
        except Exception as e:
            _log_json("CRITICAL", "An unexpected error during GEE initialization.", error=str(e))
            raise RuntimeError(f"GEE initialization failed: {e}")

    def _harmonize_band_names(self, image):
        landsat_bands = ['SR_B4', 'SR_B3', 'SR_B2']
        sentinel_bands = ['B4', 'B3', 'B2']
        return image.select(landsat_bands).rename(sentinel_bands)

    def _get_best_image(self, collection: ee.ImageCollection, geometry: ee.Geometry, source_name: str) -> Optional[ee.Image]:
        cloud_property = 'CLOUDY_PIXEL_PERCENTAGE' if source_name == 'sentinel2' else 'CLOUD_COVER'
        best_image_candidate = collection.filterBounds(geometry).sort(cloud_property).first()
        
        if not best_image_candidate.getInfo():
            _log_json("WARNING", f"No suitable image found for source: {source_name}")
            return None

        try:
            cloud_cover_property = best_image_candidate.get(cloud_property)
            cloud_cover = cloud_cover_property.getInfo() if cloud_cover_property else None
        except Exception:
            cloud_cover = None
        
        if cloud_cover is not None and isinstance(cloud_cover, (int, float)):
            _log_json("INFO", f"Found best {source_name} image with approx {cloud_cover:.2f}% cloud cover.")
        else:
            _log_json("WARNING", f"Found {source_name} image but it is missing or has invalid '{cloud_property}' property value. Proceeding anyway.")
            
        return best_image_candidate.select(['B4', 'B3', 'B2']).unitScale(0, 16000).multiply(255).toByte()

    def acquire_and_export_imagery(self, monitored_regions: List[Dict[str, Any]],
                                  acquisition_date: Optional[str] = None) -> List[ee.batch.Task]:
        if not acquisition_date:
            acquisition_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        target_date_obj = datetime.strptime(acquisition_date, '%Y-%m-%d')
        date_end_obj = target_date_obj + timedelta(days=1)
        date_start_obj = target_date_obj - timedelta(days=13)
        date_start_str = date_start_obj.strftime('%Y-%m-%d')
        date_end_str = date_end_obj.strftime('%Y-%m-%d')

        _log_json("INFO", "Starting multi-source satellite imagery acquisition.",
                  target_date_for_imagery=acquisition_date,
                  gee_date_filter_range=f"[{date_start_str}, {date_end_str})")
        
        s2_collection = ee.ImageCollection(SENTINEL2_COLLECTION).filterDate(date_start_str, date_end_str)
        l8_collection = ee.ImageCollection(LANDSAT8_COLLECTION).filterDate(date_start_str, date_end_str).map(self._harmonize_band_names)
        l9_collection = ee.ImageCollection(LANDSAT9_COLLECTION).filterDate(date_start_str, date_end_str).map(self._harmonize_band_names)
        landsat_collection = l8_collection.merge(l9_collection)

        image_sources = {"sentinel2": s2_collection, "landsat": landsat_collection}
        export_tasks = []

        for region in monitored_regions:
            region_id = region["id"]
            geometry = ee.Geometry.Rectangle(region["bbox"])

            for source_name, collection in image_sources.items():
                image_to_export = self._get_best_image(collection, geometry, source_name)
                if not image_to_export:
                    _log_json("WARNING", f"Skipping export for {region_id} from {source_name}: no image found.")
                    continue

                try:
                    gcs_file_prefix = f"{GCS_PATHS['SATELLITE_IMAGERY']}/{acquisition_date}/{region_id}_{source_name}"

                    task = ee.batch.Export.image.toCloudStorage(
                        image=image_to_export,
                        description=f"Export_{region_id}_{source_name}_{acquisition_date.replace('-', '')}",
                        bucket=self.gcs_bucket_name,
                        fileNamePrefix=gcs_file_prefix,
                        dimensions='4096x4096',
                        region=geometry.getInfo()['coordinates'],
                        fileFormat='GeoTIFF',
                        maxPixels=2e10
                    )
                    task.start()
                    
                    task.cluster_metadata = {
                        "cluster_id": region_id, "source": source_name, "image_bbox": region["bbox"]
                    }
                    export_tasks.append(task)
                    
                    _log_json("INFO", "GEE export task initiated.", region_id=region_id, source=source_name, task_id=task.id)
                except Exception as e:
                    _log_json("ERROR", f"Failed to initiate GEE export for {region_id}, {source_name}.", error=str(e))
        
        return export_tasks

# ==============================================================================
# === NEW SECTION: Code to make this script directly runnable for testing    ===
# ==============================================================================

def main():
    """
    Main function to run the image acquisition test.
    This function is executed when the script is run directly.
    """
    _log_json("INFO", "Starting local image acquisition test.")

    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    _log_json("INFO", f"Processing for run_date: {run_date}")

    # --- Step 1: Load Incident Data from GCS ---
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    incidents_blob_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date}/{FILE_NAMES['incident_data']}"

    try:
        incidents_content = bucket.blob(incidents_blob_path).download_as_string()
        incidents = [json.loads(line) for line in incidents_content.decode('utf-8').strip().split('\n')]
        _log_json("INFO", f"Successfully loaded {len(incidents)} incidents from GCS.", path=f"gs://{GCS_BUCKET_NAME}/{incidents_blob_path}")
    except Exception as e:
        _log_json("ERROR", f"Failed to read incidents file. Ensure it exists for today's date.", path=f"gs://{GCS_BUCKET_NAME}/{incidents_blob_path}", error=str(e))
        return

    if not incidents:
        _log_json("WARNING", "Incidents file was empty or could not be parsed. Exiting.")
        return

    # --- Step 2: Prepare Bounding Boxes for Each Incident ---
    BOUNDING_BOX_PADDING_DEGREES = 0.1
    regions_to_acquire = []
    for incident in incidents:
        cluster_id, hotspots = incident.get("cluster_id"), incident.get("hotspots", [])
        if not all([cluster_id, hotspots]): continue
        
        longitudes = [h['geometry']['coordinates'][0] for h in hotspots]
        latitudes = [h['geometry']['coordinates'][1] for h in hotspots]
        
        cluster_bbox = [
            min(longitudes) - BOUNDING_BOX_PADDING_DEGREES, min(latitudes) - BOUNDING_BOX_PADDING_DEGREES,
            max(longitudes) + BOUNDING_BOX_PADDING_DEGREES, max(latitudes) + BOUNDING_BOX_PADDING_DEGREES
        ]
        regions_to_acquire.append({"id": cluster_id, "name": f"Incident {cluster_id}", "bbox": cluster_bbox})

    _log_json("INFO", f"Prepared {len(regions_to_acquire)} regions for image acquisition.")

    # --- Step 3: Acquire Imagery and Wait for Completion ---
    try:
        # Instantiate the class defined in this same file
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        export_tasks = imagery_acquirer.acquire_and_export_imagery(regions_to_acquire, acquisition_date=run_date)

        if not export_tasks:
            _log_json("ERROR", "No imagery export tasks were initiated.")
            return

        _log_json("INFO", f"Waiting for {len(export_tasks)} GEE export tasks to complete. This may take several minutes...")
        
        completed_uris = []
        for task in export_tasks:
            task_start_time = time.time()
            while task.active():
                status = task.status()
                _log_json("INFO", f"Task {task.id} ({status.get('description')}) is {status['state']}...")
                time.sleep(30)
            
            final_status = task.status()
            elapsed = time.time() - task_start_time

            if final_status['state'] == 'COMPLETED':
                destination_uris = final_status.get('destination_uris', [])
                _log_json("SUCCESS", f"Task {task.id} completed in {elapsed:.2f}s.", uris=destination_uris)
                completed_uris.extend(destination_uris)
            else:
                _log_json("ERROR", f"Task {task.id} failed after {elapsed:.2f}s.", full_status=final_status)

        _log_json("INFO", "--- TEST COMPLETE ---", total_files_created=len(completed_uris))
        print("\nFinal list of created files:")
        for uri in completed_uris:
            print(uri)

    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during the test.", error=str(e), exc_info=True)


if __name__ == "__main__":
    print("--- Running Satellite Imagery Acquirer as a standalone script ---")
    
    # --- Configuration for Local Test ---
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'

    print("NOTE: This script assumes you have run 'gcloud auth application-default login'")
    print("      and that an incidents.jsonl file exists for today's date in GCS.")
    
    main()
    
    print("--- Script Finished ---")
