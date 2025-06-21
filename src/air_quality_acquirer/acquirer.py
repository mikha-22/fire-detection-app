# src/air_quality_acquirer/acquirer.py
"""
Air Quality Data Acquirer for Wildfire Detection System.
Retrieves and assesses Sentinel-5P air quality data from Google Earth Engine.
This version is corrected to ONLY use Near Real-Time (NRTI) data.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import ee

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "haryo-kebakaran")

# --- CORRECTED: Use ONLY Near Real-Time (NRTI) collections ---
S5P_FIRE_PRODUCTS = {
    "CO": {
        "collection": "COPERNICUS/S5P/NRTI/L3_CO",
        "band": "CO_column_number_density",
        "unit": "µmol/m²", "scale_factor": 1e6,
        "fire_threshold": 50000, "severe_threshold": 100000,
        "description": "Carbon Monoxide"
    },
    "AEROSOL": {
        "collection": "COPERNICUS/S5P/NRTI/L3_AER_AI",
        "band": "absorbing_aerosol_index",
        "unit": "index", "scale_factor": 1,
        "fire_threshold": 1.0, "severe_threshold": 3.0,
        "description": "Aerosol Index"
    },
    "NO2": {
        "collection": "COPERNICUS/S5P/NRTI/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
        "unit": "µmol/m²", "scale_factor": 1e6,
        "fire_threshold": 20, "severe_threshold": 50,
        "description": "Nitrogen Dioxide"
    }
}

class AirQualityAcquirer:
    def __init__(self):
        logging.info("AirQualityAcquirer initialized (NRTI-only).")
        try:
            if not ee.data._credentials:
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project=GCP_PROJECT_ID)
                logging.info("Successfully initialized Google Earth Engine client.")
            else:
                logging.info("Google Earth Engine client already initialized.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to initialize GEE: {e}", exc_info=True)
            raise

    def get_air_quality_for_incident(self, latitude: float, longitude: float, incident_timestamp: datetime, buffer_km: float = 50.0) -> Dict[str, Any]:
        # Use a 3-day window to maximize chance of finding NRTI data
        start_date = incident_timestamp - timedelta(days=3)
        end_date = incident_timestamp + timedelta(days=3)
        point = ee.Geometry.Point([longitude, latitude])
        geometry = point.buffer(buffer_km * 1000)
        
        logging.info(f"Fetching NRTI air quality for incident at ({latitude:.3f}, {longitude:.3f})")
        
        results = {"measurements": {}, "fire_indicators": [], "air_quality_severity": "unknown"}
        
        for name, config in S5P_FIRE_PRODUCTS.items():
            try:
                collection = ee.ImageCollection(config['collection']).filterBounds(geometry).filterDate(ee.Date(start_date), ee.Date(end_date)).select(config['band'])
                
                if collection.size().getInfo() == 0:
                    results["measurements"][name] = {"status": "no_data", "error": "No NRTI images available for this time range."}
                    continue
                
                composite = collection.median()
                stats = composite.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=5000, maxPixels=1e9, bestEffort=True)
                mean_val = stats.get(config['band']).getInfo()
                
                if mean_val is not None:
                    scaled_mean = mean_val * config['scale_factor']
                    results["measurements"][name] = {"status": "success", "value": round(scaled_mean, 2), "unit": config['unit'], "source": config['collection']}
                    
                    if scaled_mean > config['severe_threshold']:
                        results["fire_indicators"].append({"product": name, "severity": "severe", "value": scaled_mean, "message": f"{config['description']} severely elevated"})
                    elif scaled_mean > config['fire_threshold']:
                        results["fire_indicators"].append({"product": name, "severity": "moderate", "value": scaled_mean, "message": f"{config['description']} moderately elevated"})
                else:
                    results["measurements"][name] = {"status": "no_valid_pixels", "error": "No valid pixels in NRTI data for region"}
            except Exception as e:
                logging.warning(f"Could not retrieve GEE NRTI data for {name}: {e}")
                results["measurements"][name] = {"status": "error", "error": str(e)}
        
        severe_count = sum(1 for ind in results["fire_indicators"] if ind["severity"] == "severe")
        moderate_count = sum(1 for ind in results["fire_indicators"] if ind["severity"] == "moderate")
        
        if severe_count >= 2 or (severe_count == 1 and moderate_count >= 1): results["air_quality_severity"] = "severe"
        elif severe_count >= 1 or moderate_count >= 2: results["air_quality_severity"] = "high"
        elif moderate_count == 1: results["air_quality_severity"] = "moderate"
        else: results["air_quality_severity"] = "low"
        
        return results
