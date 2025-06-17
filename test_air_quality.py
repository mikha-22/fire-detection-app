# test_air_quality_standalone.py
# A fully standalone script to test air quality data acquisition from Google Earth Engine.
# This version uses a more appropriate QA threshold for CO event detection.

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import pandas as pd
import ee

# --- Configuration ---
GCP_PROJECT_ID = "haryo-kebakaran"

S5P_PRODUCTS = {
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m²",
        "qa_band": "qa_value",
        # --- KEY CHANGE: Use a more lenient QA threshold for CO event detection ---
        "qa_threshold": 0.25, 
        "scale_factor": 1e6,
        "description": "Carbon Monoxide"
    },
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2", 
        "band": "tropospheric_NO2_column_number_density",
        "unit": "mol/m²",
        "qa_band": "qa_value",
        "qa_threshold": 0.75,
        "scale_factor": 1e6,
        "description": "Nitrogen Dioxide"
    },
    "AEROSOL": {
        "collection": "COPERNICUS/S5P/OFFL/L3_AER_AI",
        "band": "absorbing_aerosol_index",
        "unit": "index",
        "qa_band": None,
        "qa_threshold": None,
        "scale_factor": 1,
        "description": "Aerosol Index"
    }
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_air_quality_data(
    latitude: float, 
    longitude: float, 
    timestamp: datetime,
    buffer_km: float = 50.0,
    products: Optional[List[str]] = None
) -> Dict[str, Any]:
    if products is None:
        products = list(S5P_PRODUCTS.keys())
        
    point = ee.Geometry.Point([longitude, latitude])
    geometry = point.buffer(buffer_km * 1000)
    
    start_date = timestamp - timedelta(days=1)
    end_date = timestamp + timedelta(days=1)
    
    logging.info(f"Fetching air quality data for lat={latitude}, lon={longitude}")
    
    results = {
        "timestamp": timestamp.isoformat(),
        "location": {"latitude": latitude, "longitude": longitude},
        "buffer_km": buffer_km,
        "measurements": {},
        "air_quality_assessment": {"fire_indicators": [], "severity": "unknown"}
    }
    
    for product_name in products:
        params = S5P_PRODUCTS[product_name]
        try:
            collection = ee.ImageCollection(params['collection']) \
                .filterBounds(geometry) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .select(params['band'])
            
            if params['qa_band']:
                collection = collection.filter(
                    ee.Filter.gt(params['qa_band'], params['qa_threshold'])
                )
            
            image = collection.mean()
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=5000,
                maxPixels=1e9
            )
            stats_dict = stats.getInfo()
            
            mean_value = stats_dict.get(params['band'])
            
            if mean_value is not None:
                scaled_mean = mean_value * params['scale_factor']
                results["measurements"][product_name] = {
                    "value": round(scaled_mean, 3),
                    "unit": params['unit'].replace("mol/m²", "µmol/m²") if params['scale_factor'] > 1 else params['unit'],
                    "description": params['description']
                }
                logging.info(f"Successfully retrieved {product_name}: {scaled_mean:.3f}")
                
                if product_name == "CO" and scaled_mean > 100:
                    results["air_quality_assessment"]["fire_indicators"].append("Elevated CO levels")
                elif product_name == "AEROSOL" and mean_value > 1.0:
                    results["air_quality_assessment"]["fire_indicators"].append("High aerosol index")
                    
            else:
                results["measurements"][product_name] = {"value": None, "error": "No data available"}
                logging.warning(f"No data for {product_name}")
                
        except Exception as e:
            logging.error(f"Failed to retrieve {product_name}: {e}")
            results["measurements"][product_name] = {"value": None, "error": str(e)}
    
    if len(results["air_quality_assessment"]["fire_indicators"]) >= 1:
        results["air_quality_assessment"]["severity"] = "high"
    else:
        results["air_quality_assessment"]["severity"] = "low"
    
    return results


def run_air_quality_test():
    print("\n" + "=" * 60)
    print("AIR QUALITY DATA ACQUISITION TEST")
    print("=" * 60)
    
    try:
        ee.Initialize(project=GCP_PROJECT_ID)
        print("✓ Successfully initialized Google Earth Engine")
    except Exception as e:
        print(f"✗ CRITICAL: Failed to initialize GEE. Error: {e}")
        return

    print("\n\n=== Test 1: Amazon Rainforest Fires (August 2019) ===")
    
    test_result = get_air_quality_data(
        latitude=-8.5,
        longitude=-62.5,
        timestamp=datetime(2019, 8, 20),
        buffer_km=50.0
    )
    
    print("\n--- Results ---")
    print(json.dumps(test_result, indent=2))
    
    print("\n--- Validation ---")
    if test_result['measurements']:
        valid_measurements = sum(1 for m in test_result['measurements'].values() if m.get('value') is not None)
        print(f"✓ Retrieved {valid_measurements} valid measurements.")
        if test_result['air_quality_assessment']['severity'] != 'low':
            print(f"✓ Air quality severity: {test_result['air_quality_assessment']['severity']}")
            print(f"✓ Fire indicators detected: {', '.join(test_result['air_quality_assessment']['fire_indicators'])}")

    print("\n\n" + "=" * 60)
    print("TEST COMPLETED")

if __name__ == "__main__":
    run_air_quality_test()
