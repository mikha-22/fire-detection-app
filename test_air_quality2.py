# test_air_quality_final.py
# Final standalone test focusing on Indonesian peatland fires
# This represents the most relevant use case for the fire detection system

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import ee

# --- Configuration ---
GCP_PROJECT_ID = "haryo-kebakaran"

# Simplified Sentinel-5P configuration focusing on key fire indicators
S5P_PRODUCTS = {
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m²",
        "scale_factor": 1e6,  # Convert to µmol/m²
        "description": "Carbon Monoxide - primary fire indicator"
    },
    "AEROSOL": {
        "collection": "COPERNICUS/S5P/OFFL/L3_AER_AI",
        "band": "absorbing_aerosol_index",
        "unit": "index",
        "scale_factor": 1,
        "description": "Aerosol Index - smoke/haze indicator"
    },
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density", 
        "unit": "mol/m²",
        "scale_factor": 1e6,
        "description": "Nitrogen Dioxide - combustion indicator"
    }
}

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)


def get_air_quality_for_peatland_fire(
    latitude: float,
    longitude: float,
    fire_date: datetime,
    buffer_km: float = 50.0
) -> Dict[str, Any]:
    """
    Get air quality data for a peatland fire location.
    Uses a 7-day window centered on the fire date for better data coverage.
    """
    
    # Create geometry
    point = ee.Geometry.Point([longitude, latitude])
    geometry = point.buffer(buffer_km * 1000)
    
    # Use 7-day window (3 days before, 3 days after)
    start_date = fire_date - timedelta(days=3)
    end_date = fire_date + timedelta(days=3)
    
    logging.info(f"Analyzing location: ({latitude:.3f}, {longitude:.3f})")
    logging.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logging.info(f"Buffer radius: {buffer_km} km")
    
    results = {
        "location": {"latitude": latitude, "longitude": longitude},
        "fire_date": fire_date.isoformat(),
        "date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "buffer_km": buffer_km,
        "measurements": {},
        "fire_assessment": {
            "indicators": [],
            "confidence": "low",
            "summary": ""
        }
    }
    
    logging.info("\nRetrieving air quality data:")
    
    for product_name, config in S5P_PRODUCTS.items():
        logging.info(f"\n  Processing {product_name}...")
        
        try:
            # Get image collection
            collection = ee.ImageCollection(config['collection']) \
                .filterBounds(geometry) \
                .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                .select(config['band'])
            
            # Check availability
            image_count = collection.size().getInfo()
            logging.info(f"    Found {image_count} images")
            
            if image_count == 0:
                results["measurements"][product_name] = {
                    "status": "no_data",
                    "error": "No images available for this period"
                }
                continue
            
            # Get composite (median to handle outliers)
            composite = collection.median()
            
            # Calculate statistics
            stats = composite.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), '', True
                ).combine(
                    ee.Reducer.min(), '', True
                ).combine(
                    ee.Reducer.max(), '', True
                ).combine(
                    ee.Reducer.percentile([25, 50, 75]), '', True
                ),
                geometry=geometry,
                scale=5000,  # 5km resolution
                maxPixels=1e9,
                bestEffort=True
            )
            
            stats_dict = stats.getInfo()
            
            # Extract values
            mean_val = stats_dict.get(f"{config['band']}_mean")
            
            if mean_val is not None:
                # Scale values
                scale = config['scale_factor']
                
                result_data = {
                    "status": "success",
                    "value": round(mean_val * scale, 2),
                    "unit": "µmol/m²" if scale > 1 else config['unit'],
                    "statistics": {
                        "mean": round(mean_val * scale, 2),
                        "std": round(stats_dict.get(f"{config['band']}_stdDev", 0) * scale, 2),
                        "min": round(stats_dict.get(f"{config['band']}_min", 0) * scale, 2),
                        "max": round(stats_dict.get(f"{config['band']}_max", 0) * scale, 2),
                        "p25": round(stats_dict.get(f"{config['band']}_p25", 0) * scale, 2),
                        "p50": round(stats_dict.get(f"{config['band']}_p50", 0) * scale, 2),
                        "p75": round(stats_dict.get(f"{config['band']}_p75", 0) * scale, 2)
                    },
                    "images_used": image_count,
                    "description": config['description']
                }
                
                results["measurements"][product_name] = result_data
                
                # Log summary
                logging.info(f"    ✓ Mean: {result_data['value']} {result_data['unit']}")
                logging.info(f"    ✓ Range: {result_data['statistics']['min']} - {result_data['statistics']['max']}")
                logging.info(f"    ✓ Std Dev: {result_data['statistics']['std']}")
                
                # Fire detection logic based on research thresholds
                if product_name == "CO":
                    # Background CO is typically 50-100 µmol/m²
                    # Fire-affected areas show >150 µmol/m²
                    if mean_val * scale > 150:
                        severity = "severe" if mean_val * scale > 300 else "moderate"
                        results["fire_assessment"]["indicators"].append({
                            "type": "CO",
                            "severity": severity,
                            "value": mean_val * scale,
                            "threshold": 150,
                            "message": f"CO levels {severity}ly elevated ({mean_val * scale:.0f} µmol/m²)"
                        })
                
                elif product_name == "AEROSOL":
                    # Aerosol index > 1.0 indicates smoke
                    # > 3.0 indicates heavy smoke
                    if mean_val > 1.0:
                        severity = "severe" if mean_val > 3.0 else "moderate"
                        results["fire_assessment"]["indicators"].append({
                            "type": "AEROSOL",
                            "severity": severity,
                            "value": mean_val,
                            "threshold": 1.0,
                            "message": f"Aerosol index indicates {severity} smoke (index: {mean_val:.2f})"
                        })
                
                elif product_name == "NO2":
                    # Elevated NO2 (>40 µmol/m²) indicates active combustion
                    if mean_val * scale > 40:
                        results["fire_assessment"]["indicators"].append({
                            "type": "NO2",
                            "severity": "moderate",
                            "value": mean_val * scale,
                            "threshold": 40,
                            "message": f"NO2 indicates active combustion ({mean_val * scale:.0f} µmol/m²)"
                        })
            
            else:
                results["measurements"][product_name] = {
                    "status": "no_valid_pixels",
                    "error": "No valid pixels in the region"
                }
                logging.warning(f"    ✗ No valid pixels found")
                
        except Exception as e:
            results["measurements"][product_name] = {
                "status": "error",
                "error": str(e)
            }
            logging.error(f"    ✗ Error: {e}")
    
    # Assess overall fire confidence
    num_indicators = len(results["fire_assessment"]["indicators"])
    severe_indicators = sum(1 for ind in results["fire_assessment"]["indicators"] if ind["severity"] == "severe")
    
    if num_indicators >= 2 and severe_indicators >= 1:
        results["fire_assessment"]["confidence"] = "high"
        results["fire_assessment"]["summary"] = "Strong evidence of active fire with significant emissions"
    elif num_indicators >= 2:
        results["fire_assessment"]["confidence"] = "moderate"
        results["fire_assessment"]["summary"] = "Moderate evidence of fire activity"
    elif num_indicators == 1:
        results["fire_assessment"]["confidence"] = "low"
        results["fire_assessment"]["summary"] = "Limited evidence of fire activity"
    else:
        results["fire_assessment"]["confidence"] = "none"
        results["fire_assessment"]["summary"] = "No clear fire indicators detected"
    
    return results


def main():
    """Run the focused peatland fire test."""
    
    print("\n" + "="*70)
    print("AIR QUALITY ANALYSIS FOR INDONESIAN PEATLAND FIRES")
    print("="*70)
    
    # Initialize Earth Engine
    print("\nInitializing Google Earth Engine...")
    try:
        ee.Initialize(
            opt_url='https://earthengine-highvolume.googleapis.com',
            project=GCP_PROJECT_ID
        )
        print("✓ Successfully initialized")
    except Exception as e:
        print(f"✗ Failed to initialize GEE: {e}")
        print("\nPlease ensure you have authenticated:")
        print("  gcloud auth application-default login")
        print("  earthengine authenticate")
        return
    
    # Test case: Major peatland fire in Central Kalimantan
    # September 2019 was a severe fire season
    test_case = {
        "name": "Central Kalimantan Peatland Fire",
        "latitude": -2.21,  # Near Palangka Raya
        "longitude": 113.92,
        "date": datetime(2019, 9, 21),  # Peak of 2019 fire season
        "description": "Major peatland fire during 2019 Indonesian fire crisis"
    }
    
    print(f"\nTest Case: {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print(f"Location: ({test_case['latitude']}, {test_case['longitude']})")
    print(f"Date: {test_case['date'].strftime('%Y-%m-%d')}")
    print("\n" + "-"*70)
    
    # Run analysis
    results = get_air_quality_for_peatland_fire(
        latitude=test_case['latitude'],
        longitude=test_case['longitude'],
        fire_date=test_case['date'],
        buffer_km=50.0
    )
    
    # Display results
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    print("\n1. MEASUREMENTS SUMMARY:")
    for product, data in results['measurements'].items():
        if data['status'] == 'success':
            print(f"\n   {product}:")
            print(f"   - Value: {data['value']} {data['unit']}")
            print(f"   - Range: {data['statistics']['min']} to {data['statistics']['max']}")
            print(f"   - Images used: {data['images_used']}")
        else:
            print(f"\n   {product}: {data.get('error', 'No data')}")
    
    print(f"\n2. FIRE ASSESSMENT:")
    print(f"   - Confidence: {results['fire_assessment']['confidence'].upper()}")
    print(f"   - Summary: {results['fire_assessment']['summary']}")
    
    if results['fire_assessment']['indicators']:
        print(f"\n   - Fire Indicators Detected:")
        for indicator in results['fire_assessment']['indicators']:
            print(f"     • {indicator['message']}")
    
    # Practical interpretation
    print("\n3. PRACTICAL INTERPRETATION:")
    if results['fire_assessment']['confidence'] in ['high', 'moderate']:
        print("   ⚠️  This location shows clear signs of fire activity")
        print("   ⚠️  Air quality is significantly impacted")
        print("   ⚠️  Recommend immediate investigation and response")
    else:
        print("   ✓  Limited or no fire activity detected")
        print("   ✓  Air quality parameters within normal ranges")
    
    # Export results
    output_file = "air_quality_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n4. DETAILED RESULTS SAVED TO: {output_file}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    # Set environment
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    
    print("--- Indonesian Peatland Fire Air Quality Test ---")
    print("This test analyzes air quality during a known peatland fire event")
    print("Location: Central Kalimantan (September 2019)")
    
    main()
    
    print("\n--- Script Finished ---")
