import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import os

# --- 1. Configuration ---
DBSCAN_RADIUS_KM = 2
MIN_SAMPLES = 5
# --- Path now points to the geodata folder from your repo structure ---
SHAPEFILE_PATH = 'geodata/Indonesia_peat_lands.shp' 
BUFFER_METERS = 1000


# --- 2. Main Processing Function (USING REFERENCE LOGIC) ---
def analyze_peatland_clusters_daily(input_file, output_file):
    """
    Loads FIRMS data, filters for hotspots ON OR NEAR PEATLANDS (with a 1km buffer),
    clusters them daily, and saves a prioritized summary for the entire year.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} total hotspots from '{input_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return

    # --- Preprocessing and Standard Filtering ---
    df_filtered = df[df['confidence'].isin(['n', 'h'])].copy()
    df_filtered = df_filtered[df_filtered['type'] == 0]
    df_filtered['acq_date'] = pd.to_datetime(df_filtered['acq_date'])
    print(f"Filtered down to {len(df_filtered)} high-confidence vegetation fire hotspots.")

    # --- Load Shapefile ---
    try:
        peatlands_gdf = gpd.read_file(SHAPEFILE_PATH)
        print(f"Successfully loaded shapefile from '{SHAPEFILE_PATH}'.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read the shapefile. Make sure it's in the correct folder. Error: {e}")
        return

    # --- Hotspots GeoDataFrame ---
    hotspots_gdf = gpd.GeoDataFrame(
        df_filtered,
        geometry=gpd.points_from_xy(df_filtered.longitude, df_filtered.latitude),
        crs="EPSG:4326"
    )

    # --- ROBUST SPATIAL FILTERING LOGIC (from incident_detector) ---
    print(f"Applying a {BUFFER_METERS / 1000}km buffer to peatland polygons...")
    
    # 1. Project peatlands to Web Mercator (EPSG:3857) which uses meters
    peatlands_for_buffer = peatlands_gdf.to_crs(epsg=3857)
    
    # 2. Apply the buffer in meters
    peatlands_for_buffer['geometry'] = peatlands_for_buffer.geometry.buffer(BUFFER_METERS)
    
    # 3. Project the buffered peatlands back to the same CRS as the hotspots (EPSG:4326)
    buffered_peatlands_in_wgs84 = peatlands_for_buffer.to_crs(hotspots_gdf.crs)

    # 4. Perform the spatial join
    peatland_fires_gdf = gpd.sjoin(hotspots_gdf, buffered_peatlands_in_wgs84, how="inner", predicate='within')
    # --- END OF REFERENCE LOGIC ---

    print(f"Filtered down to {len(peatland_fires_gdf)} hotspots located on or within 1km of peatlands.")

    if peatland_fires_gdf.empty:
        print("No fires found on or near peatlands for this year.")
        return

    # --- Daily Clustering Loop ---
    all_clustered_data = []
    for date, daily_df in peatland_fires_gdf.groupby('acq_date'):
        if len(daily_df) < MIN_SAMPLES:
            continue

        coords = daily_df[['latitude', 'longitude']].to_numpy()
        rads = np.radians(coords)
        epsilon = DBSCAN_RADIUS_KM / 6371
        
        db = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, algorithm='ball_tree', metric='haversine').fit(rads)
        
        date_str = date.strftime('%Y%m%d')
        daily_df_copy = daily_df.copy()
        # Drop the extra 'index_right' column from the sjoin before concatenation
        daily_df_copy = daily_df_copy.drop(columns=['index_right'])
        daily_df_copy['cluster_id'] = [f"{date_str}_{label}" if label != -1 else 'noise' for label in db.labels_]
        
        all_clustered_data.append(daily_df_copy)

    if not all_clustered_data:
        print("No significant daily clusters found on or near peatlands.")
        return
        
    # --- Aggregation and Saving ---
    final_df = pd.concat(all_clustered_data)
    clusters_df = final_df[final_df['cluster_id'] != 'noise']
    
    if clusters_df.empty:
        print("No peatland-related clusters were formed that met the criteria across all days.")
        return

    cluster_summary = clusters_df.groupby('cluster_id').agg(
        hotspot_count=('latitude', 'size'),
        total_frp=('frp', 'sum'),
        centroid_lat=('latitude', 'mean'),
        centroid_lon=('longitude', 'mean'),
        date=('acq_date', 'first')
    ).reset_index()
    
    prioritized_clusters = cluster_summary.sort_values(by='hotspot_count', ascending=False)
    
    prioritized_clusters.to_csv(output_file, index=False)
    print(f"\nFound {len(prioritized_clusters)} distinct daily fire clusters on or near peatlands for the year.")
    print(f"Successfully saved the prioritized cluster list to '{output_file}'")
    if not prioritized_clusters.empty:
        print(f"Largest daily cluster had {prioritized_clusters.iloc[0]['hotspot_count']} hotspots.")

# --- 3. Batch Processing Main Block ---
if __name__ == "__main__":
    files_to_process = [
        'viirs-jpss1_2019_Indonesia.csv',
        'viirs-jpss1_2020_Indonesia.csv',
        'viirs-jpss1_2021_Indonesia.csv',
        'viirs-jpss1_2022_Indonesia.csv',
        'viirs-jpss1_2023_Indonesia.csv',
        'viirs-jpss1_2024_Indonesia.csv',
    ]

    output_dir = 'yearly_output_cluster_shapefile_filtered'
    os.makedirs(output_dir, exist_ok=True)

    for input_csv in files_to_process:
        print(f"\n{'='*20}\nProcessing file: {input_csv}\n{'='*20}")
        
        year = os.path.basename(input_csv).split('_')[1].split('.')[0]
        
        output_csv = os.path.join(output_dir, f'peatland_1km_buffer_clusters_{year}.csv')
        
        analyze_peatland_clusters_daily(input_csv, output_csv)
    
    print("\nBatch processing for peatland fires (with 1km buffer) complete.")
