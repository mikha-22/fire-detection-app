# src/cloud_functions/result_processor/test_fully_offline_result_processor.py

import os
import sys
import json
import logging
from collections import defaultdict
from datetime import datetime

# --- Add project root to path for imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

import folium
from folium import plugins
import geopandas as gpd

# --- Build absolute paths to data files ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INCIDENTS_FILE = os.path.join(SCRIPT_DIR, "incidents.jsonl")
PREDICTIONS_FILE = os.path.join(SCRIPT_DIR, "mock_prediction_results.jsonl")
OUTPUT_REPORT_HTML = os.path.join(SCRIPT_DIR, "report.html")
PEATLAND_SHAPEFILE = os.path.join(PROJECT_ROOT, "src", "geodata", "Indonesia_peat_lands.shp")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dashboard_html(total_incidents, total_hotspots, risk_counts):
    """Creates the HTML for the floating header and stats panel."""
    run_date = datetime.now().strftime('%B %d, %Y')
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        body {{ font-family: 'Roboto', sans-serif; }}
        .dashboard-header {{ position: fixed; top: 10px; left: 50px; right: 10px; height: 60px; background-color: rgba(255, 255, 255, 0.9); border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 1000; display: flex; align-items: center; padding: 0 20px; font-size: 18px; }}
        .dashboard-stats {{ position: fixed; top: 80px; right: 10px; width: 250px; background-color: rgba(255, 255, 255, 0.9); border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 1000; padding: 20px; }}
        .stat-item {{ margin-bottom: 15px; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .stat-value {{ font-size: 22px; font-weight: bold; color: #111; }}
        .risk-distro span {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px; color: white;}}
    </style>
    <div class="dashboard-header"><strong>🔥 Indonesia Fire Tracker</strong> | {run_date}</div>
    <div class="dashboard-stats">
        <div class="stat-item"><div class="stat-label">Active Fire Clusters</div><div class="stat-value">{total_incidents}</div></div>
        <div class="stat-item"><div class="stat-label">Total Hotspots Detected</div><div class="stat-value">{total_hotspots}</div></div>
        <div class="stat-item"><div class="stat-label">Risk Distribution</div><div class="risk-distro">
            <span style="background-color:red;">High: {risk_counts['High']}</span>
            <span style="background-color:orange;">Mod: {risk_counts['Moderate']}</span>
            <span style="background-color:green;">Low: {risk_counts['Low']}</span>
        </div></div><hr>
        <p style="font-size:12px; color:#555;">This is an automated report from satellite data. For emergency services, contact local authorities.</p>
    </div>
    """
    return html


def run_offline_dashboard_test():
    """Generates a full dashboard-style HTML report using local mock files."""
    logger.info("--- Starting dashboard generation test ---")

    # 1. Load Data
    try:
        with open(PREDICTIONS_FILE, 'r') as f: predictions = [json.loads(line) for line in f.read().strip().split('\n')]
        with open(INCIDENTS_FILE, 'r') as f:
            content = f.read().strip()
            incidents_data = [json.loads(content)] if content.startswith('{') and content.endswith('}') else [json.loads(line) for line in content.split('\n')]
        incidents_by_cluster = {inc['cluster_id']: inc for inc in incidents_data}
        predictions_by_cluster = {pred['instance_id']: pred for pred in predictions}
        logger.info(f"Loaded {len(incidents_data)} incidents and {len(predictions)} predictions.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}"); return

    # Load Peatland Shapefile
    try:
        peatlands_gdf = gpd.read_file(PEATLAND_SHAPEFILE)
        logger.info("Successfully loaded peatland shapefile.")
    except Exception as e:
        logger.warning(f"Could not load peatland shapefile: {e}. Proceeding without it.")
        peatlands_gdf = None
    
    # 2. Initialize the Map
    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles='CartoDB positron')

    # 3. Add the Peatland Layer to the map
    if peatlands_gdf is not None:
        # --- FIX: Changed colors to a greenish theme ---
        style_function = lambda x: {
            'fillColor': '#6B8E23',  # OliveDrab
            'color': '#556B2F',      # DarkOliveGreen
            'weight': 1,
            'fillOpacity': 0.4
        }
        folium.GeoJson(
            peatlands_gdf, name='Peatland Areas', style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['layer_revi'], aliases=['Type:']),
            show=True
        ).add_to(m)
        
    # 4. Create a MarkerCluster group and populate it
    marker_cluster = plugins.MarkerCluster(name='Fire Clusters').add_to(m)
    total_hotspots, risk_counts = 0, {'High': 0, 'Moderate': 0, 'Low': 0}

    for cluster_id, incident in incidents_by_cluster.items():
        prediction = predictions_by_cluster.get(cluster_id)
        if not prediction: continue

        total_hotspots += incident.get('point_count', 0)
        score = prediction.get('confidence_score', 0.0)
        
        if score > 0.75: risk_level, marker_color, risk_counts['High'] = "High", "red", risk_counts['High'] + 1
        elif score > 0.5: risk_level, marker_color, risk_counts['Moderate'] = "Moderate", "orange", risk_counts['Moderate'] + 1
        else: risk_level, marker_color, risk_counts['Low'] = "Low", "green", risk_counts['Low'] + 1
        
        popup_html = f"""<div style="width: 250px; font-family: sans-serif;"><h4 style="margin-bottom:0; color:{marker_color};">{risk_level.upper()} RISK FIRE</h4><p style="font-size:12px; color:#555;">near {incident.get('centroid_latitude'):.2f}, {incident.get('centroid_longitude'):.2f}</p><hr style="margin: 8px 0;"><p style="font-size:13px;">This fire consists of <b>{incident.get('point_count', 'N/A')} hotspots</b>. The system confidence for this event is <b>{score:.0%}</b>.</p><a href="https://www.google.com/maps/search/?api=1&query={incident['centroid_latitude']},{incident['centroid_longitude']}" target="_blank">View on Google Maps</a></div>"""
        iframe = folium.IFrame(popup_html, width=280, height=180)
        popup = folium.Popup(iframe, max_width=280)

        folium.Marker(
            location=[incident['centroid_latitude'], incident['centroid_longitude']],
            popup=popup, tooltip=f"{cluster_id} - {risk_level} Risk", icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
        ).add_to(marker_cluster)

    # 5. Add Dashboard UI and Layer Control
    dashboard_html = create_dashboard_html(len(incidents_data), total_hotspots, risk_counts)
    m.get_root().html.add_child(folium.Element(dashboard_html))
    folium.LayerControl().add_to(m)

    # 6. Save the final dashboard file
    try:
        m.save(OUTPUT_REPORT_HTML)
        logger.info(f"✅ Success! Dashboard saved to: {os.path.abspath(OUTPUT_REPORT_HTML)}")
    except Exception as e:
        logger.error(f"Failed to save the HTML dashboard: {e}")

if __name__ == "__main__":
    run_offline_dashboard_test()
