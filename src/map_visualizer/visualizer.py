# src/map_visualizer/visualizer.py

import logging
import json # For _log_json
import io # For BytesIO
import os # For checking font file existence
from PIL import Image, ImageDraw, ImageFont # Correct Pillow imports
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd # Added import for type hinting firms_hotspots_df

# --- Logging Setup ---
logger = logging.getLogger(__name__) # Standard

def _log_json(severity: str, message: str, **kwargs):
    """
    Helper to log structured JSON messages to stdout for GCP Cloud Logging.
    """
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "MapVisualizer",
        **kwargs
    }
    print(json.dumps(log_entry))


class MapVisualizer:
    """
    Component 4: Creates a user-friendly map image showing fire detections
    overlaid on a satellite image.
    """

    def __init__(self, default_font_size: int = 20, fir_marker_radius: int = 8, ai_marker_size: int = 35):
        """
        Initializes the MapVisualizer with configurable drawing parameters.

        Args:
            default_font_size (int): Default font size for text overlays.
            fir_marker_radius (int): Radius for FIRMS hotspot markers.
            ai_marker_size (int): Size for AI detection markers.
        """
        _log_json("INFO", "MapVisualizer initialized.")
        self.default_font_size = default_font_size
        self.fir_marker_radius = fir_marker_radius
        self.ai_marker_size = ai_marker_size
        
        # Try to use a better font with anti-aliasing
        self.font = None
        font_paths_to_try = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        for path in font_paths_to_try:
            try:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, self.default_font_size)
                    _log_json("INFO", f"Successfully loaded font '{path}' with size {self.default_font_size}.")
                    break
            except Exception as e:
                _log_json("WARNING", f"Could not load font at '{path}': {e}")
        
        # If no TrueType font found, create a larger default font
        if self.font is None:
            try:
                # Try to load default font with larger size
                self.font = ImageFont.load_default()
                _log_json("WARNING", "Using default PIL font. Text quality may be poor.")
            except:
                self.font = ImageFont.load_default()
                _log_json("WARNING", "Using basic default font.")


    def _get_pixel_coords(self, img_width: int, img_height: int, image_geospatial_bbox: List[float],
                          point_latitude: float, point_longitude: float) -> Optional[Tuple[int, int]]:
        """
        Converts a geographic coordinate (lat, lon) to pixel coordinates (x, y)
        within an image given its geographic bounding box.
        """
        min_lon, min_lat, max_lon, max_lat = image_geospatial_bbox

        if max_lon == min_lon or max_lat == min_lat:
            _log_json("ERROR", "Invalid image_geospatial_bbox: longitude or latitude range is zero.",
                      bbox=image_geospatial_bbox)
            return None

        norm_lon = (point_longitude - min_lon) / (max_lon - min_lon)
        norm_lat = (point_latitude - min_lat) / (max_lat - min_lat)

        pixel_x = int(norm_lon * img_width)
        pixel_y = int((1.0 - norm_lat) * img_height)

        pixel_x = max(0, min(pixel_x, img_width - 1))
        pixel_y = max(0, min(pixel_y, img_height - 1))

        return pixel_x, pixel_y

    def generate_fire_map(
        self,
        base_image_bytes: bytes,
        image_bbox: List[float], # Geographic BBOX of the base_image_bytes
        ai_detections: List[Dict[str, Any]], # List of AI detection dicts for this image/region
        firms_hotspots_df: Optional[pd.DataFrame] = None,
        acquisition_date_str: str = "N/A
