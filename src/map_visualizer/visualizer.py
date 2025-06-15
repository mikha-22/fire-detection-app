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
                self.font = ImageFont.load_default(size=self.default_font_size)
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

        # Clamp the coordinates to the bounding box to prevent errors
        point_longitude = max(min_lon, min(point_longitude, max_lon))
        point_latitude = max(min_lat, min(point_latitude, max_lat))

        norm_lon = (point_longitude - min_lon) / (max_lon - min_lon)
        norm_lat = (point_latitude - min_lat) / (max_lat - min_lat)

        pixel_x = int(norm_lon * img_width)
        pixel_y = int((1.0 - norm_lat) * img_height) # Y is inverted in PIL

        return pixel_x, pixel_y

    def generate_fire_map(
        self,
        base_image_bytes: bytes,
        image_bbox: List[float],
        ai_detections: List[Dict[str, Any]],
        firms_hotspots_df: Optional[pd.DataFrame] = None,
        acquisition_date_str: str = "N/A",
    ) -> Image.Image:
        """
        Generates a map by overlaying fire data onto a base satellite image.

        Args:
            base_image_bytes: The raw bytes of the satellite image (e.g., PNG/GeoTIFF).
            image_bbox: The geographic bounding box [min_lon, min_lat, max_lon, max_lat] of the image.
            ai_detections: A list of prediction dictionaries from the AI model.
            firms_hotspots_df: A DataFrame of FIRMS hotspots with 'latitude' and 'longitude'.
            acquisition_date_str: The acquisition date of the imagery.

        Returns:
            A PIL Image object with all overlays drawn.
        """
        _log_json("INFO", "Starting map generation process.")

        # 1. Load base image
        try:
            img = Image.open(io.BytesIO(base_image_bytes)).convert("RGBA")
            img_width, img_height = img.size
        except Exception as e:
            _log_json("ERROR", f"Failed to open base image bytes: {e}", exc_info=True)
            img = Image.new("RGBA", (800, 600), "grey")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Error: Could not load base image.", fill="red", font=self.font)
            return img

        draw = ImageDraw.Draw(img, "RGBA")

        # 2. Draw FIRMS hotspots
        if firms_hotspots_df is not None and not firms_hotspots_df.empty:
            _log_json("INFO", f"Drawing {len(firms_hotspots_df)} FIRMS hotspots.")
            for _, row in firms_hotspots_df.iterrows():
                coords = self._get_pixel_coords(img_width, img_height, image_bbox, row['latitude'], row['longitude'])
                if coords:
                    x, y = coords
                    radius = self.fir_marker_radius
                    draw.ellipse(
                        [(x - radius, y - radius), (x + radius, y + radius)],
                        fill=(255, 215, 0, 180),  # Gold with transparency
                        outline=(255, 255, 255, 200) # Light outline
                    )

        # 3. Create a semi-transparent banner at the top for text
        banner_height = 100
        banner_color = (0, 0, 0, 170)  # Black, ~67% opacity
        draw.rectangle([(0, 0), (img_width, banner_height)], fill=banner_color)

        # 4. Draw text on the banner
        y_pos = 10
        margin = 15
        line_height = self.default_font_size + 8

        title_text = "Wildfire Analysis Report"
        draw.text((margin, y_pos), title_text, font=self.font, fill="white")
        y_pos += line_height

        date_text = f"Imagery Date: {acquisition_date_str}"
        draw.text((margin, y_pos), date_text, font=self.font, fill="white")
        y_pos += line_height

        firms_count = len(firms_hotspots_df) if firms_hotspots_df is not None else 0
        firms_text = f"FIRMS Hotspots in View: {firms_count}"
        draw.text((margin, y_pos), firms_text, font=self.font, fill="white")

        # 5. Display the primary AI Detection Result on the right side of the banner
        if ai_detections:
            detection = ai_detections[0]
            detected = detection.get("detected", False)
            confidence = detection.get("confidence", 0.0)

            status_text = "FIRE DETECTED" if detected else "No Fire Detected"
            status_color = (255, 80, 80, 255) if detected else (80, 255, 80, 255)

            try:
                status_font = ImageFont.truetype(self.font.path, self.default_font_size + 10)
            except Exception:
                status_font = self.font

            status_width = draw.textlength(status_text, font=status_font)
            draw.text((img_width - status_width - margin, 15), status_text, font=status_font, fill=status_color)

            conf_text = f"Confidence: {confidence:.1%}"
            conf_width = draw.textlength(conf_text, font=self.font)
            draw.text((img_width - conf_width - margin, 15 + line_height + 5), conf_text, font=self.font, fill="white")

        _log_json("INFO", "Map generation complete.")
        return img
