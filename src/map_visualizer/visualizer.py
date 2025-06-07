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

    def __init__(self, default_font_size: int = 16, fir_marker_radius: int = 6, ai_marker_size: int = 25):
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
        
        # Attempt to load a preferred font, fallback to default
        font_paths_to_try = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", # Common on Linux
            "DejaVuSans-Bold.ttf", # If in current dir or system path
        ]
        self.font = ImageFont.load_default() # Start with default
        for path in font_paths_to_try:
            try:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, self.default_font_size)
                    _log_json("INFO", f"Successfully loaded font '{path}' with size {self.default_font_size}.")
                    break # Found a font, no need to try others
            except Exception as e:
                _log_json("WARNING", f"Could not load font at '{path}', trying next. Error: {e}")
        
        if self.font == ImageFont.load_default(): # Check if still default after trying
             _log_json("WARNING", "Using default PIL font. For better text, install a TTF font (e.g., DejaVuSans-Bold.ttf) "
                                "and ensure it's accessible.", tried_paths=font_paths_to_try)


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
        acquisition_date_str: str = "N/A" # Date of imagery acquisition
    ) -> Image.Image:
        """
        Generates a composite map image with fire detections and FIRMS hotspots.
        """
        _log_json("INFO", "Starting map generation process.",
                           acquisition_date=acquisition_date_str,
                           num_ai_detections_provided=len(ai_detections),
                           num_firms_hotspots_provided=len(firms_hotspots_df) if firms_hotspots_df is not None else 0)

        # --- THE FINAL FIX: Increase Pillow's image size limit ---
        # This prevents the DecompressionBombWarning and ensures the full image is loaded.
        Image.MAX_IMAGE_PIXELS = 2_000_000_000
        # --------------------------------------------------------

        try:
            base_image = Image.open(io.BytesIO(base_image_bytes)).convert("RGBA") # Use RGBA for transparency options
            draw = ImageDraw.Draw(base_image)
            img_width, img_height = base_image.size
        except IOError as ioe:
            _log_json("ERROR", "Failed to open base image from bytes for map generation.", error=str(ioe))
            raise # Re-raise to be handled by caller

        # --- Overlay FIRMS Hotspots ---
        if firms_hotspots_df is not None and not firms_hotspots_df.empty:
            _log_json("INFO", "Overlaying FIRMS hotspots onto map.", count=len(firms_hotspots_df))
            for _, row in firms_hotspots_df.iterrows():
                try:
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                except (ValueError, TypeError):
                    _log_json("WARNING", "Skipping FIRMS hotspot with invalid lat/lon.", data=row.to_dict())
                    continue

                confidence = str(row.get('confidence', 'unknown')).lower()
                pixel_coords = self._get_pixel_coords(img_width, img_height, image_bbox, lat, lon)

                if pixel_coords:
                    px, py = pixel_coords
                    radius = self.fir_marker_radius
                    if confidence == 'high': firms_color = (255, 0, 0, 200) # Red, slightly transparent
                    elif confidence == 'nominal': firms_color = (255, 165, 0, 200) # Orange
                    else: firms_color = (255, 255, 0, 180) # Yellow (for low or unknown)
                    
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                                 fill=firms_color, outline=(0, 0, 0, 220)) # Black outline
                else:
                    _log_json("WARNING", "Could not get pixel coordinates for FIRMS hotspot.", lat=lat, lon=lon)
        
        # --- Overlay AI Detections ---
        fire_detected_by_ai_in_this_image = False
        ai_confidence_for_this_image = 0.0

        if ai_detections:
            primary_ai_detection = ai_detections[0]
            if primary_ai_detection.get("detected", False):
                fire_detected_by_ai_in_this_image = True
                ai_confidence_for_this_image = primary_ai_detection.get("confidence", 0.0)

                center_lon = (image_bbox[0] + image_bbox[2]) / 2
                center_lat = (image_bbox[1] + image_bbox[3]) / 2
                
                pixel_coords_center = self._get_pixel_coords(img_width, img_height, image_bbox, center_lat, center_lon)

                if pixel_coords_center:
                    px_center, py_center = pixel_coords_center
                    marker_sz = self.ai_marker_size
                    ai_marker_color = (0, 255, 255, 220) # Cyan, slightly transparent
                    ai_text = f"AI: Fire ({ai_confidence_for_this_image:.2f})"
                    
                    draw.line((px_center - marker_sz, py_center, px_center + marker_sz, py_center), fill=ai_marker_color, width=3)
                    draw.line((px_center, py_center - marker_sz, px_center, py_center + marker_sz), fill=ai_marker_color, width=3)
                    
                    try:
                        text_bbox = draw.textbbox((0,0), ai_text, font=self.font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        text_width, text_height = draw.textsize(ai_text, font=self.font)

                    text_pos_x = px_center + marker_sz // 2 + 5
                    text_pos_y = py_center - text_height // 2
                    text_pos_x = min(text_pos_x, img_width - text_width - 5)
                    text_pos_y = max(5, min(text_pos_y, img_height - text_height - 5))

                    draw.text((text_pos_x, text_pos_y), ai_text, fill=ai_marker_color, font=self.font)
                    _log_json("INFO", "AI fire detection marker drawn on map.", confidence=ai_confidence_for_this_image)
            else:
                 _log_json("INFO", "AI processed this image: No fire detected.",
                           confidence=primary_ai_detection.get("confidence", 0.0))
        else:
            _log_json("INFO", "No AI detection results were provided for this image map.")

        # --- Add Timestamp and Legend ---
        text_color = (255, 255, 255, 255) # White, opaque
        shadow_color = (0, 0, 0, 200)    # Black, slightly transparent shadow

        date_text_content = f"Imagery Date: {acquisition_date_str}"
        date_text_pos_y = img_height - self.default_font_size - 10
        draw.text((11, date_text_pos_y + 1), date_text_content, font=self.font, fill=shadow_color)
        draw.text((10, date_text_pos_y), date_text_content, font=self.font, fill=text_color)

        legend_item_height = self.default_font_size + 4
        legend_y_start = img_height - 10
        legend_x_start = img_width - 200
        
        lgd_y = legend_y_start - legend_item_height
        draw.ellipse((legend_x_start, lgd_y - self.fir_marker_radius // 2, 
                      legend_x_start + self.fir_marker_radius, lgd_y + self.fir_marker_radius // 2),
                     fill=(255, 0, 0, 200), outline=(0,0,0))
        draw.text((legend_x_start + self.fir_marker_radius + 5, lgd_y - self.default_font_size // 2 -2),
                  "FIRMS (High)", fill=text_color, font=self.font)

        lgd_y -= legend_item_height
        draw.ellipse((legend_x_start, lgd_y - self.fir_marker_radius // 2, 
                      legend_x_start + self.fir_marker_radius, lgd_y + self.fir_marker_radius // 2),
                     fill=(255, 165, 0, 200), outline=(0,0,0))
        draw.text((legend_x_start + self.fir_marker_radius + 5, lgd_y - self.default_font_size // 2 -2),
                  "FIRMS (Nominal)", fill=text_color, font=self.font)

        if fire_detected_by_ai_in_this_image:
            lgd_y -= legend_item_height
            ai_lgd_marker_sz = self.ai_marker_size // 3
            ai_lgd_marker_center_x = legend_x_start + ai_lgd_marker_sz // 2
            ai_lgd_marker_center_y = lgd_y
            draw.line((ai_lgd_marker_center_x - ai_lgd_marker_sz, ai_lgd_marker_center_y,
                       ai_lgd_marker_center_x + ai_lgd_marker_sz, ai_lgd_marker_center_y),
                      fill=(0, 255, 255, 220), width=2)
            draw.line((ai_lgd_marker_center_x, ai_lgd_marker_center_y - ai_lgd_marker_sz,
                       ai_lgd_marker_center_x, ai_lgd_marker_center_y + ai_lgd_marker_sz),
                      fill=(0, 255, 255, 220), width=2)
            draw.text((legend_x_start + self.fir_marker_radius + 5, lgd_y - self.default_font_size // 2 -2),
                      "AI Detection", fill=(0, 255, 255, 220), font=self.font)


        _log_json("INFO", "Map generation completed successfully.")
        return base_image


# --- Example Usage (for local testing) ---
if __name__ == "__main__":
    _log_json("INFO", "Running local test for MapVisualizer.")

    dummy_image_width, dummy_image_height = 800, 600
    dummy_base_pil_image = Image.new('RGB', (dummy_image_width, dummy_image_height), color = (73, 109, 137))
    buffer = io.BytesIO()
    dummy_base_pil_image.save(buffer, format="PNG")
    dummy_image_bytes = buffer.getvalue()

    dummy_image_bbox = [-122.0, 36.0, -118.0, 38.0]

    dummy_firms_data = pd.DataFrame({
        'latitude': [37.5, 37.2, 36.8, 36.1],
        'longitude': [-121.5, -119.5, -118.5, -121.8],
        'confidence': ['high', 'nominal', 'low', 'high'],
    })
    
    ai_fire_detected = [{"detected": True, "confidence": 0.91, "details": "Fire confirmed by AI"}]
    ai_no_fire_detected = [{"detected": False, "confidence": 0.12, "details": "No fire confirmed by AI"}]

    visualizer = MapVisualizer(default_font_size=18, fir_marker_radius=7, ai_marker_size=30)

    try:
        _log_json("INFO", "Generating map with AI fire detection and FIRMS hotspots...")
        map_with_fire = visualizer.generate_fire_map(
            base_image_bytes=dummy_image_bytes,
            image_bbox=dummy_image_bbox,
            ai_detections=ai_fire_detected,
            firms_hotspots_df=dummy_firms_data,
            acquisition_date_str="2024-01-15"
        )
        map_with_fire.save("test_map_fire_detected_refined.png")
        _log_json("INFO", "Map with fire saved as test_map_fire_detected_refined.png")

        _log_json("INFO", "Generating map with NO AI fire detection and FIRMS hotspots...")
        map_no_fire = visualizer.generate_fire_map(
            base_image_bytes=dummy_image_bytes,
            image_bbox=dummy_image_bbox,
            ai_detections=ai_no_fire_detected,
            firms_hotspots_df=dummy_firms_data,
            acquisition_date_str="2024-01-16"
        )
        map_no_fire.save("test_map_no_fire_detected_refined.png")
        _log_json("INFO", "Map with no AI fire saved as test_map_no_fire_detected_refined.png")

        _log_json("INFO", "Generating map with NO FIRMS data (AI fire detected)...")
        map_no_firms = visualizer.generate_fire_map(
            base_image_bytes=dummy_image_bytes,
            image_bbox=dummy_image_bbox,
            ai_detections=ai_fire_detected,
            firms_hotspots_df=None,
            acquisition_date_str="2024-01-17"
        )
        map_no_firms.save("test_map_no_firms_refined.png")
        _log_json("INFO", "Map with no FIRMS saved as test_map_no_firms_refined.png")

    except Exception as e:
        _log_json("CRITICAL", f"Local map visualizer test failed: {e}", error_type=type(e).__name__)
        import traceback
        traceback.print_exc()
