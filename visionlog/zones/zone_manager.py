"""
zones/zone_manager.py
Defines named zones as polygons per camera and checks which zone a bbox is in.
"""
import numpy as np
from typing import Optional
from visionlog import config


class ZoneManager:
    """
    Converts frame-relative polygon definitions (in config) into pixel coords
    and checks which zone a detection's bounding box center falls in.
    """

    def __init__(self, camera_id: str, frame_w: int, frame_h: int):
        self.camera_id = camera_id
        self.frame_w   = frame_w
        self.frame_h   = frame_h

        raw_zones = config.ZONES.get(camera_id, {})
        # Convert % -> pixel polygons
        self.zones: dict[str, np.ndarray] = {}
        for name, pts in raw_zones.items():
            px_pts = [(x * frame_w, y * frame_h) for x, y in pts]
            self.zones[name] = np.array(px_pts, dtype=np.float32)

        print(f"[ZoneManager] {camera_id}: {len(self.zones)} zones: {list(self.zones)}")

    def get_zone(self, bbox: tuple) -> str:
        """
        Given a bbox (x1,y1,x2,y2), find which zone the center point is in.
        Returns zone name or 'UNKNOWN'.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        pt = (float(cx), float(cy))

        for name, poly in self.zones.items():
            if self._point_in_polygon(pt, poly):
                return name
        return "UNKNOWN"

    @staticmethod
    def _point_in_polygon(pt: tuple, poly: np.ndarray) -> bool:
        """Ray-casting algorithm for point-in-polygon."""
        x, y = pt
        n = len(poly)
        inside = False
        px, py = poly[-1]
        for i in range(n):
            nx, ny = poly[i]
            if ((ny > y) != (py > y)) and (x < (px - nx) * (y - ny) / (py - ny + 1e-9) + nx):
                inside = not inside
            px, py = nx, ny
        return inside
