import numpy as np
from scipy.spatial.transform import Rotation

from sts_ladder_metrology.point3d import Point3D

class SensorShape:
    """
    Rectangular shape in 3D, defined by a center and orientation.
    """
    def __init__(self, size: tuple[float, float],
                 center: tuple[float,float,float] | None = None,
                 angles: tuple[float,float,float] | None = None):
        
        if size[0] < 1e-5 or size[1] < 1e-5:
            raise ValueError("Dimensions cannot be 0")
        
        self.size = size
        self.center = np.zeros(3) if center is None else np.array(center, dtype=float)
        self.angles = np.zeros(3) if angles is None else np.array(angles, dtype=float)

    @property
    def R(self) -> np.ndarray:
        """
        Return the 3x3 rotation matrix corresponding to the current Euler angles
        """
        return Rotation.from_euler('xyz', self.angles).as_matrix()

    def distance_to_edge(self, point: tuple[float, float, float] | Point3D) -> float:
        """
        Compute the true Euclidean distance from a 3D point to the nearest edge
        of this rectangle (lying in a plane defined by self.center, self.R, self.size).
        """
        
        if isinstance(point, Point3D):
            point = point.to_numpy()[:2]
        else:    
            point = np.array(point, dtype=float)
        
        local = (point - self.center) @ self.R  # world → local
        px, py, pz = local
        hx, hy = self.size[0] / 2.0, self.size[1] / 2.0

        # distance along normal (off-plane)
        dz = abs(pz)

        # clamp px, py to rectangle bounds for corner/edge distance
        cx = np.clip(px, -hx, hx)
        cy = np.clip(py, -hy, hy)

        # closest point *on rectangle boundary* (edges)
        # push clamped point to edge if inside rectangle
        if -hx < px < hx and -hy < py < hy:
            # inside — move to nearest edge
            dx_to_edges = [hx - px, px + hx, hy - py, py + hy]
            nearest = np.argmin(dx_to_edges)
            if nearest == 0:
                cx = hx
            elif nearest == 1:
                cx = -hx
            elif nearest == 2:
                cy = hy
            elif nearest == 3:
                cy = -hy

        # vector to nearest point on edge in local plane
        dx, dy = px - cx, py - cy

        # combine in-plane distance and off-plane distance
        return np.sqrt(dx * dx + dy * dy + dz * dz)
        
 