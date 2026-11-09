import numpy as np
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class Point3D:
    """
    Represents a single 3D point measuremnt
    """
    x: float
    y: float
    z: float
    x_err: float
    y_err: float
    z_err: float
    
    # ============ Serialization Methods ============
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Point3D':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Point3D':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, x_err, y_err, z_err]"""
        return np.array([self.x, self.y, self.z, self.x_err, self.y_err, self.z_err])
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Point3D':
        """Create from numpy array"""
        return cls(*arr)
    
    def to_bytes(self) -> bytes:
        """Convert to binary format (numpy binary)"""
        return self.to_numpy().tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Point3D':
        """Create from binary data"""
        arr = np.frombuffer(data, dtype=np.float64, count=6)
        return cls.from_numpy(arr)
        
if __name__ == "__main__":
    p = Point3D(1,2,3,4,5,6)
    print(p)