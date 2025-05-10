import numpy as np
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ImageData:
    """
    A class to store image data and exposure information.
    
    Attributes:
        image: The image data as a numpy array
        raw_path: Path to the original RAW file
        shutter_speed: Shutter speed in seconds
        ev: Exposure value
        capture_time: Time when the image was captured
    """
    image: np.ndarray
    raw_path: str
    shutter_speed: float
    ev: float
    capture_time: datetime | None = None
    
    def __str__(self) -> str:
        """Return a string representation of the image data."""
        return (f"ImageData(filename='{os.path.basename(self.raw_path)}', "
                f"shape={self.image.shape}, "
                f"shutter_speed={self.shutter_speed:.3f}s, "
                f"ev={self.ev:.1f}, "
                f"capture_time={self.capture_time}")
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the image data."""
        return (f"ImageData(\n"
                f"    image=array(shape={self.image.shape}, dtype={self.image.dtype}),\n"
                f"    raw_path='{self.raw_path}',\n"
                f"    shutter_speed={self.shutter_speed:.3f},\n"
                f"    ev={self.ev:.1f},\n"
                f"    capture_time={self.capture_time}\n"
                f")") 