import numpy as np
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ImageData:
    """
    A class to store image data and exposure information.
    
    Attributes:
        image: The image data as a numpy array (or None if only metadata is needed)
        raw_path: Path to the original RAW file
        shutter_speed: Shutter speed in seconds
        ev: Exposure value
        capture_time: Time when the image was captured
    """
    image: Optional[np.ndarray]
    raw_path: str
    shutter_speed: float
    ev: float
    capture_time: datetime | None = None
    
    def __str__(self) -> str:
        """Return a string representation of the image data."""
        return (f"ImageData(filename='{os.path.basename(self.raw_path)}', "
                f"shape={self.image.shape if self.image is not None else None}, "
                f"shutter_speed={self.shutter_speed:.3f}s, "
                f"ev={self.ev:.1f}, "
                f"capture_time={self.capture_time}")
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the image data."""
        return (f"ImageData(\n"
                f"    image={'array(shape=' + str(self.image.shape) + ', dtype=' + str(self.image.dtype) + ')' if self.image is not None else 'None'},\n"
                f"    raw_path='{self.raw_path}',\n"
                f"    shutter_speed={self.shutter_speed:.3f},\n"
                f"    ev={self.ev:.1f},\n"
                f"    capture_time={self.capture_time}\n"
                f")") 