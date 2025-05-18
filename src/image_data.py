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
        ev: Exposure value (EV + EV offset)
        ev_offset: Additional EV offset from folder name (e.g. +2EV, -1EV)
        ev_original: Original EV value from the image metadata
        capture_time: Time when the image was captured
        aperture: Aperture value (f-number)
        iso: ISO sensitivity
    """
    image: Optional[np.ndarray]
    raw_path: str
    filename: str
    shutter_speed: float
    shutter_speed_str: str
    ev: float
    ev_offset: float = 0.0
    ev_original: float = 0.0
    capture_time: datetime | None = None
    aperture: float = 0.0
    iso: int = 0
    
    def __str__(self) -> str:
        """Return a string representation of the image data."""
        return (f"ImageData(filename='{os.path.basename(self.raw_path)}', "
                f"filename='{self.filename}', "
                f"shape={self.image.shape if self.image is not None else None}, "
                f"shutter_speed={self.shutter_speed:.3f}s, "
                f"shutter_speed_str='{self.shutter_speed_str}', "
                f"ev={self.ev:.1f}, "
                f"ev_offset={self.ev_offset:+.1f}, "
                f"aperture=f/{self.aperture:.1f}, "
                f"iso={self.iso}, "
                f"capture_time={self.capture_time}")
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the image data."""
        return (f"ImageData(\n"
                f"    image={'array(shape=' + str(self.image.shape) + ', dtype=' + str(self.image.dtype) + ')' if self.image is not None else 'None'},\n"
                f"    raw_path='{self.raw_path}',\n"
                f"    filename='{self.filename}',\n"
                f"    shutter_speed={self.shutter_speed:.3f},\n"
                f"    shutter_speed_str='{self.shutter_speed_str}',\n"
                f"    ev={self.ev:.1f},\n"
                f"    ev_offset={self.ev_offset:+.1f},\n"
                f"    aperture={self.aperture:.1f},\n"
                f"    iso={self.iso},\n"
                f"    capture_time={self.capture_time}\n"
                f")") 