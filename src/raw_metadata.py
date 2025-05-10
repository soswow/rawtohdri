import numpy as np
import subprocess
import json
from logging_config import setup_loggers

# Set up loggers
loggers = setup_loggers()
metadata_logger = loggers['metadata']

def get_exposure_info(raw_path: str, verbose: bool = False) -> dict:
    """
    Get exposure information from a RAW file using exiftool.
    Since aperture and ISO are constant across all images, we only need shutter speed
    to determine relative exposure differences.
    
    Args:
        raw_path: Path to RAW file
        verbose: Whether to enable detailed logging
        
    Returns:
        dict: Dictionary containing exposure information with keys:
            - shutter_speed: Shutter speed in seconds
            - ev: Relative exposure value (based on shutter speed only)
            - capture_time: Time when the image was captured (datetime object)
    """
    try:
        # Run exiftool to get exposure info and capture time
        result = subprocess.run(
            ['exiftool', '-j', '-ShutterSpeed', '-DateTimeOriginal', raw_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"exiftool failed: {result.stderr}")
        
        # Parse JSON output
        data = json.loads(result.stdout)[0]
        
        # Extract and parse shutter speed
        shutter_value = data.get('ShutterSpeed', '1')
        
        # Handle different shutter speed formats
        if isinstance(shutter_value, (int, float)):
            # Already a number
            shutter_speed = float(shutter_value)
        elif isinstance(shutter_value, str):
            if '/' in shutter_value:
                # Handle fraction format (e.g., '1/8000')
                num, denom = map(float, shutter_value.split('/'))
                shutter_speed = num / denom
            else:
                # Handle decimal format (e.g., '30')
                shutter_speed = float(shutter_value)
        else:
            raise ValueError(f"Unexpected shutter speed format: {shutter_value}")
        
        # Calculate relative EV based on shutter speed only
        # This gives us the relative exposure difference between images
        ev = np.log2(1/shutter_speed)
        
        # Extract and parse capture time
        from datetime import datetime
        capture_time_str = data.get('DateTimeOriginal', '')
        if capture_time_str:
            capture_time = datetime.strptime(capture_time_str, '%Y:%m:%d %H:%M:%S')
        else:
            capture_time = None
        
        if verbose:
            metadata_logger.info(f"Exposure info for {raw_path}:")
            metadata_logger.info(f"  Shutter speed: {shutter_speed}")
            metadata_logger.info(f"  Relative EV: {ev:.1f}")
            if capture_time:
                metadata_logger.info(f"  Capture time: {capture_time}")
        
        return {
            'shutter_speed': shutter_speed,
            'ev': ev,
            'capture_time': capture_time
        }
        
    except Exception as e:
        if verbose:
            metadata_logger.error(f"Error getting exposure info: {e}")
        raise 