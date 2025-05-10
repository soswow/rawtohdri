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
    """
    try:
        # Run exiftool to get exposure info
        result = subprocess.run(
            ['exiftool', '-j', '-ShutterSpeed', raw_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"exiftool failed: {result.stderr}")
        
        # Parse JSON output
        data = json.loads(result.stdout)[0]
        
        # Extract and parse shutter speed
        shutter_str = data.get('ShutterSpeed', '1')
        if '/' in shutter_str:
            # Handle fraction format (e.g., '1/8000')
            num, denom = map(float, shutter_str.split('/'))
            shutter_speed = num / denom
        else:
            # Handle decimal format (e.g., '30')
            shutter_speed = float(shutter_str)
        
        # Calculate relative EV based on shutter speed only
        # This gives us the relative exposure difference between images
        ev = np.log2(1/shutter_speed)
        
        if verbose:
            metadata_logger.info(f"Exposure info for {raw_path}:")
            metadata_logger.info(f"  Shutter speed: {shutter_speed}")
            metadata_logger.info(f"  Relative EV: {ev:.1f}")
        
        return {
            'shutter_speed': shutter_speed,
            'ev': ev
        }
        
    except Exception as e:
        metadata_logger.error(f"Error getting exposure info for {raw_path}: {str(e)}")
        raise 