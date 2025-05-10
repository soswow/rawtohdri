import os
import glob
import subprocess
import numpy as np
import Imath
import OpenEXR
from exposure_fusion import ExposureFusion
from logging_config import setup_loggers
from image_data import ImageData
from raw_metadata import get_exposure_info

# Set up loggers
loggers = setup_loggers()
logger = loggers['main']
exr_logger = loggers['exr']
raw_logger = loggers['raw']

def save_exr(img, output_path):
    """Save the image as an EXR file."""
    # Convert to float32
    img = img.astype(np.float32)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Prepare header
    header = OpenEXR.Header(width, height)
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    header['channels'] = dict([(c, Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))) 
                             for c in ['R', 'G', 'B']])
    
    # Create file
    out = OpenEXR.OutputFile(output_path, header)
    
    # Write pixel data
    R = img[:,:,0].tobytes()
    G = img[:,:,1].tobytes()
    B = img[:,:,2].tobytes()
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()

def process_raw_to_aces(input_file, verbose=False) -> str:
    """
    Process RAW file to ACES using RAWtoACES.
    
    Args:
        input_file: Path to input RAW file
        
    Returns:
        str: Path to the created ACES EXR file
    """
    # Check if EXR file already exists
    output_file: str = os.path.splitext(input_file)[0] + '_aces.exr'
    if os.path.exists(output_file):
        if verbose:
            raw_logger.info(f"ACES file already exists: {os.path.basename(output_file)}")
        return output_file
    
    cmd = [
        'rawtoaces',
        '--wb-method', '0',  # Use as-shot white balance
        '--mat-method', '0',  # Use camera metadata
        '-W',  # Write output file
        '--headroom', '6',  # Set headroom to 6 stops
        '-v',  # Verbose output
        input_file
    ]
    
    try:
        if verbose:
            raw_logger.info(f"Running RAWtoACES on {os.path.basename(input_file)}")
            raw_logger.info(f"Command: {' '.join(cmd)}")
        
        subprocess.run(cmd, check=True)
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"RAWtoACES did not create output file: {output_file}")
            
        if verbose:
            raw_logger.info(f"Successfully created {output_file}")
            
        return output_file
    except subprocess.CalledProcessError as e:
        raw_logger.error(f"Error running RAWtoACES: {e}")
        raise

def read_exr(exr_file: str, verbose: bool = False) -> np.ndarray:
    """
    Read an EXR file and return the image data.
    
    Args:
        exr_file: Path to EXR file
        verbose: Whether to enable verbose output
        
    Returns:
        numpy.ndarray: Image data in RGB format
    """
    img = OpenEXR.InputFile(exr_file)
    dw = img.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    if verbose:
        exr_logger.info(f"  Image size: {width}x{height}")
        exr_logger.info(f"  Data window: x={dw.min.x}->{dw.max.x}, y={dw.min.y}->{dw.max.y}")
    
    # Check the pixel type in the header
    channel_type = str(img.header()['channels']['R'].type)
    if verbose:
        exr_logger.info(f"  Channel type: {channel_type}")
    
    # Read the channels with appropriate dtype
    if channel_type == "HALF":
        dtype = np.float16
    else:  # FLOAT
        dtype = np.float32
    
    if verbose:
        exr_logger.info(f"  Using dtype: {dtype}")
    
    R_1d = np.frombuffer(img.channel('R'), dtype=dtype)
    G_1d = np.frombuffer(img.channel('G'), dtype=dtype)
    B_1d = np.frombuffer(img.channel('B'), dtype=dtype)
    
    # Calculate expected size
    expected_size = width * height
    actual_size = len(R_1d)
    
    if verbose:
        exr_logger.info(f"  Expected pixels: {expected_size}")
        exr_logger.info(f"  Actual pixels: {actual_size}")
    
    # Reshape the arrays
    try:
        R = R_1d.reshape(height, width)
        G = G_1d.reshape(height, width)
        B = B_1d.reshape(height, width)
    except ValueError as e:
        if verbose:
            exr_logger.info(f"  Failed to reshape to {height}x{width}")
            # Try to find factors of actual_size
            factors = []
            for i in range(1, int(np.sqrt(actual_size)) + 1):
                if actual_size % i == 0:
                    factors.append((i, actual_size // i))
            exr_logger.info(f"  Possible dimensions: {factors}")
        raise ValueError(f"Could not reshape image data. Expected {expected_size} pixels, got {actual_size}") from e
    
    # Combine into RGB array
    aces = np.stack([R, G, B], axis=2)
    
    if verbose:
        exr_logger.info(f"  Value range: [{aces.min():.3f}, {aces.max():.3f}]")
        exr_logger.info(f"  Mean value: {aces.mean():.3f}")
        exr_logger.info(f"  Final shape: {aces.shape}")
    
    return aces

def main(verbose=False):
    if verbose:
        logger.info("Starting RAW to HDRI process...")
    
    # Get all CR3 files
    input_files = sorted(glob.glob('input/*.CR3'))
    if not input_files:
        logger.error("No CR3 files found in input directory!")
        return
    
    if verbose:
        logger.info(f"Found {len(input_files)} CR3 files:")
        for f in input_files:
            logger.info(f"  {os.path.basename(f)}")
    
    # Process RAW files to ACES
    logger.info("Processing RAW files to ACES...")
    image_data_list: list[ImageData] = []
    
    for i, file_path in enumerate(input_files):
        if verbose:
            logger.info(f"\nProcessing {os.path.basename(file_path)} ({i+1}/{len(input_files)})")
        
        # Process RAW to ACES
        aces_file = process_raw_to_aces(file_path, verbose)
        
        # Read the ACES EXR file
        try:
            aces = read_exr(aces_file, verbose)
            
            # Get exposure information using the standalone function
            exp_info = get_exposure_info(file_path, verbose)
            
            # Create ImageData object
            image_data = ImageData(
                image=aces,
                raw_path=file_path,
                shutter_speed=exp_info['shutter_speed'],
                ev=exp_info['ev']
            )
            image_data_list.append(image_data)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    # Perform exposure fusion
    logger.info("\nPerforming exposure fusion...")
    fusion = ExposureFusion(verbose=verbose)
    fused = fusion.fuse(image_data_list)
    
    # Save result
    output_path = 'output/fused.exr'
    logger.info(f"Saving result to {output_path}")
    save_exr(fused, output_path)
    
    logger.info("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert RAW files to HDR EXR using exposure fusion')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    main(verbose=args.verbose)