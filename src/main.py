import os
import glob
import subprocess
import numpy as np
import Imath
import OpenEXR
import re
from datetime import datetime, timedelta
from exposure_fusion import ExposureFusion
from logging_config import setup_loggers
from image_data import ImageData
from raw_metadata import get_exposure_info
from exr_utils import save_exr
import shutil
import sys

# Set up loggers
loggers = setup_loggers()
logger = loggers['main']
exr_logger = loggers['exr']
raw_logger = loggers['raw']

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
        '--headroom', '2',  # Set headroom to 2
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

def group_images_by_time(image_data_list: list[ImageData], time_delta: timedelta = timedelta(seconds=1)) -> list[list[ImageData]]:
    """
    Group images into stacks based on capture time.
    
    Args:
        image_data_list: List of ImageData objects
        time_delta: Maximum time difference between images in the same stack
        
    Returns:
        list[list[ImageData]]: List of image stacks
    """
    # Filter out images without capture time and sort by capture time
    valid_images = [img for img in image_data_list if img.capture_time is not None]
    if not valid_images:
        return []
    
    # Sort by capture time (we know all images have capture_time now)
    sorted_images = sorted(valid_images, key=lambda x: x.capture_time)  # type: ignore
    
    stacks = []
    current_stack = [sorted_images[0]]
    
    for img in sorted_images[1:]:
        if img.capture_time - current_stack[-1].capture_time <= time_delta:
            current_stack.append(img)
        else:
            if len(current_stack) > 1:  # Only keep stacks with multiple images
                stacks.append(current_stack)
            current_stack = [img]
    
    if len(current_stack) > 1:  # Don't forget the last stack
        stacks.append(current_stack)
    
    return stacks

def is_ev_offset_folder(folder_name: str) -> bool:
    """Check if a folder name matches the <int>EV pattern."""
    return bool(re.match(r'^([+-]?\d+)EV$', folder_name))

def get_ev_offset(folder_name: str) -> float:
    """Get EV offset from folder name if it matches <int>EV pattern."""
    match = re.match(r'^([+-]?\d+)EV$', folder_name)
    if match:
        return float(match.group(1))
    return 0.0

def find_stack_folders(input_dir: str) -> list[str]:
    """Return a list of subfolders in input_dir, each representing a stack."""
    return [os.path.join(input_dir, d) for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))]

def find_ev_offset_folders(stack_folder: str) -> list[str]:
    """
    Find all EV offset folders inside a stack folder.
    Returns a list of EV offset folder paths.
    """
    ev_folders = []
    for d in os.listdir(stack_folder):
        if is_ev_offset_folder(d):
            ev_folders.append(os.path.join(stack_folder, d))
    return ev_folders

def find_loose_images(input_dir: str) -> list[str]:
    """Return a list of image files in input_dir that are not in any subfolder."""
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.cr3')]

def move_images_to_stack_folders(input_dir: str, time_delta: int, verbose: bool = False):
    """
    Find loose images, group them into stacks, and move each stack into a new folder named
    after the first and last image in the stack.
    """
    loose_images = find_loose_images(input_dir)
    if not loose_images:
        return
    
    # Get ImageData for loose images (only need capture_time and path)
    image_data_list = []
    for file_path in sorted(loose_images):
        exp_info = get_exposure_info(file_path, verbose)
        image_data_list.append(ImageData(
            image=None,  # Not needed for stacking
            raw_path=file_path,
            filename=os.path.basename(file_path),
            shutter_speed=exp_info['shutter_speed'],
            shutter_speed_str=exp_info['shutter_speed_str'],
            ev=exp_info['ev'],
            capture_time=exp_info['capture_time']
        ))
    
    # Group by time
    from datetime import timedelta
    stacks = group_images_by_time(image_data_list, timedelta(seconds=time_delta))
    
    for stack in stacks:
        first_name = os.path.splitext(os.path.basename(stack[0].raw_path))[0]
        last_name = os.path.splitext(os.path.basename(stack[-1].raw_path))[0]
        folder_name = f"{first_name}-{last_name}"
        folder_path = os.path.join(input_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for img in stack:
            dest = os.path.join(folder_path, os.path.basename(img.raw_path))
            if verbose:
                logger.info(f"Moving {img.raw_path} -> {dest}")
            shutil.move(img.raw_path, dest)

def main(verbose=False, time_delta=1, input_dir='input', output_dir='output', organize_only=False, debug_pixel=None):
    if verbose:
        logger.info("Starting RAW to HDRI process...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        if debug_pixel:
            logger.info(f"Debug pixel: {debug_pixel}")
    
    # Step 1: Move loose images into stack folders if needed
    move_images_to_stack_folders(input_dir, time_delta, verbose)
    
    if organize_only:
        logger.info("Organize-only mode: Stopping after folder organization")
        return
    
    # Step 2: Find all stack folders
    stack_folders = find_stack_folders(input_dir)
    if not stack_folders:
        logger.error(f"No stack folders found in {input_dir}!")
        return
    
    if verbose:
        logger.info(f"Found {len(stack_folders)} stack folders:")
        for folder in stack_folders:
            logger.info(f"  {os.path.basename(folder)}")
            ev_folders = find_ev_offset_folders(folder)
            for ev_folder in ev_folders:
                ev_offset = get_ev_offset(os.path.basename(ev_folder))
                logger.info(f"    EV offset folder: {os.path.basename(ev_folder)} ({ev_offset:+.1f} EV)")
    
    # Step 3: Process each stack folder
    logger.info("\nProcessing exposure stacks...")
    fusion = ExposureFusion(verbose=verbose, debug_intermediate_results=args.debug_intermediate_results, debug_pixel=debug_pixel)
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in stack_folders:
        # Get all CR3 files in the main folder
        input_files = sorted(glob.glob(os.path.join(folder, '*.CR3')))
        if not input_files:
            logger.warning(f"No CR3 files found in {folder}, skipping.")
            continue
        
        # Get CR3 files from EV offset folders if they exist
        ev_folders = find_ev_offset_folders(folder)
        for ev_folder in ev_folders:
            ev_files = sorted(glob.glob(os.path.join(ev_folder, '*.CR3')))
            input_files.extend(ev_files)
        
        image_data_list: list[ImageData] = []
        for i, file_path in enumerate(input_files):
            if verbose:
                logger.info(f"\nProcessing {os.path.basename(file_path)} ({i+1}/{len(input_files)})")
            aces_file = process_raw_to_aces(file_path, verbose)
            try:
                aces = read_exr(aces_file, verbose)
                exp_info = get_exposure_info(file_path, verbose)
                
                # Get EV offset from folder name if in EV folder
                folder_path = os.path.dirname(file_path)
                ev_offset = 0.0
                if any(part.endswith('EV') for part in folder_path.split(os.sep)):
                    ev_offset = get_ev_offset(os.path.basename(folder_path))
                
                image_data = ImageData(
                    image=aces,
                    raw_path=file_path,
                    filename=os.path.basename(file_path),
                    shutter_speed=exp_info['shutter_speed'],
                    shutter_speed_str=exp_info['shutter_speed_str'],
                    ev=exp_info['ev'] + ev_offset,
                    ev_offset=ev_offset,
                    ev_original=exp_info['ev'],
                    capture_time=exp_info['capture_time'],
                    aperture=exp_info['aperture'],
                    iso=exp_info['iso']
                )
                image_data_list.append(image_data)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                raise
                
        # Output name based on folder
        output_name = f"{os.path.basename(folder)}_fused.exr"
        output_path = os.path.join(output_dir, output_name)
        fused = fusion.fuse(image_data_list)
        logger.info(f"Saving HDR image to {output_path}")
        save_exr(fused, output_path)
    logger.info("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert RAW files to HDR EXR using exposure fusion')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug-intermediate-results', '-d', action='store_true', 
                       help='Save intermediate results as EXR files for debugging')
    parser.add_argument('--time-delta', type=int, default=1,
                      help='Maximum time difference (in seconds) between images in the same stack')
    parser.add_argument('--input-dir', '-i', default='input',
                      help='Input directory containing RAW files (default: input)')
    parser.add_argument('--output-dir', '-o', default='output',
                      help='Output directory for HDR images (default: output)')
    parser.add_argument('--organize-only', action='store_true',
                      help='Only organize files into stack folders without processing them')
    parser.add_argument('--debug-pixel', type=str, help='Pixel coordinates to debug in format "x,y" (e.g., "100,200")')
    args = parser.parse_args()
    
    # Parse debug pixel coordinates if provided
    debug_pixel = None
    if args.debug_pixel:
        try:
            x, y = map(int, args.debug_pixel.split(','))
            debug_pixel = (x, y)
        except ValueError:
            logger.error("Debug pixel must be in format 'x,y' (e.g., '100,200')")
            sys.exit(1)
    
    main(verbose=args.verbose, time_delta=args.time_delta, 
         input_dir=args.input_dir, output_dir=args.output_dir,
         organize_only=args.organize_only, debug_pixel=debug_pixel)