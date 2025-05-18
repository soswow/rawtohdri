# RAW to HDRI

A Python tool for converting RAW image sequences to HDR images using exposure fusion. The tool automatically detects and groups images into exposure stacks based on capture time, making it easy to process multiple HDR sequences in a single run.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Prerequisites

### System Dependencies

1. **exiftool**
   - Required for extracting exposure metadata from RAW files
   - Install on macOS: `brew install exiftool`
   - Install on Ubuntu: `sudo apt-get install exiftool`
   - Install on Windows: Download from [ExifTool website](https://exiftool.org/)

2. **rawtoaces**
   - Required for converting RAW files to ACES color space
   - Install on macOS: `brew install rawtoaces`
   - Install on Ubuntu: `sudo apt-get install rawtoaces`
   - Install on Windows: Download from [ACES website](https://acescentral.com/t/aces-1-3-release/2497)

### Python Dependencies

All Python dependencies are listed in `requirements.txt` and include:
- numpy: For numerical operations
- opencv-python: For image processing
- OpenEXR: For HDR image I/O
- Imath: Required by OpenEXR

## Project Structure

```
rawtohdri/
├── src/
│   ├── main.py           # Main script
│   ├── exposure_fusion.py # HDR fusion implementation
│   ├── raw_metadata.py   # RAW file metadata extraction
│   ├── image_data.py     # Image data structure
│   ├── exr_utils.py      # EXR file I/O utilities
│   └── logging_config.py # Logging configuration
├── input/                # Place your CR3 files here
├── output/              # Generated EXR files will be saved here
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your RAW files in the input directory (default: `input/`)
2. Run the script:
   ```bash
   python src/main.py -v
   ```

The script will:
1. Read all RAW files from the input directory
2. Extract capture times and exposure information
3. Automatically group images into stacks based on capture time
4. Process each stack separately to create HDR images
5. Save the results in the output directory

### Command Line Options

- `-v, --verbose`: Enable detailed logging
- `-d, --debug-intermediate-results`: Save intermediate results (weight maps and scaled images) as EXR files for debugging
- `--time-delta`: Maximum time difference (in seconds) between images in the same stack (default: 1)
- `-i, --input-dir`: Input directory containing RAW files (default: input)
- `-o, --output-dir`: Output directory for HDR images (default: output)
- `--organize-only`: Only organize files into stack folders without processing them

### Examples
```bash
# Use default directories
python src/main.py -v

# Save intermediate results for debugging
python src/main.py -v -d

# Specify custom directories
python src/main.py -v -i /path/to/raws -o /path/to/output

# Adjust time delta for stacking
python src/main.py -v --time-delta 5

# Only organize files into stack folders without processing
python src/main.py -v --organize-only
```

### Output
- Each HDR image is saved as an EXR file in the output directory
- Files are named using the first and last image in each stack (e.g., `IMG_001_to_IMG_005_fused.exr`)
- The EXR files are in ACES color space and contain the full HDR range
- When using `--debug-intermediate-results`, additional EXR files are saved:
  - `debug_weights_*.exr`: Weight maps for each input image
  - `debug_scaled_image_*.exr`: Scaled input images before fusion

## Technical Notes

### Image Stacking
The tool automatically groups images into exposure stacks based on their capture times. This is useful when you have multiple HDR sequences in your input directory. For example:
- If you have 9 images taken in 3 different locations
- And each location has 3 exposures
- The tool will automatically create 3 separate HDR images, one for each location

The stacking is controlled by the `--time-delta` parameter:
- Images taken within this time window are considered part of the same stack
- Default is 1 second, which works well for most HDR sequences
- Adjust this value if your images are taken with longer intervals

### Directory Organization
- The tool will automatically organize loose images inside input folder into stack folders
- Each stack folder is named using the first and last image in the stack
- Only images in stack folders are processed
- You can also manually organize images into folders before running the script

#### EV Offset Subfolders
You can create subfolders with EV offsets inside stack folders to include images with different exposure ranges in the same stack. This is useful when:
- Using ND filters that reduce the light by a known amount
- Wanting to manually adjust the exposure value of some images in a stack
- Combining images with different exposure ranges into a single stack

Example folder structure:
```
input/
└── IMG_001-IMG_005/           # Main stack folder
    ├── IMG_001.CR3            # Normal exposure
    ├── IMG_002.CR3            # Normal exposure
    ├── IMG_003.CR3            # Normal exposure
    └── 10EV/                  # EV offset subfolder
        ├── IMG_004.CR3        # +10 EV offset
        └── IMG_005.CR3        # +10 EV offset
```

The EV offset is applied on top of the camera's exposure value, allowing you to:
1. Correct for ND filter usage
2. Manually adjust exposure values for better HDR fusion
3. Combine images with different exposure ranges into a single stack

### HDR Fusion
- Uses Debevec's exposure fusion method
- Weights are calculated based on exposure values
- Properly handles the full HDR range from the input images

## Troubleshooting

1. If you get "exiftool not found":
   - Ensure exiftool is installed and in your system PATH
   - Try running `exiftool -ver` to verify installation

2. If you get "rawtoaces not found":
   - Ensure rawtoaces is installed and in your system PATH
   - Try running `rawtoaces -h` to verify installation

3. If you get Python import errors:
   - Ensure you're in the virtual environment
   - Run `pip install -r requirements.txt` again

4. If you're having issues with the fusion:
   - Use `--debug-intermediate-results` to save intermediate files
   - Check the weight maps to verify exposure weighting
   - Inspect the scaled images to ensure proper exposure compensation

## Verification

- Check the verbose output to see how images are grouped into stacks
- Each stack's images should have similar capture times
- The output EXR files should have a wide dynamic range
- When using `--debug-intermediate-results`, inspect the intermediate files to verify the fusion process 