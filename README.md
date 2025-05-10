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

1. Place your CR3 files in the `input` directory
   - Files should be from the same scene with different exposures
   - Recommended: 5-7 exposures with 2 EV steps between them

2. Run the script:
```bash
python src/main.py -v
```

3. The resulting HDR EXR file will be saved in the `output` directory as `fused.exr`

### Command Line Options

- `-v` or `--verbose`: Enable detailed logging output
- `--time-delta`: Maximum time difference (in seconds) between images in the same stack (default: 1)

## Output

The script produces a scene-referred HDR EXR file that:
- Is in ACES color space
- Has a high dynamic range (typically 10-15 stops)
- Is suitable for use in HDR workflows
- Can be opened in software like Nuke, Houdini, or Blender

## Notes

- The script uses Debevec's method for HDR reconstruction
- Weights are computed using a smooth weighting function
- The middle exposure is used as a reference
- All images are converted to ACES color space before fusion

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

### HDR Fusion
- Uses Debevec's exposure fusion method
- Weights are calculated based on exposure values
- Properly handles the full HDR range from the input images

## Verification

- Check the verbose output to see how images are grouped into stacks
- Each stack's images should have similar capture times
- The output EXR files should have a wide dynamic range 