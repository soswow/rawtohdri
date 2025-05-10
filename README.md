# RAW to HDRI Exposure Fusion

This project implements exposure fusion for CR3 RAW files, converting them to ACES color space and producing a scene-referred HDR EXR output. It follows Debevec's method for HDR reconstruction from multiple exposures.

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
python src/main.py
```

3. The resulting HDR EXR file will be saved in the `output` directory as `fused.exr`

### Command Line Options

- `-v` or `--verbose`: Enable detailed logging output
```bash
python src/main.py -v
```

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