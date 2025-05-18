import numpy as np
import Imath
import OpenEXR

def save_exr(img: np.ndarray[tuple[int, int, int], np.dtype[np.float32]], filename: str):
    """
    Save an image as an EXR file.
    
    Args:
        img: Input image of shape (height, width, 3) with RGB channels
        filename: Output filename
        compression: OpenEXR compression method (default: PIZ_COMPRESSION)
    """
    # Ensure image is float32 and in the right memory layout
    img_float32 = img.astype(np.float32)
    
    # Create header
    header = OpenEXR.Header(img_float32.shape[1], img_float32.shape[0])
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    # Create output file
    exr = OpenEXR.OutputFile(filename, header)
    
    # Write each channel
    exr.writePixels({
        'R': img_float32[:,:,0].tobytes(),
        'G': img_float32[:,:,1].tobytes(),
        'B': img_float32[:,:,2].tobytes()
    })
    exr.close()