import numpy as np
import cv2
from logging_config import setup_loggers
from image_data import ImageData
from exr_utils import save_exr

# Set up loggers
loggers = setup_loggers()
fusion_logger = loggers['fusion']
weight_logger = loggers['weight']

class ExposureFusion:
    def __init__(self, verbose=False, debug_intermediate_results=False):
        """
        Initialize the exposure fusion processor.
        
        Args:
            verbose: Whether to enable detailed logging
            debug_intermediate_results: Whether to save intermediate results as EXR files
        """
        self.verbose = verbose
        self.debug_intermediate_results = debug_intermediate_results
        
        if verbose:
            fusion_logger.info("Initialized ExposureFusion")

        if debug_intermediate_results:
            fusion_logger.info("Debug intermediate results enabled")
    
    def compute_weight_map(self, img: np.ndarray[tuple[int, int, int], np.dtype[np.float32]], ev: float, filename: str) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        """
        Compute exposure weight map using Debevec's smooth weighting function.
        Weights are calculated and kept separate for each RGB channel.
        
        Args:
            img: Input image of shape (height, width, 3) with RGB channels
            ev: Exposure value
            filename: Base filename for logging
            
        Returns:
            np.ndarray: Exposure weight map of shape (height, width, 3) with per-channel weights in [0,1]
        """ 
        # Smooth weighting function centered at 0.18

        target = 0.18  # Middle gray in linear space
        sigma_dark = 0.02    # Controls the spread of the function
        sigma_bright = 0.1  
        linear_end = 0.07     # End point of linear ramp

        # Calculate the Lorentzian value at linear_end to connect the linear ramp
        lorentzian_at_end = 1 / (1 + ((linear_end - target) / sigma_dark)**2)
        
        # Linear ramp from 0 to linear_end, then Lorentzian
        weights = np.where(
            img <= linear_end,
            img * (lorentzian_at_end / linear_end),  # Linear ramp scaled to connect with Lorentzian
            np.where(
                img <= target,
                1 / (1 + ((img - target) / sigma_dark)**2),    # Darker side
                1 / (1 + ((img - target) / sigma_bright)**2)   # Brighter side
            )
        )
        
        if self.verbose:
            fusion_logger.info(f"  Weights range:          [{weights.min():.3f}, {weights.max():.3f}]")
        
        return weights
    
    def fuse(self, image_data_list: list[ImageData]) -> np.ndarray:
        """
        Perform exposure fusion on a list of images using Debevec's method.
        
        Args:
            image_data_list: List of ImageData objects containing images and exposure info
            
        Returns:
            np.ndarray: Fused HDR image
        """
        if not image_data_list:
            raise ValueError("No images provided for fusion")
        
        if self.verbose:
            fusion_logger.info(f"Starting fusion of {len(image_data_list)} images")
        
        # Sort all images by total EV (base + offset)
        sorted_data: list[ImageData] = sorted(image_data_list, key=lambda x: x.ev)
        
        # Get reference EV value
        evs = [data.ev for data in sorted_data if data.ev_offset == 0.0]
        ref_ev_idx = len(evs) // 2
        ref_ev = evs[ref_ev_idx]
        
        # Perform fusion according to Debevec's formula:
        # E = Σ(w(Z_ij) * Z_ij / Δt_j) / Σ(w(Z_ij))
        fused = np.zeros_like(sorted_data[0].image, dtype=np.float32)
        weight_sum = np.zeros_like(sorted_data[0].image, dtype=np.float32)
        
        for i, data in enumerate(sorted_data):
            filename = data.filename
            ev = data.ev

            if data.image is None:
                fusion_logger.info(f"Skipping {filename} because image data is missing")
                continue
            
            if self.verbose:
                fusion_logger.info(f"\nImage {filename}:")

                is_reference_ev_str = i == ref_ev and '(reference)' or ''
                has_offset_ev_str = data.ev_offset != 0.0 and f'(offset: {data.ev_offset:+.1f}, original: {data.ev_original:.1f})' or ''
                fusion_logger.info(f"  EV =                 {ev:.1f} {is_reference_ev_str} {has_offset_ev_str} ({data.shutter_speed:.3f}s)")
                fusion_logger.info(f"  Input image range:   [{data.image.min():.3f}, {data.image.max():.3f}]")
                          
            image = np.clip(data.image, 0, None).astype(np.float32)
            
            exposure_scale = 2.0 ** (ref_ev - ev)
            scaled_image = image / exposure_scale
            
            weight = self.compute_weight_map(image, ev, filename)
            contribution = weight * scaled_image
            
            fused += contribution
            weight_sum += weight
            
            if self.verbose:
                fusion_logger.info(f"  exposure scale:      {exposure_scale:.3f}")
                fusion_logger.info(f"  contribution range:  [{contribution.min():.3f}, {contribution.max():.3f}]")

            if self.debug_intermediate_results:
                filename_base = filename.split(".")[0]
                # Save weight map and scaled image as debug images
                weights_debug_filename = f"debug_weights_{filename_base}.exr"
                scaled_image_debug_filename = f"debug_scaled_image_{filename_base}.exr"

                save_exr(weight, weights_debug_filename)
                fusion_logger.info(f"  Saved debug weight map to {weights_debug_filename}")
                
                save_exr(scaled_image, scaled_image_debug_filename)
                fusion_logger.info(f"  Saved debug scaled image to {scaled_image_debug_filename}")
            
        # Normalize by sum of weights
        fused = fused / (weight_sum + 1e-10)  # Add small epsilon to avoid division by zero
        
        if self.verbose:
            fusion_logger.info("Fusion complete")
            fusion_logger.info(f"  Output range: [{fused.min():.3f}, {fused.max():.3f}]")
            fusion_logger.info(f"  Output mean: {fused.mean():.3f}")
        
        return fused 
