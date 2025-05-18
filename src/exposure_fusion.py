import numpy as np
import cv2
from logging_config import setup_loggers
from image_data import ImageData
from exr_utils import save_exr
from tabulate import tabulate

# Set up loggers
loggers = setup_loggers()
fusion_logger = loggers['fusion']
weight_logger = loggers['weight']

class ExposureFusion:
    def __init__(self, verbose=False, debug_intermediate_results=False, debug_pixel=None):
        """
        Initialize the exposure fusion processor.
        
        Args:
            verbose: Whether to enable detailed logging
            debug_intermediate_results: Whether to save intermediate results as EXR files
            debug_pixel: (x,y) coordinates of pixel to debug, or None
        """
        self.verbose = verbose
        self.debug_intermediate_results = debug_intermediate_results
        self.debug_pixel = debug_pixel
        
        if verbose:
            fusion_logger.info("Initialized ExposureFusion")

        if debug_intermediate_results:
            fusion_logger.info("Debug intermediate results enabled")
            
        if debug_pixel:
            fusion_logger.info(f"Debug pixel enabled at {debug_pixel}")
            self.table_data = []
            self.total_weight = 0
            self.total_contribution = 0
            
        # Initialize table for image processing details
        self.image_table_data = []
            
    def _format_value(self, value: float) -> str:
        """Format a value, showing ~0 for very small values."""
        if abs(value) < 0.00001:
            return "~0"
        return f"{value:.5f}"
    
    def compute_weight_map(self, img: np.ndarray[tuple[int, int, int], np.dtype[np.float32]], exposure_scale: float) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        """
        Compute exposure weight map using Debevec's smooth weighting function.
        Weights are calculated and kept separate for each RGB channel.
        
        Args:
            img: Input image of shape (height, width, 3) with RGB channels
            exposure_scale: Exposure scale for the image
            
        Returns:
            np.ndarray: Exposure weight map of shape (height, width, 3) with per-channel weights in [0,1]
        """ 
        # Smooth weighting function centered at 0.18
        target = 0.18  # Middle gray in linear space
        base_sigma_dark = 0.02    # Base sigma for dark regions
        base_sigma_bright = 0.1   # Base sigma for bright regions
        
        # Adjust sigmas based on exposure scale. The bigger the scale, the more weight 
        # concentration should be around the target value. Otherwise noise in very fast (or very slow) images
        # would be magnified by massive exposure scale (due to big EV diff from reference)
        #
        # For scale > 1: reduce bright sigma
        # For scale < 1: reduce dark sigma
        # Using log2 to get the power of 2 relationship
        if exposure_scale > 1:
            # For scale=10, log2(10)≈3.32, so sigma will be ~1/8 of base
            # For scale=100, log2(100)≈6.64, so sigma will be ~1/64 of base
            sigma_bright = base_sigma_bright / (2 ** np.log2(exposure_scale))
            sigma_dark = base_sigma_dark
        else:
            # For scale=1/10, log2(1/10)≈-3.32, so sigma will be ~1/8 of base
            # For scale=1/100, log2(1/100)≈-6.64, so sigma will be ~1/64 of base
            sigma_dark = base_sigma_dark / (2 ** abs(np.log2(exposure_scale)))
            sigma_bright = base_sigma_bright
        
        # Linear ramp from 0 to linear_end, then Lorentzian
        weights = np.where(
            img <= target,
            1 / (1 + ((img - target) / sigma_dark)**2),    # Darker side
            1 / (1 + ((img - target) / sigma_bright)**2)   # Brighter side
        )

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
                fusion_logger.info(f"\nProcessing {filename} ({i+1}/{len(sorted_data)}) ...")
            
            image = np.clip(data.image, 0, None).astype(np.float32)
            
            exposure_scale = 2.0 ** (ref_ev - ev)
            scaled_image = image / exposure_scale
            
            weight = self.compute_weight_map(image, exposure_scale)
            contribution = weight * scaled_image
            
            fused += contribution
            weight_sum += weight
            
            if self.verbose:
                # Collect image processing details for the table
                is_reference = ev == ref_ev
                ev_info = f"{ev:.1f}"
                if is_reference:
                    ev_info += " *"
                
                self.image_table_data.append([
                    f"{i+1}: {filename}",
                    ev_info,
                    f"{data.ev_offset:+.1f}" if data.ev_offset != 0.0 else "",
                    f"{data.ev_original:.1f}" if data.ev_offset != 0.0 else "",
                    f"{data.shutter_speed:.3f}s",
                    f"[{self._format_value(image.min())}, {self._format_value(image.max())}]",
                    f"[{self._format_value(weight.min())}, {self._format_value(weight.max())}]",
                    self._format_value(exposure_scale),
                    f"[{self._format_value(contribution.min())}, {self._format_value(contribution.max())}]"
                ])
            
            if self.debug_intermediate_results:
                filename_base = filename.split(".")[0]
                # Save weight map and scaled image as debug images
                weights_debug_filename = f"debug_weights_{filename_base}.exr"
                scaled_image_debug_filename = f"debug_scaled_image_{filename_base}.exr"

                save_exr(weight, weights_debug_filename)
                fusion_logger.info(f"  Saved debug weight map to {weights_debug_filename}")
                
                save_exr(scaled_image, scaled_image_debug_filename)
                fusion_logger.info(f"  Saved debug scaled image to {scaled_image_debug_filename}")
            
            # Add to debug table if pixel debugging is enabled
            if self.debug_pixel:
                x, y = self.debug_pixel
                pixel_value = image[y, x, 0]  # Red channel
                pixel_weight = weight[y, x, 0]
                pixel_scaled = scaled_image[y, x, 0]
                pixel_contribution = contribution[y, x, 0]
                
                self.table_data.append([
                    f"{i+1}: {filename}",
                    self._format_value(pixel_value),
                    f"{ev:.1f}",
                    self._format_value(exposure_scale),
                    self._format_value(pixel_scaled),
                    self._format_value(pixel_weight),
                    self._format_value(pixel_contribution)
                ])
                
                self.total_weight += pixel_weight
                self.total_contribution += pixel_contribution
            
        # Normalize by sum of weights
        fused = fused / (weight_sum + 1e-10)  # Add small epsilon to avoid division by zero
        
        if self.verbose:
            fusion_logger.info("Fusion complete")
            fusion_logger.info(f"  Output range: [{fused.min():.3f}, {fused.max():.3f}]")
            fusion_logger.info(f"  Output mean: {fused.mean():.3f}")
            
            # Print image processing details table
            fusion_logger.info("\nImage Processing Details:")
            headers = ["Image", "EV", "EV Offset", "Original EV", "Shutter", "Input Range", "Weight Range", "Scale", "Contribution Range"]
            fusion_logger.info("\n" + tabulate(self.image_table_data, headers=headers, tablefmt="grid"))
            
            # Print debug table if pixel debugging is enabled
            if self.debug_pixel:
                x, y = self.debug_pixel
                fusion_logger.info(f"\nPixel Debug Summary for ({x}, {y}):")
                headers = ["Image", "Actual Value", "EV", "Scale", "Scaled Value", "Weight", "Contribution"]
                fusion_logger.info("\n" + tabulate(self.table_data, headers=headers, tablefmt="grid"))
                fusion_logger.info(f"\nTotal weight: {self._format_value(self.total_weight)}")
                fusion_logger.info(f"Total contribution: {self._format_value(self.total_contribution)}")
                fusion_logger.info(f"Final value: {self._format_value(self.total_contribution/self.total_weight)}")
        
        return fused 
