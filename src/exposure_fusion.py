import numpy as np
import cv2
from logging_config import setup_loggers
from image_data import ImageData

# Set up loggers
loggers = setup_loggers()
fusion_logger = loggers['fusion']
weight_logger = loggers['weight']

class ExposureFusion:
    def __init__(self, verbose=False):
        """
        Initialize the exposure fusion processor.
        
        Args:
            verbose: Whether to enable detailed logging
        """
        self.verbose = verbose
        
        if verbose:
            fusion_logger.info("Initialized ExposureFusion")
    
    def compute_exposure(self, img: np.ndarray, ev: float) -> np.ndarray:
        """
        Compute exposure weight map using Debevec's smooth weighting function.
        
        Args:
            img: Input image
            ev: Exposure value
            
        Returns:
            np.ndarray: Exposure weight map
        """
        # Convert to float32 if needed
        if img.dtype == np.float16:
            img = img.astype(np.float32)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if self.verbose:
            fusion_logger.info(f"  Input image range: [{gray.min():.3f}, {gray.max():.3f}]")
            fusion_logger.info(f"  EV difference from middle: {ev:.1f}")
        
        # Debevec's smooth weighting function
        # w(z) = z - Zmin for z <= (Zmin + Zmax)/2
        # w(z) = Zmax - z for z > (Zmin + Zmax)/2
        z_min = 0.0
        z_max = 1.0
        z_mid = (z_min + z_max) / 2.0
        
        # Normalize input to [0,1] range
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min())
        
        # Apply Debevec's weighting function
        weights = np.where(
            gray_norm <= z_mid,
            gray_norm - z_min,  # For darker pixels
            z_max - gray_norm   # For brighter pixels
        )
        
        if self.verbose:
            fusion_logger.info(f"  Base weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        return weights
    
    def normalize_weights(self, weights: list[np.ndarray]) -> np.ndarray:
        """Normalize weight maps to sum to 1 at each pixel."""
        stacked: np.ndarray = np.stack(weights, axis=0)
        
        if self.verbose:
            fusion_logger.info("Weight normalization:")
            fusion_logger.info(f"  Stacked shape: {stacked.shape}")
            fusion_logger.info(f"  Stacked range: [{stacked.min():.3f}, {stacked.max():.3f}]")
        
        # Add small epsilon to avoid division by zero
        sum_weights = np.sum(stacked, axis=0) + 1e-10
        
        if self.verbose:
            fusion_logger.info(f"  Sum weights range: [{sum_weights.min():.3f}, {sum_weights.max():.3f}]")
        
        normalized: np.ndarray = stacked / sum_weights
        
        if self.verbose:
            fusion_logger.info(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
            fusion_logger.info(f"  Normalized mean: {normalized.mean():.3f}")
        
        return normalized
    
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
        
        # Sort images by EV to find middle exposure
        sorted_data = sorted(image_data_list, key=lambda x: x.ev)
        middle_idx = len(sorted_data) // 2
        middle_ev = sorted_data[middle_idx].ev
        
        if self.verbose:
            fusion_logger.info(f"Using middle exposure as reference (EV: {middle_ev:.1f})")
            for i, data in enumerate(sorted_data):
                fusion_logger.info(f"  Image {i+1}: EV = {data.ev:.1f}")
        
        # Extract images and exposure info
        images = [data.image for data in sorted_data]
        evs = [data.ev for data in sorted_data]
        
        # Convert EV differences to exposure time ratios
        # For EV differences, each stop is a factor of 2
        # So if image A is 2 stops brighter than image B, its exposure time is 1/4 of B's
        exposure_ratios = np.array([2.0 ** (middle_ev - ev) for ev in evs])
        
        if self.verbose:
            fusion_logger.info("Exposure ratios relative to middle:")
            for i, r in enumerate(exposure_ratios):
                fusion_logger.info(f"  Image {i+1}: {r:.6f}")
        
        # Compute exposure weights
        weights = [self.compute_exposure(img, ev - middle_ev) for img, ev in zip(images, evs)]
        
        if self.verbose:
            fusion_logger.info("Computed exposure weights:")
            for i, w in enumerate(weights):
                fusion_logger.info(f"  Image {i+1} weights: [{w.min():.3f}, {w.max():.3f}]")
        
        # Normalize weights
        weights = self.normalize_weights(weights)
        
        # Perform fusion according to Debevec's formula:
        # E = Σ(w(Z_ij) * Z_ij / Δt_j) / Σ(w(Z_ij))
        fused = np.zeros_like(images[0])
        for i, (img, weight, ratio) in enumerate(zip(images, weights, exposure_ratios)):
            # Scale by inverse of exposure ratio (divide by exposure time)
            contribution = (img / ratio) * weight[..., np.newaxis]
            if self.verbose:
                fusion_logger.info(f"  Image {i+1} contribution range: [{contribution.min():.3f}, {contribution.max():.3f}]")
            fused += contribution
        
        if self.verbose:
            fusion_logger.info("Fusion complete")
            fusion_logger.info(f"  Output range: [{fused.min():.3f}, {fused.max():.3f}]")
            fusion_logger.info(f"  Output mean: {fused.mean():.3f}")
        
        return fused 