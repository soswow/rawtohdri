import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import cast

def calculate_ev(shutter_speed, aperture):
    """
    Calculate EV (Exposure Value) from shutter speed and aperture.
    
    Args:
        shutter_speed (float): Shutter speed in seconds (e.g., 1/125 = 0.008)
        aperture (float): Aperture f-number (e.g., f/2.8 = 2.8)
    
    Returns:
        float: EV value
    """
    # EV = log2(L × S / K)
    # where L is the luminance, S is the ISO speed, and K is the calibration constant
    # For standard conditions (ISO 100), K = 12.5
    # We can simplify to: EV = log2(1/S) + log2(N²)
    # where S is shutter speed and N is the f-number
    
    # Convert shutter speed to EV component
    ev_shutter = -np.log2(shutter_speed)
    
    # Convert aperture to EV component
    ev_aperture = 2 * np.log2(aperture)
    
    # Total EV
    return ev_shutter + ev_aperture

def plot_ev_surface():
    # Create a grid of shutter speeds and apertures
    shutter_speeds = np.array([1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4, 1/2, 1])
    apertures = np.array([1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0])
    
    # Create meshgrid
    S, N = np.meshgrid(shutter_speeds, apertures)
    
    # Calculate EV values
    EV = calculate_ev(S, N)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
    
    # Plot surface
    surf = ax.plot_surface(np.log2(1/S), np.log2(N), EV, cmap='viridis',
                          linewidth=0, antialiased=True)
    
    # Customize the plot
    ax.set_xlabel('Shutter Speed (EV)')
    ax.set_ylabel('Aperture (EV)')
    ax.set_zlabel('Total EV')
    ax.set_title('Exposure Value Surface')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Format axis labels
    shutter_labels = [f"1/{int(1/s)}" for s in shutter_speeds]
    aperture_labels = [f"f/{a:.1f}" for a in apertures]
    
    ax.set_xticks(np.log2(1/shutter_speeds))
    ax.set_xticklabels(shutter_labels, rotation=45)
    
    ax.set_yticks(np.log2(apertures))
    ax.set_yticklabels(aperture_labels)
    
    plt.tight_layout()
    plt.show()

def plot_ev_contour():
    # Create a grid of shutter speeds and apertures
    shutter_speeds = np.array([1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4, 1/2, 1])
    apertures = np.array([1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0])
    
    # Create meshgrid
    S, N = np.meshgrid(shutter_speeds, apertures)
    
    # Calculate EV values
    EV = calculate_ev(S, N)
    
    # Create contour plot
    plt.figure(figsize=(12, 8))
    
    # Plot contour
    contour = plt.contour(np.log2(1/S), np.log2(N), EV, levels=20, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Customize the plot
    plt.xlabel('Shutter Speed (EV)')
    plt.ylabel('Aperture (EV)')
    plt.title('Exposure Value Contour Plot')
    
    # Format axis labels
    shutter_labels = [f"1/{int(1/s)}" for s in shutter_speeds]
    aperture_labels = [f"f/{a:.1f}" for a in apertures]
    
    plt.xticks(np.log2(1/shutter_speeds), shutter_labels, rotation=45)
    plt.yticks(np.log2(apertures), aperture_labels)
    
    plt.colorbar(contour, label='EV')
    plt.tight_layout()
    plt.show()

def plot_ev_combinations(target_ev=12):
    """
    Plot all possible combinations of shutter speed and aperture that give the target EV.
    """
    # Create a grid of shutter speeds and apertures
    shutter_speeds = np.array([1/8000, 1/4000, 1/2000, 1/1000, 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8, 1/4, 1/2, 1])
    apertures = np.array([1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0])
    
    # Create meshgrid
    S, N = np.meshgrid(shutter_speeds, apertures)
    
    # Calculate EV values
    EV = calculate_ev(S, N)
    
    # Find combinations close to target EV
    tolerance = 0.5
    valid_combinations = np.abs(EV - target_ev) <= tolerance
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(np.log2(1/S), np.log2(N), c='gray', alpha=0.3, label='All combinations')
    
    # Plot valid combinations
    valid_shutter = np.log2(1/S[valid_combinations])
    valid_aperture = np.log2(N[valid_combinations])
    plt.scatter(valid_shutter, valid_aperture, c='red', label=f'EV = {target_ev} ± {tolerance}')
    
    # Customize the plot
    plt.xlabel('Shutter Speed (EV)')
    plt.ylabel('Aperture (EV)')
    plt.title(f'Exposure Value Combinations (Target EV = {target_ev})')
    
    # Format axis labels
    shutter_labels = [f"1/{int(1/s)}" for s in shutter_speeds]
    aperture_labels = [f"f/{a:.1f}" for a in apertures]
    
    plt.xticks(np.log2(1/shutter_speeds), shutter_labels, rotation=45)
    plt.yticks(np.log2(apertures), aperture_labels)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Plot 3D surface
    plot_ev_surface()
    
    # Plot contour map
    plot_ev_contour()
    
    # Plot combinations for a specific EV
    plot_ev_combinations(target_ev=12) 