import numpy as np
import matplotlib.pyplot as plt

def format_coord(x, y):
    return f'x={x:.3f}, y={y:.3f}'

def gaussian_weight(x, target, sigma):
    return np.exp(-(x - target)**2 / (2 * sigma**2))

def biweight_weight(x, target, sigma):
    # Tukey's biweight function
    z = (x - target) / sigma
    return np.where(np.abs(z) < 1, (1 - z**2)**2, 0)

def lorentzian_weight(x, target, sigma):
    # Lorentzian function
    return 1 / (1 + ((x - target) / sigma)**2)

def lorentzian_weight_split(x, target, sigma_dark, sigma_bright, linear_end=0.1):
    # Calculate the Lorentzian value at linear_end to connect the linear ramp
    lorentzian_at_end = 1 / (1 + ((linear_end - target) / sigma_dark)**2)
    
    # Linear ramp from 0 to linear_end, then Lorentzian
    return np.where(
        x <= linear_end,
        x * (lorentzian_at_end / linear_end),  # Linear ramp scaled to connect with Lorentzian
        np.where(
            x <= target,
            1 / (1 + ((x - target) / sigma_dark)**2),    # Darker side
            1 / (1 + ((x - target) / sigma_bright)**2)   # Brighter side
        )
    )

def sigmoid_weight(x, target, sigma):
    # Sigmoid-based weighting
    return 1 / (1 + np.exp((x - target) / sigma))

def plot_weight_functions(target=0.18, sigma=0.07):
    # Create input values from 0 to 2
    x = np.linspace(0, 2, 1000)
    
    # Calculate weights for each function
    weights_gaussian = gaussian_weight(x, target, sigma)
    weights_biweight = biweight_weight(x, target, sigma)
    weights_lorentzian = lorentzian_weight(x, target, sigma)
    weights_sigmoid = sigmoid_weight(x, target, sigma)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each function with different colors
    ax.plot(x, weights_gaussian, 'b-', label='Gaussian')
    ax.plot(x, weights_biweight, 'r-', label='Tukey\'s Biweight')
    ax.plot(x, weights_lorentzian, 'g-', label='Lorentzian')
    ax.plot(x, weights_sigmoid, 'm-', label='Sigmoid')
    
    # Add target line
    ax.axvline(x=target, color='k', linestyle='--', label=f'Target ({target})')
    
    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Weight')
    ax.set_title(f'Weight Functions Comparison (σ={sigma})')
    ax.legend()
    
    # Add interactive coordinate display
    ax.format_coord = format_coord
    
    plt.show()

def plot_lorentzian_variations(target=0.18, sigma_min=0.01, sigma_max=0.3, num_curves=10):
    # Create input values from 0 to 2
    x = np.linspace(0, 2, 1000)
    
    # Create sigma values
    sigmas = np.linspace(sigma_min, sigma_max, num_curves)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each sigma variation
    for sigma in sigmas:
        weights = lorentzian_weight(x, target, sigma)
        ax.plot(x, weights, label=f'σ={sigma:.3f}')
    
    # Add target line
    ax.axvline(x=target, color='k', linestyle='--', label=f'Target ({target})')
    
    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Weight')
    ax.set_title('Lorentzian Weight Function Variations')
    ax.legend()
    
    # Add interactive coordinate display
    ax.format_coord = format_coord
    
    plt.show()

def plot_lorentzian_split(target=0.18, sigma_dark=0.02, sigma_bright=0.1, linear_end=0.1):
    # Create input values from 0 to 2
    x = np.linspace(0, 2, 1000)
    
    # Calculate weights
    weights = lorentzian_weight_split(x, target, sigma_dark, sigma_bright, linear_end)
    
    # Calculate connection point
    lorentzian_at_end = 1 / (1 + ((linear_end - target) / sigma_dark)**2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the linear part
    linear_mask = x <= linear_end
    ax.plot(x[linear_mask], weights[linear_mask], 'g-', label='Linear Ramp')
    
    # Plot the Lorentzian part
    lorentzian_mask = x > linear_end
    ax.plot(x[lorentzian_mask], weights[lorentzian_mask], 'b-', 
            label=f'Lorentzian (σ_dark={sigma_dark:.3f}, σ_bright={sigma_bright:.3f})')
    
    # Add connection point marker and annotation
    ax.plot(linear_end, lorentzian_at_end, 'ro', label='Connection Point')
    ax.annotate(f'({linear_end:.3f}, {lorentzian_at_end:.3f})',
                xy=(linear_end, lorentzian_at_end),
                xytext=(linear_end + 0.05, lorentzian_at_end),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add target line
    ax.axvline(x=target, color='r', linestyle='--', label=f'Target ({target})')
    
    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel('Input Value')
    ax.set_ylabel('Weight')
    ax.set_title(f'Lorentzian Weight Function with Linear Ramp (end={linear_end})')
    ax.legend()
    
    # Add interactive coordinate display
    ax.format_coord = format_coord
    
    plt.show()

if __name__ == "__main__":
    # Plot all weight functions
    # plot_weight_functions(target=0.18, sigma=0.07)
    
    # Plot Lorentzian variations
    # plot_lorentzian_variations(target=0.18, sigma_min=0.01, sigma_max=0.3, num_curves=10)
    
    # Plot single Lorentzian with split sigmas
    plot_lorentzian_split(target=0.18, sigma_dark=0.05, sigma_bright=0.1, linear_end=0.07)