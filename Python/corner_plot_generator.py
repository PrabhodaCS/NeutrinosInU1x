#!/usr/bin/env python3
"""
Corner plot generator for neutrino inference posteriors.

Loads posterior samples from a saved ArviZ NetCDF file and generates a corner plot
(triangle plot) for the variables:
  - epsilon
  - log10_M0
  - sin²(θ₁₂)
  - sin²(θ₁₃)
  - sin²(θ₂₃)
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import corner

def load_posterior(idata_path):
    """Load posterior from NetCDF file."""
    print(f"Loading posterior from: {idata_path}")
    idata = az.from_netcdf(idata_path)
    print(f"Posterior shape: {idata.posterior.sizes}")
    return idata


def extract_and_transform_samples(idata):
    """
    Extract posterior samples.
    
    Returns:
        data_dict: Dictionary with posterior variable names
    """
    posterior = idata.posterior
    
    # Extract base variables
    eps_vals = posterior['epsilon'].values.reshape(-1)
    log10_M0_vals = posterior['log10_M0'].values.reshape(-1)
    
    # Extract sin²(θ) directly (already computed in model)
    sin2_theta12 = posterior['sin2_theta12'].values.reshape(-1)
    sin2_theta13 = posterior['sin2_theta13'].values.reshape(-1)
    sin2_theta23 = posterior['sin2_theta23'].values.reshape(-1)
    
    data_dict = {
        'epsilon': eps_vals,
        'log₁₀(M₀)': log10_M0_vals,
        'sin²(θ₁₂)': sin2_theta12,
        'sin²(θ₁₃)': sin2_theta13,
        'sin²(θ₂₃)': sin2_theta23,
    }
    
    return data_dict


def create_corner_plot_with_corner(data_dict, output_path):
    """
    Create corner plot using the corner package (publication-quality).
    
    Parameters:
        data_dict: Dictionary of variable names -> 1D arrays
        output_path: Path to save the figure
    """
    # Stack columns: (n_samples, n_variables)
    var_names = list(data_dict.keys())
    data = np.column_stack([data_dict[name] for name in var_names])
    
    # Create corner plot
    fig = corner.corner(
        data,
        labels=var_names,
        figsize=(12, 12),
        quantiles=[0.16, 0.5, 0.84],  # 1-sigma credible intervals on 1D marginalized posteriors
        show_titles=True,
        title_fmt='.3g',
        truths=None,
        truth_color='red',
        color='#1f77b4',
        bins=50,
        smooth=1.5,
        levels=[0.68, 0.95],  # 1-sigma and 2-sigma contours
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved to: {output_path}")
    plt.close()
'''color='#1f77b4',
color="#e39014",'''
def main():
    # Path to the posterior file
    # Adjust this path if needed
    base_path = Path("/mnt/c/Users/Asus/Documents/Cambridge/Project/Particle Physics Project/Python/resultsIO")
    
    idata_file = base_path / "posterior_idata.nc"
    
    if not idata_file.exists():
        print(f"Error: File not found at {idata_file}")
        print("\nPlease check the path and ensure the posterior_idata.nc file exists.")
        sys.exit(1)
    
    # Load posterior
    idata = load_posterior(str(idata_file))
    
    # Extract and transform samples
    data_dict = extract_and_transform_samples(idata)
    
    print(f"\nExtracted variables:")
    for name, vals in data_dict.items():
        print(f"  {name}: shape={vals.shape}, mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")
    
    # Create output directory if it doesn't exist
    output_dir = base_path / "corner_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try corner package first (best quality)
    print("\n=== Creating corner plot with corner package ===")
    output_path = output_dir / "corner_plot_corner.png"
    create_corner_plot_with_corner(data_dict, str(output_path))

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nSummary statistics:")
    for name, vals in data_dict.items():
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        print(f"  {name:15s}: {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}")


if __name__ == "__main__":
    main()
