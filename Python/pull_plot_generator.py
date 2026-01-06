from ast import main
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# --- NuFIT 5.2 (2024) Experimental Data ---
# --- NuFIT 5.2 (2024) Experimental Data ---
# Values are symmetric tuples: (central, sigma)
# Note: For angles, we use sin^2(theta)
NUFIT_DATA = {
    'sin²(θ₁₂)': (0.308, 0.0115),
    'sin²(θ₁₃)': (0.02231, 0.00056),  
    'sin²(θ₂₃)': (0.550, 0.0135),
    'Δm²₂₁':     (7.49e-5, 0.19e-5),
    'Δm²₃₁':     (2.484e-3, 0.020e-3)
}

EXP_DATA = {
    'dm21': (7.49e-5, 0.19e-5),     # eV^2
    'dm31': (2.484e-3, 0.020e-3),   # eV^2 (|Δm²₃₁|)
    # Angles: IO values with symmetric 1-sigma (average of asymmetric)
    's12':  (0.308, 0.0115),         # sin^2(theta12)  (0.308 +0.012 -0.011) -> sigma~0.0115
    's13':  (0.02231, 0.00056),      # sin^2(theta13) (0.0231 +0.00056 -0.00056) -> sigma~0.00056
    's23':  (0.550, 0.0135)          # sin^2(theta23) (0.550 +0.012 -0.015) -> sigma~0.0135
}
def load_posterior(idata_path):
    """Load posterior from NetCDF file."""
    print(f"Loading posterior from: {idata_path}")
    idata = az.from_netcdf(idata_path)
    print(f"Posterior shape: {idata.posterior.sizes}")
    return idata


def extract_and_transform_samples(idata):
    posterior = idata.posterior
    
    # 1. Extract base variables
    eps_vals = posterior['epsilon'].values.reshape(-1)
    log10_M0_vals = posterior['log10_M0'].values.reshape(-1)
    
    # 2. Extract and Transform angles to sin²(θ)
    s12_sq = posterior['sin2_theta12'].values.reshape(-1)
    s13_sq = posterior['sin2_theta13'].values.reshape(-1)
    s23_sq = posterior['sin2_theta23'].values.reshape(-1)
    
    data_dict = {
        'epsilon': eps_vals,
        'log₁₀(M₀)': log10_M0_vals,
        'sin²(θ₁₂)': s12_sq,
        'sin²(θ₁₃)': s13_sq,
        'sin²(θ₂₃)': s23_sq,
    }

    # 3. Extract Mass Squared Differences (Try multiple naming conventions)
    # Option A: If your model saved 'dm21' directly
    if 'dm21' in posterior:
        data_dict['dm21'] = posterior['dm21'].values.reshape(-1)
    elif 'dm_21' in posterior:
        data_dict['dm21'] = posterior['dm_21'].values.reshape(-1)
        
    if 'dm31' in posterior:
        data_dict['dm31'] = posterior['dm31'].values.reshape(-1)
    elif 'dm_31' in posterior:
        data_dict['dm31'] = posterior['dm_31'].values.reshape(-1)

    # Option B: If not found, calculate from m1, m2, m3
    if 'dm21' not in data_dict and 'm1' in posterior:
        m1 = posterior['m1'].values.reshape(-1)
        m2 = posterior['m2'].values.reshape(-1)
        m3 = posterior['m3'].values.reshape(-1)
        data_dict['dm21'] = m2**2 - m1**2
        data_dict['dm31'] = m3**2 - m1**2

    return data_dict


def create_physics_pull_plot(data_dict, output_path):
    """
    Generates a 'Traffic Light' Pull Plot comparing Model vs Experiment.
    Pulls > 0 mean Model is higher than Exp.
    Pulls < 0 mean Model is lower than Exp.
    """
    
    # 1. Map your variable names to the NuFIT keys
    # Adjust these keys to match exactly what is in your data_dict
    name_map = {
        'sin²(θ₁₂)': 'sin²(θ₁₂)',
        'sin²(θ₁₃)': 'sin²(θ₁₃)',
        'sin²(θ₂₃)': 'sin²(θ₂₃)',
        'dm21':      'Δm²₂₁',
        'dm31':      'Δm²₃₁' 
    }
    
    observables = []
    pulls = []
    pull_errors = []

    print("\n=== Calculating Pulls (Model vs Experiment) ===")
    
    for model_key, exp_key in name_map.items():
        if model_key not in data_dict:
            continue
            
        # Get Model stats
        samples = data_dict[model_key]
        model_mean = np.mean(samples)
        model_std = np.std(samples)
        
        # Get Exp stats (symmetric sigma)
        exp_val, exp_err = NUFIT_DATA[exp_key]
        
        # Calculate Pull
        # Formula: (Model - Exp) / Sigma_Exp (uses appropriate asymmetric sigma)
        pull = (model_mean - exp_val) / exp_err
        
        # Calculate 'Error on the Pull' 
        # This shows if your posterior is wider or narrower than the experiment
        pull_err = model_std / exp_err
        
        observables.append(exp_key)
        pulls.append(pull)
        pull_errors.append(pull_err)
        
        print(f"{exp_key}: Model={model_mean:.5f}, Exp={exp_val:.5f} => Pull={pull:.2f} sigma")

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Traffic Light Bands
    ax.axvspan(-1, 1, color='green', alpha=0.15, label=r'$1\sigma$ Agreement')
    ax.axvspan(1, 3, color='yellow', alpha=0.15, label=r'$2\sigma$ Tension')
    ax.axvspan(-3, -1, color='yellow', alpha=0.15)
    ax.axvspan(3, 5, color='red', alpha=0.1, label=r'$>3\sigma$ Exclusion')
    ax.axvspan(-5, -3, color='red', alpha=0.1)
    
    # The Dots
    y_pos = np.arange(len(observables))
    # We invert y_pos so the first item is at the top
    ax.errorbar(pulls, y_pos[::-1], xerr=pull_errors, fmt='o', 
                color='black', markersize=8, capsize=5, label='Model Prediction')
    
    # Formatting
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(observables, fontsize=12)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Pull: $(\mu_{model} - \mu_{exp}) / \sigma_{exp}$', fontsize=12)
    ax.set_xlim(-3.5, 3.5)
    ax.set_title(f"Model Goodness-of-Fit (Normal Ordering)", fontsize=14)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Physics Pull Plot saved to: {output_path}")

def main():
    # Path to the posterior file
    # Adjust this path if needed
    base_path = Path("/mnt/c/Users/Asus/Documents/Cambridge/Project/Particle Physics Project/Python/resultsIO")
    
    idata_file = base_path / "posterior_idata.nc"
    
    # Load posterior
    idata = load_posterior(str(idata_file))
    
    # Extract and transform samples
    data_dict = extract_and_transform_samples(idata)
    
    print(f"\nExtracted variables:")
    for name, vals in data_dict.items():
        print(f"  {name}: shape={vals.shape}, mean={np.mean(vals):.6f}, std={np.std(vals):.6f}")
    
    # Create output directory if it doesn't exist
    output_dir = base_path / "pull_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try corner package first (best quality)
    print("\n=== Creating corner plot with corner package ===")
    output_path = output_dir / "pull_plot.png"
    create_physics_pull_plot(data_dict, str(output_path))

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nSummary statistics:")
    for name, vals in data_dict.items():
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        print(f"  {name:15s}: {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}")

if __name__ == "__main__":
    main()