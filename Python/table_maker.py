import arviz as az
import numpy as np
from scipy.stats import norm
import jax.numpy as jnp

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to 'NO' or 'IO' depending on which table you want
ORDERING = 'IO'  

# Path to your .nc file
FILE_PATH = "resultsN/posterior_idata.nc" if ORDERING == 'NO' else "resultsIO/posterior_idata.nc"

# ==========================================
# EXPERIMENTAL DATA (NuFIT 6.0 with SK)
# ==========================================
# We define these in sin^2(theta) because that's what NuFIT provides.
# The code will automatically convert these to RADIANS for the table.
# Format: (Central, Sigma)

if ORDERING == 'NO':
    EXP_DATA = {
        'dm21': (7.49e-5, 0.19e-5),
        'dm31': (2.513e-3, 0.020e-3),
        's12':  (0.308, 0.012),
        's13':  (0.02215, 0.00057),
        's23':  (0.470, 0.015) 
    }
else: # IO
    EXP_DATA = {
        'dm21': (7.49e-5, 0.19e-5),
        'dm31': (2.484e-3, 0.020e-3), # Absolute value of dm31
        's12':  (0.308, 0.012),
        's13':  (0.02231, 0.00056),
        's23':  (0.550, 0.0135)
    }

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def to_radians(sin2_val, sin2_err):
    """Converts sin^2(theta) +/- err to Radians +/- err."""
    # theta = arcsin(sqrt(S))
    theta = np.arcsin(np.sqrt(sin2_val))
    # Error propagation: d(theta) = dS / sin(2*theta)
    theta_err = sin2_err / np.sin(2 * theta)
    return theta, theta_err

def z_test_pvalue(model_mean, exp_mean, exp_sigma):
    """Calculates two-sided p-value from Z-score."""
    z = (model_mean - exp_mean) / exp_sigma
    return 2 * (1 - norm.cdf(abs(z)))

def format_sci(val, err=None):
    """Formats numbers for LaTeX (e.g., 7.49 x 10^-5)."""
    # Helper to detect order of magnitude
    if val == 0: return "0"
    order = int(np.floor(np.log10(abs(val))))
    
    # If it's an order 1 number (like angles), just print decimals
    if -2 < order < 2:
        return f"{val:.4f}"
    
    # Otherwise use scientific notation
    norm_val = val / (10**order)
    norm_err = err / (10**order) if err else 0
    
    if err:
        return f"{norm_val:.4f}\\times 10^{{{order}}}"
    else:
        return f"{norm_val:.4f}\\times 10^{{{order}}}"

def get_tex_row(name, obs_key, samples, scale=1.0):
    """Generates a single row of the table."""
    # 1. Get Model Stats
    model_mean = np.mean(samples) * scale
    
    # 2. Get Experimental Stats (and convert if angle)
    exp_val_raw, exp_err_raw = EXP_DATA[obs_key]
    # If the observable key starts with 's' we treat it as sin^2(theta)
    # and report sin^2 values directly (matching NuFIT).
    if obs_key.startswith('s'):
        exp_val = exp_val_raw
        exp_err = exp_err_raw
        model_mean = np.mean(samples) * scale
    else:
        exp_val = exp_val_raw * scale
        exp_err = exp_err_raw * scale
        model_mean = np.mean(samples) * scale

    # 3. Calculate P-value
    pval = z_test_pvalue(model_mean, exp_val, exp_err)
    
    # 4. Format Latex
    # Name & Mean & Exp & Sigma & P-val
    
    # Special formatting for scientific notation columns
    model_str = format_sci(model_mean)
    exp_str = format_sci(exp_val)
    sig_str = format_sci(exp_err)
    
    return f"        {name} & ${model_str}$ & ${exp_str}$ & ${sig_str}$ & ${pval:.6f}$ \\\\"

# ==========================================
# MAIN SCRIPT
# ==========================================
print(f"Loading {FILE_PATH}...")
idata = az.from_netcdf(FILE_PATH)
post = idata.posterior

# Extract flattened samples
s12 = post['sin2_theta12'].values.flatten()
s13 = post['sin2_theta13'].values.flatten()
s23 = post['sin2_theta23'].values.flatten()
dm21 = post['dm21'].values.flatten()
dm31 = post['dm31'].values.flatten()
eps = post['epsilon'].values.flatten()
logM0 = post['log10_M0'].values.flatten()

# --- Calculate Combined PPP ---
# This requires recreating the Chi2 statistic
print("Calculating PPP...")
rng = np.random.default_rng(42)
n_samples = len(s12)

# Stack Model Samples
# Note: Angles in sin^2, Masses in eV^2
model_stack = np.vstack([dm21, dm31, s12, s13, s23]).T

# Stack Exp Means and Sigmas
exp_means = np.array([EXP_DATA['dm21'][0], EXP_DATA['dm31'][0], 
                      EXP_DATA['s12'][0], EXP_DATA['s13'][0], EXP_DATA['s23'][0]])
exp_sigs  = np.array([EXP_DATA['dm21'][1], EXP_DATA['dm31'][1], 
                      EXP_DATA['s12'][1], EXP_DATA['s13'][1], EXP_DATA['s23'][1]])

# Calculate Chi2 (Vectorized)
# Chi2_obs: Distance from Mean
chi2_obs = np.sum(((model_stack - exp_means)/exp_sigs)**2, axis=1)

# Chi2_rep: Generate synthetic data from model, check distance from model
# This simulates "If the model was true, how much would data fluctuate?"
synthetic_data = rng.normal(loc=model_stack, scale=exp_sigs)
chi2_rep = np.sum(((synthetic_data - model_stack)/exp_sigs)**2, axis=1)

ppp_val = np.mean(chi2_rep > chi2_obs)

# ==========================================
# PRINT LATEX TABLE
# ==========================================
print("\n" + "="*30)
print(" LATEX CODE (COPY BELOW)")
print("="*30 + "\n")

print(r"\begin{table}[h!]")
print(r"    \centering")
print(r"    \begin{tabular}{|c|c|c|c|c|}")
print(r"      \hline")
print(r"        Observable & Theoretical Mean & Experimental Value & $\sigma$ & p-value \\")
print(r"        \hline")
print(r"        \hline")

# Rows
print(get_tex_row(r"$\Delta m_{21}^2$", 'dm21', dm21, scale=1.0)) # Scale handled by format_sci
print(get_tex_row(r"$\Delta m_{31}^2$", 'dm31', dm31, scale=1.0))
print(get_tex_row(r"$\sin^2\theta_{12}$", 's12',  s12))
print(get_tex_row(r"$\sin^2\theta_{13}$", 's13',  s13))
print(get_tex_row(r"$\sin^2\theta_{23}$", 's23',  s23))

print(r"        \hline")
print(rf"        Combined \textbf{{PPP}} & & & & ${ppp_val:.6f}$ \\")
print(r"        \hline")

# Parameters
eps_mean = np.mean(eps)
eps_std = np.std(eps)
print(rf"        $\epsilon$ & ${eps_mean:.4f}$ &  & ${eps_std:.4f}$ &  \\")

m0_mean = np.mean(logM0)
m0_std = np.std(logM0)
print(rf"        $\log M_0$ & ${m0_mean:.4f}$ &  & ${m0_std:.4f}$ &  \\")

print(r"      \hline")
print(r"    \end{tabular}")
ordering_name = "Normal Hierarchy (NO)" if ORDERING == 'NO' else "Inverted Hierarchy (IO)"
print(rf"    \caption{{Posterior means and p-values for the {ordering_name} fit.")
print(r"    The theoretical means are obtained from the MCMC posterior, and the experimental values and uncertainties are taken from NuFIT 6.0 (2024) global fits (with SK).")
print(r"    The PPP entries represent the posterior predictive probabilities for the full observable set.}")
label = "NO Table" if ORDERING == 'NO' else "IO Table"
print(rf"    \label{{{label}}}")
print(r"\end{table}")