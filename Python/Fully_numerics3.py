#!/usr/bin/env python3
"""
optimized_neutrino_model_fixed_units_corrected_with_pvalues_full.py

Full integrated version:
 - samples A (Cholesky) entries O(1) and forms a = A @ A.T (order-1 coefficients a_{ij})
 - applies FN epsilon power pattern elementwise: MR_dimless = P * a
 - records a_ij and jitter used in adaptive inversion for diagnostics
 - fits Δm^2 (dm21, dm31) and angles directly (no ratio double counting)
 - computes posterior-predictive p-values (vectorized)
 - canonicalizes eigenvector columns (real-case) before extracting mixing angles
"""

import time
import os
import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

import arviz as az
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
from scipy.stats import norm  # for z-p-values

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# -----------------------
# Settings
# -----------------------
NUM_CHAINS = 10
SEED = 0

NUM_WARMUP = 20000
NUM_SAMPLES = 100000

TARGET_ACCEPT = 0.98

v = 246.22  # Higgs VEV in GeV
GEV2_TO_EV2 = 1e18  # conversion factor: 1 GeV^2 = 1e18 eV^2

DISPLAY_PLOTS = True

os.makedirs("resultsN/plots", exist_ok=True)

num_cores = multiprocessing.cpu_count()
numpyro.set_host_device_count(num_cores)

# -----------------------
# Numerics helpers
# -----------------------
@jax.jit
def adaptive_cholesky_inverse(A, rel_jitter=1e-8, min_add=1e-12):
    """
    Returns (invA, jitter) where jitter is the scalar added to the diagonal
    to make A positive-definite during Cholesky.
    """
    A_sym = 0.5 * (A + A.T)
    eigs = jnp.linalg.eigvalsh(A_sym)
    min_eig = jnp.min(eigs)
    normA = jnp.linalg.norm(A_sym)
    rel = rel_jitter * jnp.maximum(normA, 1e-12)
    needed = jnp.maximum(-min_eig + 1e-16, 0.0)
    jitter = jnp.maximum(rel, needed)
    jitter = jnp.maximum(jitter, min_add)
    A2 = A_sym + jnp.eye(A_sym.shape[0], dtype=A_sym.dtype) * jitter
    L = jnp.linalg.cholesky(A2)
    invA = jsp.cho_solve((L, True), jnp.eye(A2.shape[0], dtype=A2.dtype))
    return invA, jitter

# ---- New builder: create a = A @ A.T from sampled A entries, then apply epsilon powers ----
@jax.jit
def build_MR_from_A_entries(a_entries, eps):
    """
    a_entries ordering: [A11_pos, A21, A22_pos, A31, A32, A33_pos]
    A is lower-triangular:
      A = [[A11, 0,   0],
           [A21, A22, 0],
           [A31, A32, A33]]
    Then a = A @ A.T (dimensionless O(1) coefficients).
    Finally apply FN power-pattern elementwise: MR_dimless = eps_power_matrix * a
    Returns: MR_dimless, a_mat
    """
    A11_pos, A21, A22_pos, A31, A32, A33_pos = a_entries
    A = jnp.array([
        [A11_pos, 0.0,    0.0],
        [A21,     A22_pos,0.0],
        [A31,     A32,   A33_pos]
    ], dtype=eps.dtype)

    # a_ij coefficients (order 1 if A entries are order 1)
    a_mat = A @ A.T

    # FN power matrix matching your LaTeX pattern
    P = jnp.array([
        [eps**2, eps**3, eps**1],
        [eps**3, eps**4, eps**2],
        [eps**1, eps**2, eps**0]
    ], dtype=eps.dtype)

    MR_dimless = P * a_mat  # elementwise multiply
    return MR_dimless, a_mat

@jax.jit
def Yukawa(Y, eps):
    Y11 = Y[0, 0]
    Y12 = Y[0, 1] * eps
    Y13 = Y[0, 2] * eps
    Y21 = Y[1, 0] * eps
    Y23 = Y[1, 2] * (eps**2)
    Y31 = Y[2, 0] * eps
    Y32 = Y[2, 1] * (eps**2)
    Y33 = Y[2, 2]
    Ynu = jnp.array([[Y11, Y12, Y13],
                     [Y21,   0.0,  Y23],
                     [Y31,  Y32,  Y33]], dtype=eps.dtype)
    return Ynu

@jax.jit
def canonicalize_real_U(U):
    """
    For a real orthonormal U (columns are eigenvectors), fix the sign of each column
    so that the entry with largest absolute value in that column is positive.
    This removes column sign ambiguity from eigendecomposition for real symmetric matrices.
    """
    # column 0
    col0 = U[:, 0]
    idx0 = jnp.argmax(jnp.abs(col0))
    s0 = jnp.sign(col0[idx0])
    s0 = jnp.where(s0 == 0.0, 1.0, s0)  # if zero, keep as +1
    col0 = col0 * s0

    col1 = U[:, 1]
    idx1 = jnp.argmax(jnp.abs(col1))
    s1 = jnp.sign(col1[idx1])
    s1 = jnp.where(s1 == 0.0, 1.0, s1)
    col1 = col1 * s1

    col2 = U[:, 2]
    idx2 = jnp.argmax(jnp.abs(col2))
    s2 = jnp.sign(col2[idx2])
    s2 = jnp.where(s2 == 0.0, 1.0, s2)
    col2 = col2 * s2

    return jnp.column_stack([col0, col1, col2])

@jax.jit
def extract_angles_from_U(U):
    """
    Extracts mixing angles from a real PMNS-like matrix U, using absolute values
    and canonicalized columns. Returns theta12, theta13, theta23 (radians).
    """
    s13 = jnp.clip(jnp.abs(U[0, 2]), 0.0, 1.0)
    theta13 = jnp.arcsin(s13)
    theta12 = jnp.arctan2(jnp.abs(U[0, 1]), jnp.abs(U[0, 0]) + 1e-30)
    theta23 = jnp.arctan2(jnp.abs(U[1, 2]), jnp.abs(U[2, 2]) + 1e-30)
    return theta12, theta13, theta23

@jax.jit
def light_neutrino_mass_raw(Y, a_entries, eps, M0):
    """
    Build MR_dimless from a_entries (A-cholesky entries + eps powers),
    invert it safely (adaptive jitter), compute mnu (GeV) with prefactor v^2/(2*M0),
    and return MR diagnostics, jitter, and the underlying a_mat (order-1 coefficients).
    """
    Ynu = Yukawa(Y, eps)
    MR_dimless, a_mat = build_MR_from_A_entries(a_entries, eps)

    # MR diagnostics
    eigs = jnp.linalg.eigvalsh(0.5 * (MR_dimless + MR_dimless.T))
    mr_min_eig = eigs[0]
    mr_max_eig = eigs[-1]
    mr_cond = mr_max_eig / jnp.maximum(mr_min_eig, 1e-30)

    # invert MR_dimless with adaptive jitter
    MRinv_dimless, jitter = adaptive_cholesky_inverse(MR_dimless)

    # dimensionless mnu
    mnu_dimless = Ynu.T @ MRinv_dimless @ Ynu

    # apply physical prefactor -> mnu in GeV
    prefactor = (v ** 2) / (2.0 * M0)
    mnu = prefactor * mnu_dimless

    mnu_sym = 0.5 * (mnu + mnu.T)
    return mnu_sym, mr_min_eig, mr_cond, jitter, a_mat

@jax.jit
def observables(Y, a_entries, M0, eps):
    """
    Returns dm21_GeV2, dm31_GeV2, dm32_GeV2, theta12, theta13, theta23, U_can, mr_min_eig, mr_cond, jitter, a_mat
    """
    mnu, mr_min_eig, mr_cond, jitter, a_mat = light_neutrino_mass_raw(Y, a_entries, eps, M0)

    # eigen-decomposition
    w, V = jnp.linalg.eigh(mnu)   # real symmetric -> real eigenvalues+vectors
    idx = jnp.argsort(w)
    w_sorted = w[idx]
    V_sorted = V[:, idx]

    # canonicalize column signs to remove sign indeterminacy
    U_can = canonicalize_real_U(V_sorted)

    m1, m2, m3 = w_sorted[0], w_sorted[1], w_sorted[2]
    dm21_GeV2 = m2 ** 2 - m1 ** 2
    dm31_GeV2 = m3 ** 2 - m1 ** 2
    dm32_GeV2 = m3 ** 2 - m2 ** 2

    theta12, theta13, theta23 = extract_angles_from_U(U_can)

    return dm21_GeV2, dm31_GeV2, dm32_GeV2, theta12, theta13, theta23, U_can, mr_min_eig, mr_cond, jitter, a_mat

@jax.jit
def normal_logpdf_scalar(x, mu, sigma):
    sigma = jnp.maximum(sigma, 1e-16)
    z = (x - mu) / sigma
    return -0.5 * z * z - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))

# -----------------------
# Probabilistic model
# -----------------------
def model(*,
          eps_min=1e-1, eps_max=0.5,
          dm21_obs=7.50e-5, dm21_sigma=0.18e-5,
          dm31_obs=2.50e-3, dm31_sigma=0.04e-3,
          theta12_obs=None, theta12_sigma=None,
          theta13_obs=None, theta13_sigma=None,
          theta23_obs=None, theta23_sigma=None):

    # PDG2025-based angle central values
    if theta12_obs is None or theta12_sigma is None:
        s12, ds12 = 0.307, 0.012
        theta12_obs = jnp.arcsin(jnp.sqrt(s12))
        theta12_sigma = ds12 * (1.0 / (2.0 * jnp.sqrt(s12 * (1.0 - s12))))

    if theta13_obs is None or theta13_sigma is None:
        s13, ds13 = 0.0218, 0.0007
        theta13_obs = jnp.arcsin(jnp.sqrt(s13))
        theta13_sigma = ds13 * (1.0 / (2.0 * jnp.sqrt(s13 * (1.0 - s13))))

    if theta23_obs is None or theta23_sigma is None:
        s23, ds23 = 0.537, 0.020  # inverted-ordering PDG value
        theta23_obs = jnp.arcsin(jnp.sqrt(s23))
        theta23_sigma = ds23 * (1.0 / (2.0 * jnp.sqrt(s23 * (1.0 - s23))))

    # --- Priors ---
    # Yukawa hierarchical prior (lets data choose Yukawa scale)
    sigma_Y = numpyro.sample('sigma_Y', dist.HalfCauchy(0.5))
    Y_raw = numpyro.sample('Y', dist.Normal(0.0, sigma_Y).expand([3, 3]).to_event(2))

    # Priors for A (Cholesky factor A such that a = A @ A.T are O(1) coefficients)
    sigma_A = numpyro.sample('sigma_A', dist.HalfCauchy(0.8))
    # diagonals (positive)
    A11 = numpyro.sample('A11', dist.HalfNormal(sigma_A))
    A22 = numpyro.sample('A22', dist.HalfNormal(sigma_A))
    A33 = numpyro.sample('A33', dist.HalfNormal(sigma_A))
    # off-diagonals (real)
    A21 = numpyro.sample('A21', dist.Normal(0.0, sigma_A))
    A31 = numpyro.sample('A31', dist.Normal(0.0, sigma_A))
    A32 = numpyro.sample('A32', dist.Normal(0.0, sigma_A))

    a_entries = jnp.array([A11, A21, A22, A31, A32, A33], dtype=jnp.float64)

    # epsilon (FN parameter)
    eps = numpyro.sample('epsilon', dist.Uniform(eps_min, eps_max))

    # seesaw scale (log-uniform)
    log10_M0 = numpyro.sample('log10_M0', dist.Uniform(14.0, 16.0))
    M0 = 10.0 ** log10_M0
    numpyro.deterministic('M0', M0)

    # compute observables
    dm21_GeV2, dm31_GeV2, dm32_GeV2, theta12_th, theta13_th, theta23_th, U, mr_min_eig, mr_cond, jitter, a_mat = observables(Y_raw, a_entries, M0, eps)

    # convert to eV^2
    dm21 = dm21_GeV2 * GEV2_TO_EV2
    dm31 = dm31_GeV2 * GEV2_TO_EV2

    # deterministics for downstream analysis
    numpyro.deterministic('dm21', dm21)
    numpyro.deterministic('dm31', dm31)
    numpyro.deterministic('theta12_th', theta12_th)
    numpyro.deterministic('theta13_th', theta13_th)
    numpyro.deterministic('theta23_th', theta23_th)

    numpyro.deterministic('mr_min_eig_dimless', mr_min_eig)
    numpyro.deterministic('mr_cond_est', mr_cond)
    numpyro.deterministic('jitter_dimless', jitter)

    # record a_ij coefficients (order-1)
    numpyro.deterministic('a11', a_mat[0, 0])
    numpyro.deterministic('a12', a_mat[0, 1])
    numpyro.deterministic('a13', a_mat[0, 2])
    numpyro.deterministic('a22', a_mat[1, 1])
    numpyro.deterministic('a23', a_mat[1, 2])
    numpyro.deterministic('a33', a_mat[2, 2])

    # -----------------------
    # Likelihoods: direct Δm^2 and mixing angles
    # -----------------------
    numpyro.factor('logp_dm21', normal_logpdf_scalar(dm21, dm21_obs, dm21_sigma))
    numpyro.factor('logp_dm31', normal_logpdf_scalar(dm31, dm31_obs, dm31_sigma))

    numpyro.factor('logp_theta12', normal_logpdf_scalar(theta12_th, theta12_obs, theta12_sigma))
    numpyro.factor('logp_theta13', normal_logpdf_scalar(theta13_th, theta13_obs, theta13_sigma))
    numpyro.factor('logp_theta23', normal_logpdf_scalar(theta23_th, theta23_obs, theta23_sigma))

    penalty_k = 0.0
    if penalty_k > 0.0:
        numpyro.factor('pen_mr_cond', -penalty_k * jnp.log1p(mr_cond))


# -----------------------
# Diagnostics & plotting utilities
# -----------------------
def plot_posterior_publication(idata, style='modern'):
    posterior = idata.posterior
    dm21_vals = posterior['dm21'].values.reshape(-1)
    dm31_vals = posterior['dm31'].values.reshape(-1)
    theta12_vals = posterior['theta12_th'].values.reshape(-1)
    theta13_vals = posterior['theta13_th'].values.reshape(-1)
    theta23_vals = posterior['theta23_th'].values.reshape(-1)
    eps_vals = posterior['epsilon'].values.reshape(-1)
    log10_M0_vals = posterior['log10_M0'].values.reshape(-1)
    mr_min_eig_vals = posterior['mr_min_eig_dimless'].values.reshape(-1)
    mr_cond_vals = posterior['mr_cond_est'].values.reshape(-1)
    sigma_A_vals = posterior['sigma_A'].values.reshape(-1)
    sigma_Y_vals = posterior['sigma_Y'].values.reshape(-1)
    a11_vals = posterior['a11'].values.reshape(-1)
    a12_vals = posterior['a12'].values.reshape(-1)
    a13_vals = posterior['a13'].values.reshape(-1)

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # style presets (shared across angles and dm panels)
    styles = {
        'modern': {
            'hist_color': '#1f77b4', 'alpha': 0.85, 'gauss_color': '#222222',
            'exp_color': '#d62728', 'exp_alpha': 0.18,
            'box': dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='#333333'),
            'mass_colors': ['#1f77b4', '#17becf'],
            'font': {'family':'sans-serif', 'size':10}
        },
        'classic': {
            'hist_color': '#4C72B0', 'alpha': 0.9, 'gauss_color': '#000000',
            'exp_color': '#B22222', 'exp_alpha': 0.22,
            'box': dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.95, edgecolor='#000000'),
            'mass_colors': ['#4C72B0', '#55a868'],
            'font': {'family':'serif', 'size':11}
        },
        'compact': {
            'hist_color': '#2ca02c', 'alpha': 0.8, 'gauss_color': '#444444',
            'exp_color': '#ff7f0e', 'exp_alpha': 0.14,
            'box': dict(boxstyle='round4,pad=0.15', facecolor='white', alpha=0.9, edgecolor='#666666'),
            'mass_colors': ['#2ca02c', '#9467bd'],
            'font': {'family':'sans-serif', 'size':9}
        }
    }

    cfg = styles.get(style, styles['modern'])

    # angles panel with experimental overlays
    def plot_angles_with_overlay_panel(idata, style='modern'):
        posterior = idata.posterior
        theta12_vals = np.degrees(posterior['theta12_th'].values.reshape(-1))
        theta13_vals = np.degrees(posterior['theta13_th'].values.reshape(-1))
        theta23_vals = np.degrees(posterior['theta23_th'].values.reshape(-1))

        # Experimental values (degrees) and sigmas
        s12, ds12 = 0.307, 0.012
        theta12_exp = np.degrees(np.arcsin(np.sqrt(s12)))
        theta12_sig = np.degrees(ds12 * (1.0/(2.0*np.sqrt(s12*(1.0-s12)))))

        s13, ds13 = 0.0218, 0.0007
        theta13_exp = np.degrees(np.arcsin(np.sqrt(s13)))
        theta13_sig = np.degrees(ds13 * (1.0/(2.0*np.sqrt(s13*(1.0-s13)))))

        s23, ds23 = 0.537, 0.020
        theta23_exp = np.degrees(np.arcsin(np.sqrt(s23)))
        theta23_sig = np.degrees(ds23 * (1.0/(2.0*np.sqrt(s23*(1.0-s23)))))

        fig = plt.figure(figsize=(15,4))
        gs = gridspec.GridSpec(1,3, wspace=0.4)
        # style presets
        styles = {
            'modern': {
                'hist_color': '#1f77b4', 'alpha': 0.85, 'gauss_color': '#222222',
                'exp_color': '#d62728', 'exp_alpha': 0.18,
                'box': dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='#333333'),
                'mass_colors': ['#1f77b4', '#17becf'],
                'font': {'family':'sans-serif', 'size':10}
            },
            'classic': {
                'hist_color': '#4C72B0', 'alpha': 0.9, 'gauss_color': '#000000',
                'exp_color': '#B22222', 'exp_alpha': 0.22,
                'box': dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.95, edgecolor='#000000'),
                'mass_colors': ['#4C72B0', '#55a868'],
                'font': {'family':'serif', 'size':11}
            },
            'compact': {
                'hist_color': '#2ca02c', 'alpha': 0.8, 'gauss_color': '#444444',
                'exp_color': '#ff7f0e', 'exp_alpha': 0.14,
                'box': dict(boxstyle='round4,pad=0.15', facecolor='white', alpha=0.9, edgecolor='#666666'),
                'mass_colors': ['#2ca02c', '#9467bd'],
                'font': {'family':'sans-serif', 'size':9}
            }
        }

        cfg = styles.get(style, styles['modern'])
        plt.rcParams.update({'font.family': cfg['font']['family'], 'font.size': cfg['font']['size']})

        # Collect summary texts and draw histograms without inset elements
        mu_list = []
        std_list = []
        names_list = ['θ12', 'θ13', 'θ23']
        exps = [theta12_exp, theta13_exp, theta23_exp]

        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
        for ax, vals, exp, sig, name in zip(
            axes,
            [theta12_vals, theta13_vals, theta23_vals],
            [theta12_exp, theta13_exp, theta23_exp],
            [theta12_sig, theta13_sig, theta23_sig],
            ['θ12', 'θ13', 'θ23']
        ):
            # posterior histogram (draw above the experiment band)
            ax.hist(vals, bins=80, density=True, alpha=cfg['alpha'], color=cfg['hist_color'], label='Posterior', zorder=3)
            mu = np.mean(vals)
            std = np.std(vals)
            mu_list.append(mu)
            std_list.append(std)

            # Gaussian fit overlay
            x = np.linspace(np.percentile(vals,0.5), np.percentile(vals,99.5), 400)
            if std>0:
                pdf = 1.0/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/std)**2)
                ax.plot(x, pdf, linestyle='--', color=cfg['gauss_color'], label='Gaussian fit')

            # experimental central value and ±1σ band (full-axes vertical span)
            ax.axvspan(exp-sig, exp+sig, color=cfg['exp_color'], alpha=cfg['exp_alpha'], label='Exp ±1σ', zorder=0)
            # central line drawn above the shaded band
            ax.axvline(exp, color=cfg['exp_color'], linestyle='-', label='Exp central', zorder=5)

            ax.set_xlabel(f'{name} (deg)')
            ax.set_title(f'Posterior of {name}')

            # add a small legend-style box under each axis with Exp and Posterior summary
            caption = f"Exp={exp:.2f}° ±{sig:.2f}°  |  Post={mu:.2f}° ±{std:.2f}°"
            # place in axis-relative coords (x=0.5 centered, y negative to be below axis)
            ax.text(0.5, -0.22, caption, ha='center', va='top', transform=ax.transAxes,
                fontsize=cfg['font']['size']-1,
                bbox=cfg['box'])

        # Create a single, clear legend for the whole figure (clean, publication style)
        posterior_patch = plt.Rectangle((0,0),1,1, facecolor=cfg['hist_color'], alpha=cfg['alpha'])
        gauss_line = Line2D([0],[0], color=cfg['gauss_color'], linestyle='--')
        exp_line = Line2D([0],[0], color=cfg['exp_color'], linestyle='-')
        exp_band = plt.Rectangle((0,0),1,1, facecolor=cfg['exp_color'], alpha=cfg['exp_alpha'])

        # Use a compact legend frame for 'classic', frameless for 'modern', small for 'compact'
        legend_frame = True if style == 'classic' else False
        ncol = 4 if style != 'compact' else 2
        # choose legend vertical anchor and top margin per style so it doesn't overlap titles
        if style == 'modern':
            legend_y = 1.06
            top_margin = 0.82
        elif style == 'classic':
            legend_y = 0.99
            top_margin = 0.90
        else:  # compact
            legend_y = 0.99
            top_margin = 0.88

        fig.legend([posterior_patch, gauss_line, exp_line, exp_band],
                   ['Posterior (histogram)', 'Gaussian fit', 'Experiment central', 'Experiment ±1σ'],
                   loc='upper center', bbox_to_anchor=(0.5, legend_y), ncol=ncol, frameon=legend_frame, fontsize=cfg['font']['size'])

        # increase bottom margin to make room for per-axis boxes and set top to leave space for legend
        fig.subplots_adjust(top=top_margin, bottom=0.20)
        plt.tight_layout()
        outname = f'resultsN/plots/angles_panel_{style}.png'
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        if DISPLAY_PLOTS:
            plt.show()
        plt.close()

    plot_angles_with_overlay_panel(idata)

    # dm21 and dm31 with experimental overlays
    dm21_obs = 7.50e-5
    dm21_sigma = 0.18e-5
    dm31_obs = 2.50e-3
    dm31_sigma = 0.04e-3
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    col1, col2 = cfg.get('mass_colors', ['C1','C2'])
    # dm21
    ax1.hist(dm21_vals, bins=80, density=True, alpha=cfg['alpha'], color=col1, zorder=3)
    mu21 = np.mean(dm21_vals); std21 = np.std(dm21_vals)
    x21 = np.linspace(np.percentile(dm21_vals,0.5), np.percentile(dm21_vals,99.5), 400)
    if std21>0:
        pdf21 = 1.0/(std21*np.sqrt(2*np.pi)) * np.exp(-0.5*((x21-mu21)/std21)**2)
        ax1.plot(x21, pdf21, linestyle='--', color=cfg['gauss_color'], label=f'Gaussian fit μ={mu21:.2e}, σ={std21:.2e}', zorder=4)
    # experimental band (full height)
    ax1.axvspan(dm21_obs-dm21_sigma, dm21_obs+dm21_sigma, color=cfg['exp_color'], alpha=cfg['exp_alpha'], zorder=0)
    ax1.axvline(dm21_obs, color=cfg['exp_color'], linestyle='-', zorder=5)
    ax1.set_xlabel(r'$\Delta m^2_{21}$ (eV$^2$)'); ax1.set_title('Posterior of Δm²_21')
    # small caption box under axis
    caption1 = f"Exp={dm21_obs:.2e} ±{dm21_sigma:.2e}  |  Post={mu21:.2e} ±{std21:.2e}"
    ax1.text(0.5, -0.25, caption1, ha='center', va='top', transform=ax1.transAxes, fontsize=cfg['font']['size']-1, bbox=cfg['box'])
    ax1.legend()

    # dm31 (display in x10^-3 units)
    ax2 = plt.subplot(1,2,2)
    ax2.hist(dm31_vals, bins=80, density=True, alpha=cfg['alpha'], color=col2, zorder=3)
    mu31 = np.mean(dm31_vals); std31 = np.std(dm31_vals)
    x31 = np.linspace(np.percentile(dm31_vals,0.5), np.percentile(dm31_vals,99.5), 400)
    if std31>0:
        pdf31 = 1.0/(std31*np.sqrt(2*np.pi)) * np.exp(-0.5*((x31-mu31)/std31)**2)
        ax2.plot(x31, pdf31, linestyle='--', color=cfg['gauss_color'], label=f'Gaussian fit μ={mu31:.2e}, σ={std31:.2e}', zorder=4)
    ax2.axvspan(dm31_obs-dm31_sigma, dm31_obs+dm31_sigma, color=cfg['exp_color'], alpha=cfg['exp_alpha'], zorder=0)
    ax2.axvline(dm31_obs, color=cfg['exp_color'], linestyle='-', zorder=5)
    # scale ticks to 1e-3 units for readability
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x*1e3:.2f}"))
    ax2.set_xlabel(r'$\Delta m^2_{31}$ (10$^{-3}$ eV$^2$)')
    ax2.set_title('Posterior of Δm²_31')
    caption2 = f"Exp={dm31_obs*1e3:.3f} ±{dm31_sigma*1e3:.3f}  |  Post={mu31*1e3:.3f} ±{std31*1e3:.3f} (×10^-3)"
    ax2.text(0.5, -0.25, caption2, ha='center', va='top', transform=ax2.transAxes, fontsize=cfg['font']['size']-1, bbox=cfg['box'])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('resultsN/plots/dm_splittings_with_exp.png', dpi=300)
    if DISPLAY_PLOTS: plt.show()
    plt.close()

    # eps, log10_M0, and MR diagnostics
    plt.figure(figsize=(5,3.5))
    plt.hist(eps_vals, bins=60, density=True)
    plt.xlabel(r'$\varepsilon$'); plt.title('Posterior of epsilon')
    plt.tight_layout(); plt.savefig('resultsN/plots/epsilon_posterior.png', dpi=300)
    if DISPLAY_PLOTS: plt.show(); plt.close()

    plt.figure(figsize=(5,3.5))
    plt.hist(log10_M0_vals, bins=40, density=True)
    plt.xlabel(r'$\log_{10} M_0$'); plt.title('Posterior of log10_M0')
    plt.tight_layout(); plt.savefig('resultsN/plots/log10_M0_posterior.png', dpi=300)
    if DISPLAY_PLOTS: plt.show(); plt.close()

    plt.figure(figsize=(5,3.5))
    plt.hist(mr_min_eig_vals, bins=60, density=True)
    plt.xlabel('mr_min_eig_dimless'); plt.title('Posterior MR min eigenvalue'); plt.tight_layout()
    plt.savefig('resultsN/plots/mr_min_eig_dimless.png', dpi=300)
    if DISPLAY_PLOTS: plt.show(); plt.close()

    plt.figure(figsize=(5,3.5))
    plt.hist(mr_cond_vals, bins=60, density=True)
    plt.xlabel('mr_cond_est'); plt.title('Posterior MR cond estimate'); plt.tight_layout()
    plt.savefig('resultsN/plots/mr_cond_est.png', dpi=300)
    if DISPLAY_PLOTS: plt.show(); plt.close()

# -----------------------
# Runner
# -----------------------
def run_and_save(num_chains=NUM_CHAINS, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, seed=SEED):
    print("jax devices:", jax.device_count(), "local_device_count:", jax.local_device_count())
    print(f"Running MCMC: chains={num_chains}, warmup={num_warmup}, samples={num_samples}, target_accept={TARGET_ACCEPT}")

    kernel = NUTS(model, init_strategy=init_to_median(), dense_mass=True, target_accept_prob=TARGET_ACCEPT)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="parallel", jit_model_args=True)
    rng_key = jax.random.PRNGKey(seed)

    t0 = time.time()
    mcmc.run(rng_key)
    t1 = time.time()

    print(f"Sampling finished in {t1-t0:.1f}s")
    mcmc.print_summary()

    idata = az.from_numpyro(mcmc)
    idata.posterior.attrs["num_warmup"] = num_warmup
    idata.posterior.attrs["num_chains"] = num_chains
    idata.posterior.attrs["num_samples"] = num_samples
    idata.posterior.attrs["target_accept"] = TARGET_ACCEPT
    idata.posterior.attrs["seed"] = seed
    idata.posterior.attrs["time_seconds"] = t1-t0

    az.to_netcdf(idata, "resultsN/posterior_idata.nc")
    df = az.summary(idata, kind="all", round_to=4)
    df.to_csv("resultsN/summary_with_rhat_ess.csv")

    posterior = idata.posterior
    dm21_vals = posterior['dm21'].values.reshape(-1)
    dm31_vals = posterior['dm31'].values.reshape(-1)
    ratio_vals = posterior['dm21_dm31_ratio'].values.reshape(-1) if 'dm21_dm31_ratio' in posterior else (dm21_vals / (dm31_vals + 1e-30))
    eps_vals = posterior['epsilon'].values.reshape(-1)
    log10_M0_vals = posterior['log10_M0'].values.reshape(-1)
    mr_min_eig_vals = posterior['mr_min_eig_dimless'].values.reshape(-1)
    mr_cond_vals = posterior['mr_cond_est'].values.reshape(-1)

    summary_df = pd.DataFrame({
        'dm21_eV2': dm21_vals,
        'dm31_eV2': dm31_vals,
        'dm21_dm31_ratio': ratio_vals,
        'epsilon': eps_vals,
        'log10_M0': log10_M0_vals,
        'mr_min_eig_dimless': mr_min_eig_vals,
        'mr_cond_est': mr_cond_vals
    })
    summary_df.to_csv('resultsN/dm_summary_samples.csv', index=False)
    try:
        summary_df.describe().to_excel('resultsN/dm_summary_stats.xlsx')
    except Exception:
        summary_df.describe().to_csv('resultsN/dm_summary_stats.csv')

    try:
        plot_posterior_publication(idata)
    except Exception as e:
        print('Plotting failed:', e)

    # -----------------------
    # p-value block: vectorized posterior predictive p-values and z-tests
    # -----------------------
    print("\n--- P-value diagnostics (angles + masses combined) ---")

    # PDG angle obs & σ in radians
    s12, ds12 = 0.307, 0.012
    theta12_obs = np.arcsin(np.sqrt(s12))
    theta12_sigma = ds12 * (1.0 / (2.0 * np.sqrt(s12 * (1.0 - s12))))
    s13, ds13 = 0.0218, 0.0007
    theta13_obs = np.arcsin(np.sqrt(s13))
    theta13_sigma = ds13 * (1.0 / (2.0 * np.sqrt(s13 * (1.0 - s13))))
    s23, ds23 = 0.537, 0.020
    theta23_obs = np.arcsin(np.sqrt(s23))
    theta23_sigma = ds23 * (1.0 / (2.0 * np.sqrt(s23 * (1.0 - s23))))

    # Build arrays
    dm21_post = posterior['dm21'].values.reshape(-1)
    dm31_post = posterior['dm31'].values.reshape(-1)
    theta12_post = posterior['theta12_th'].values.reshape(-1)
    theta13_post = posterior['theta13_th'].values.reshape(-1)
    theta23_post = posterior['theta23_th'].values.reshape(-1)

    # Observed values and sigmas
    dm21_obs = 7.50e-5
    dm21_sigma = 0.18e-5
    dm31_obs = 2.50e-3
    dm31_sigma = 0.04e-3
    obs_vec = np.array([dm21_obs, dm31_obs, theta12_obs, theta13_obs, theta23_obs])
    sig_vec = np.array([dm21_sigma, dm31_sigma, theta12_sigma, theta13_sigma, theta23_sigma])

    # Stack posterior draws (n_draws, 5)
    post_stack = np.vstack([dm21_post, dm31_post, theta12_post, theta13_post, theta23_post]).T
    n_draws = post_stack.shape[0]

    rng = np.random.default_rng(seed)

    obs = obs_vec[None, :]   # shape (1,5)
    sigs = sig_vec[None, :]  # shape (1,5)

    chi2_obs = np.sum(((obs - post_stack)/sigs)**2, axis=1)            # vectorized
    yrep = rng.normal(loc=post_stack, scale=sigs)                      # shape (n_draws,5)
    chi2_rep = np.sum(((yrep - post_stack)/sigs)**2, axis=1)

    ppp_5var = float(np.mean(chi2_rep >= chi2_obs))
    print(f"Posterior-predictive p-value for (dm21, dm31, θ12, θ13, θ23): {ppp_5var:.4f}")

    # angle-only PPP
    angles_stack = post_stack[:, 2:5]
    obs_angles = obs_vec[2:5][None, :]
    sig_angles = sig_vec[2:5][None, :]
    chi2_obs_ang = np.sum(((obs_angles - angles_stack)/sig_angles)**2, axis=1)
    yrep_ang = rng.normal(loc=angles_stack, scale=sig_angles)
    chi2_rep_ang = np.sum(((yrep_ang - angles_stack)/sig_angles)**2, axis=1)
    ppp_angles = float(np.mean(chi2_rep_ang >= chi2_obs_ang))
    print(f"Posterior-predictive p-value for angles only: {ppp_angles:.4f}")

    # z-tests comparing posterior mean to observation
    mean_dm21 = float(np.mean(dm21_post))
    mean_dm31 = float(np.mean(dm31_post))
    mean_t12 = float(np.mean(theta12_post))
    mean_t13 = float(np.mean(theta13_post))
    mean_t23 = float(np.mean(theta23_post))

    def two_sided_p(z):
        return 2.0 * (1.0 - norm.cdf(abs(z)))

    z_dm21 = (mean_dm21 - dm21_obs) / dm21_sigma; p_dm21 = two_sided_p(z_dm21)
    z_dm31 = (mean_dm31 - dm31_obs) / dm31_sigma; p_dm31 = two_sided_p(z_dm31)
    z12 = (mean_t12 - theta12_obs) / theta12_sigma; p12 = two_sided_p(z12)
    z13 = (mean_t13 - theta13_obs) / theta13_sigma; p13 = two_sided_p(z13)
    z23 = (mean_t23 - theta23_obs) / theta23_sigma; p23 = two_sided_p(z23)

    print("Frequentist two-sided p-values (masses + angles):")
    print(f" dm21: z={z_dm21:.3f}, p={p_dm21:.4f}")
    print(f" dm31: z={z_dm31:.3f}, p={p_dm31:.4f}")
    print(f" θ12: z={z12:.3f}, p={p12:.4f}")
    print(f" θ13: z={z13:.3f}, p={p13:.4f}")
    print(f" θ23: z={z23:.3f}, p={p23:.4f}")

    # Save p-values
    pvals_df = pd.DataFrame({
        'var': ['dm21', 'dm31', 'theta12', 'theta13', 'theta23'],
        'mean_pred': [mean_dm21, mean_dm31, mean_t12, mean_t13, mean_t23],
        'obs': [dm21_obs, dm31_obs, theta12_obs, theta13_obs, theta23_obs],
        'sigma': [dm21_sigma, dm31_sigma, theta12_sigma, theta13_sigma, theta23_sigma],
        'z_value': [z_dm21, z_dm31, z12, z13, z23],
        'two_sided_p': [p_dm21, p_dm31, p12, p13, p23]
    })
    pvals_df.loc[len(pvals_df)] = ['combined_5var_ppp', None, None, None, None, ppp_5var]
    pvals_df.loc[len(pvals_df)] = ['combined_angles_ppp', None, None, None, None, ppp_angles]
    pvals_df.to_csv('resultsN/pvalues_summary.csv', index=False)
    print("Saved p-values to resultsN/pvalues_summary.csv")

    print("\n--- Summary of means & std devs ---")
    print('θ12 mean,std (deg):', np.degrees(mean_t12), np.degrees(np.std(theta12_post)))
    print('θ13 mean,std (deg):', np.degrees(mean_t13), np.degrees(np.std(theta13_post)))
    print('θ23 mean,std (deg):', np.degrees(mean_t23), np.degrees(np.std(theta23_post)))

    print("Saved posterior & summary to resultsN/")
    return mcmc, idata

# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    mcmc, idata = run_and_save()
