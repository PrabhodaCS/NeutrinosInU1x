#!/usr/bin/env python3
"""
optimized_neutrino_model_io_eiginv.py
Inverted-hierarchy ONLY. MR built with FN eps powers and exact a11=a13=0.
MR inversion uses symmetric eigendecomposition + clipped eigenvalue inversion
to handle indefinite / tiny-eigenvalue matrices robustly while preserving physics.
"""

import time, os, math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

import arviz as az
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
from scipy.stats import norm  # for z-tests

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

outdir = "resultsIO_eig"
os.makedirs(outdir + "/plots", exist_ok=True)

num_cores = multiprocessing.cpu_count()
numpyro.set_host_device_count(num_cores)

# -----------------------
# Numerical knobs for eigen-clipping inversion
# -----------------------
# tiny relative floor for eigenvalues (in MR_dimless units)
TINY_ABS_EIG = 1e-12  

# -----------------------
# Helpers: symmetric eig-inverse with clipping
# -----------------------
@jax.jit
def symmetric_eiginv_clipped(A, tiny_abs_eig=TINY_ABS_EIG):
    """
    For symmetric A, return (A_inv, eigs, n_clipped, clipped_amount)
    A_inv computed as Q diag(1/lambda_clipped) Q^T with
      lambda_clipped = sign(lambda) * max(|lambda|, tiny_abs_eig)
    """
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)  # eigvals ascending
    abs_eigs = jnp.abs(eigvals)
    # Clip magnitude, preserve sign:
    clipped_mag = jnp.maximum(abs_eigs, tiny_abs_eig)
    lam_clipped = jnp.sign(eigvals) * clipped_mag
    inv_lam = 1.0 / lam_clipped
    Ainv = (eigvecs * inv_lam[None, :]) @ eigvecs.T
    n_clipped = jnp.sum(abs_eigs < tiny_abs_eig)
    # return diagnostic: minimum absolute eigenvalue before clipping
    min_abs_eig = jnp.min(abs_eigs)
    max_abs_eig = jnp.max(abs_eigs)
    return Ainv, eigvals, n_clipped, min_abs_eig, max_abs_eig

# -----------------------
# Build MR from a-vector (a12,a22,a23,a33) with eps-powerings and exact zeros
# -----------------------
@jax.jit
def build_MR_from_a_vec_io(a_vec, eps):
    a12, a22, a23, a33 = a_vec[0], a_vec[1], a_vec[2], a_vec[3]
    MR = jnp.array([
        [0.0 * (eps**2),         a12 * (eps**3),     0.0 * (eps**1)],
        [a12 * (eps**3),    a22 * (eps**4),    a23 * (eps**2)],
        [0.0 * (eps**1),     a23 * (eps**2),         a33]
    ], dtype=a_vec.dtype)
    return MR

# -----------------------
# Yukawa texture per FN (keep y22 nonzero for IO)
# -----------------------
@jax.jit
def Yukawa(Y, eps):
    Y11 = Y[0, 0]
    Y12 = Y[0, 1] * eps
    Y13 = Y[0, 2] * eps
    Y21 = Y[1, 0] * eps
    Y22 = Y[1, 1]          # user said NO forcing zero here for IO Yukawa
    Y23 = Y[1, 2] * (eps**2)
    Y31 = Y[2, 0] * eps
    Y32 = Y[2, 1] * (eps**2)
    Y33 = Y[2, 2]
    Ynu = jnp.array([[Y11,  Y12, Y13],
                     [Y21,  Y22,  Y23],
                     [Y31,  Y32,  Y33]], dtype=eps.dtype)
    return Ynu

# canonicalize eigenvector columns for real case
@jax.jit
def canonicalize_real_U(U):
    col0 = U[:, 0]
    idx0 = jnp.argmax(jnp.abs(col0))
    s0 = jnp.sign(col0[idx0]); s0 = jnp.where(s0 == 0.0, 1.0, s0); col0 = col0 * s0
    col1 = U[:, 1]
    idx1 = jnp.argmax(jnp.abs(col1))
    s1 = jnp.sign(col1[idx1]); s1 = jnp.where(s1 == 0.0, 1.0, s1); col1 = col1 * s1
    col2 = U[:, 2]
    idx2 = jnp.argmax(jnp.abs(col2))
    s2 = jnp.sign(col2[idx2]); s2 = jnp.where(s2 == 0.0, 1.0, s2); col2 = col2 * s2
    return jnp.column_stack([col0, col1, col2])

@jax.jit
def extract_angles_from_U(U):
    s13 = jnp.clip(jnp.abs(U[0, 2]), 0.0, 1.0)
    theta13 = jnp.arcsin(s13)
    theta12 = jnp.arctan2(jnp.abs(U[0, 1]), jnp.abs(U[0, 0]) + 1e-30)
    theta23 = jnp.arctan2(jnp.abs(U[1, 2]), jnp.abs(U[2, 2]) + 1e-30)
    return theta12, theta13, theta23

# -----------------------
# Build light neutrino mass, using eig-inversion for MR
# -----------------------
@jax.jit
def light_neutrino_mass_raw(Y, a_vec, M0, eps):
    """
    Return: mnu_sym (GeV), eig_diag of MR, n_clipped, min_abs_eig, max_abs_eig
    """
    Ynu = Yukawa(Y, eps)
    MR_dimless = build_MR_from_a_vec_io(a_vec, eps)   # dimensionless (prefactor M0 outside)
    # MR (dimensionful) = M0 * MR_dimless but we can invert MR_dimless and multiply prefactor appropriately.
    # Compute eig-inverse on MR_dimless (safer numerics, dimensionless units)
    MRinv_dimless, eigvals, n_clipped, min_abs_eig, max_abs_eig = symmetric_eiginv_clipped(MR_dimless, TINY_ABS_EIG)

    # mnu_dimless = Y^T MRinv Y
    mnu_dimless = Ynu.T @ MRinv_dimless @ Ynu
    prefactor = (v ** 2) / (2.0 * M0)
    mnu = prefactor * mnu_dimless
    mnu_sym = 0.5 * (mnu + mnu.T)
    return mnu_sym, eigvals, n_clipped, min_abs_eig, max_abs_eig, MR_dimless

# -----------------------
# Observables (specialised IO PDG reorder [1,2,0])
# -----------------------
@jax.jit
def observables(Y, a_vec, M0, eps):
    mnu, eigvals, n_clipped, min_abs_eig, max_abs_eig, MRdim = light_neutrino_mass_raw(Y, a_vec, M0, eps)
    w, V = jnp.linalg.eigh(mnu)
    idx = jnp.argsort(w); w_sorted = w[idx]; V_sorted = V[:, idx]
    # PDG IO permute: idx_inverted = [1,2,0] 
    m1 = w_sorted[1]; m2 = w_sorted[2]; m3 = w_sorted[0]
    U_reordered = jnp.column_stack((V_sorted[:,1], V_sorted[:,2], V_sorted[:,0]))
    U_can = canonicalize_real_U(U_reordered)
    dm21_GeV2 = m2**2 - m1**2
    dm31_GeV2 = m3**2 - m1**2
    dm32_GeV2 = m3**2 - m2**2
    theta12, theta13, theta23 = extract_angles_from_U(U_can)
    # diagnostics: min_abs_eig etc. also returned
    return dm21_GeV2, dm31_GeV2, dm32_GeV2, theta12, theta13, theta23, U_can, eigvals, n_clipped, min_abs_eig, max_abs_eig, MRdim

@jax.jit
def normal_logpdf_scalar(x, mu, sigma):
    sigma = jnp.maximum(sigma, 1e-16)
    z = (x - mu) / sigma
    return -0.5 * z * z - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))

# -----------------------
# Probabilistic model (IO-only)
# -----------------------
def model(*,
          eps_min=1e-1, eps_max=0.5,
          dm21_obs=7.50e-5, dm21_sigma=0.18e-5,
          dm31_obs=2.50e-3, dm31_sigma=0.04e-3,
          theta12_obs=None, theta12_sigma=None,
          theta13_obs=None, theta13_sigma=None,
          theta23_obs=None, theta23_sigma=None):

    # PDG angles -> angle + propagated sigma if not provided
    if theta12_obs is None or theta12_sigma is None:
        s12, ds12 = 0.307, 0.012
        theta12_obs = jnp.arcsin(jnp.sqrt(s12))
        theta12_sigma = ds12 * (1.0 / (2.0 * jnp.sqrt(s12 * (1.0 - s12))))
    if theta13_obs is None or theta13_sigma is None:
        s13, ds13 = 0.0218, 0.0007
        theta13_obs = jnp.arcsin(jnp.sqrt(s13))
        theta13_sigma = ds13 * (1.0 / (2.0 * jnp.sqrt(s13 * (1.0 - s13))))
    if theta23_obs is None or theta23_sigma is None:
        s23, ds23 = 0.537, 0.020
        theta23_obs = jnp.arcsin(jnp.sqrt(s23))
        theta23_sigma = ds23 * (1.0 / (2.0 * jnp.sqrt(s23 * (1.0 - s23))))

    # -----------------------
    # Priors
    # -----------------------
    # Yukawa priors (hierarchical)
    sigma_Y = numpyro.sample('sigma_Y', dist.HalfCauchy(0.5))
    Y_raw = numpyro.sample('Y', dist.Normal(0.0, sigma_Y).expand([3,3]).to_event(2))

    # Majorana 'a' O(1) priors: sample a_vec = [a12,a22,a23,a33]
    sigma_a = numpyro.sample('sigma_a', dist.HalfCauchy(0.8))
    a12 = numpyro.sample('a12', dist.Normal(0.0, sigma_a))    # can be negative
    a22 = numpyro.sample('a22', dist.HalfNormal(sigma_a))     # diagonal-ish positive
    a23 = numpyro.sample('a23', dist.Normal(0.0, sigma_a))
    a33 = numpyro.sample('a33', dist.HalfNormal(sigma_a))
    a_vec = jnp.array([a12, a22, a23, a33], dtype=jnp.float64)

    # epsilon (FN) and seesaw scale M0 (log-uniform)
    eps = numpyro.sample('epsilon', dist.Uniform(eps_min, eps_max))
    log10_M0 = numpyro.sample('log10_M0', dist.Uniform(14.0, 16.0))
    M0 = 10.0 ** log10_M0
    numpyro.deterministic('M0', M0)

    # -----------------------
    # Compute observables with eig-inversion
    # -----------------------
    (dm21_GeV2, dm31_GeV2, dm32_GeV2,
     theta12_th, theta13_th, theta23_th,
     U_can, eigvals_MR, n_clipped, min_abs_eig, max_abs_eig, MRdim) = observables(Y_raw, a_vec, M0, eps)

    # convert to eV^2; for IO we compare with abs(dm31)
    dm21 = dm21_GeV2 * GEV2_TO_EV2
    dm31_signed = dm31_GeV2 * GEV2_TO_EV2
    dm31_for_lik = jnp.abs(dm31_signed)

    # Deterministics
    numpyro.deterministic('dm21', dm21)
    numpyro.deterministic('dm31', dm31_for_lik)
    numpyro.deterministic('theta12_th', theta12_th)
    numpyro.deterministic('theta13_th', theta13_th)
    numpyro.deterministic('theta23_th', theta23_th)
    numpyro.deterministic('mr_eigvals', eigvals_MR)
    numpyro.deterministic('mr_n_clipped', n_clipped)
    numpyro.deterministic('mr_min_abs_eig', min_abs_eig)
    numpyro.deterministic('mr_max_abs_eig', max_abs_eig)
    numpyro.deterministic('mr_dimless', MRdim)

    # a_vec components are already recorded as sample sites (a12,a22,a23,a33).
    # Do not register deterministic sites with the same names (NumPyro requires unique site names).
    # If you want the compact vector as a single deterministic, uncomment below:
    # numpyro.deterministic('a_vec', a_vec)

    # -----------------------
    # Likelihoods (same as NO blueprint)
    # -----------------------
    numpyro.factor('logp_dm21', normal_logpdf_scalar(dm21, dm21_obs, dm21_sigma))
    numpyro.factor('logp_dm31', normal_logpdf_scalar(dm31_for_lik, dm31_obs, dm31_sigma))
    numpyro.factor('logp_theta12', normal_logpdf_scalar(theta12_th, theta12_obs, theta12_sigma))
    numpyro.factor('logp_theta13', normal_logpdf_scalar(theta13_th, theta13_obs, theta13_sigma))
    numpyro.factor('logp_theta23', normal_logpdf_scalar(theta23_th, theta23_obs, theta23_sigma))

# -----------------------
# Plotting & post-processing
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
    mr_min_eig_vals = posterior['mr_min_abs_eig'].values.reshape(-1)
    mr_n_clipped = posterior['mr_n_clipped'].values.reshape(-1)

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    styles = {'modern': {'hist_color': '#1f77b4', 'alpha': 0.85, 'gauss_color': '#222222', 'exp_color': '#d62728', 'exp_alpha': 0.18, 'box': dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor='#333333'), 'mass_colors': ['#1f77b4', '#17becf'], 'font': {'family':'sans-serif','size':10}}}
    cfg = styles.get(style, styles['modern'])

    # angle panel (mirrors your NO plots)
    def plot_angles_with_overlay_panel(idata, style='modern'):
        posterior = idata.posterior
        theta12_vals = np.degrees(posterior['theta12_th'].values.reshape(-1))
        theta13_vals = np.degrees(posterior['theta13_th'].values.reshape(-1))
        theta23_vals = np.degrees(posterior['theta23_th'].values.reshape(-1))
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
        plt.rcParams.update({'font.family': cfg['font']['family'], 'font.size': cfg['font']['size']})
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
        for ax, vals, exp, sig, name in zip(
            axes,
            [theta12_vals, theta13_vals, theta23_vals],
            [theta12_exp, theta13_exp, theta23_exp],
            [theta12_sig, theta13_sig, theta23_sig],
            ['θ12', 'θ13', 'θ23']
        ):
            if np.nanstd(vals) > 0:
                ax.hist(vals, bins=80, density=True, alpha=cfg['alpha'], color=cfg['hist_color'], label='Posterior', zorder=3)
            else:
                ax.axvline(float(np.nanmedian(vals)), color=cfg['hist_color'], linewidth=2, alpha=cfg['alpha'])
            mu = np.mean(vals); std = np.std(vals)
            x = np.linspace(np.percentile(vals,0.5), np.percentile(vals,99.5), 400)
            if std>0:
                pdf = 1.0/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/std)**2)
                ax.plot(x, pdf, linestyle='--', color=cfg['gauss_color'], label='Gaussian fit')
            ax.axvspan(exp-sig, exp+sig, color=cfg['exp_color'], alpha=cfg['exp_alpha'], label='Exp ±1σ', zorder=0)
            ax.axvline(exp, color=cfg['exp_color'], linestyle='-', label='Exp central', zorder=5)
            ax.set_xlabel(f'{name} (deg)'); ax.set_title(f'Posterior of {name}')
            caption = f"Exp={exp:.2f}° ±{sig:.2f}°  |  Post={mu:.2f}° ±{std:.2f}°"
            ax.text(0.5, -0.22, caption, ha='center', va='top', transform=ax.transAxes, fontsize=cfg['font']['size']-1, bbox=cfg['box'])
        posterior_patch = plt.Rectangle((0,0),1,1, facecolor=cfg['hist_color'], alpha=cfg['alpha'])
        gauss_line = Line2D([0],[0], color=cfg['gauss_color'], linestyle='--')
        exp_line = Line2D([0],[0], color=cfg['exp_color'], linestyle='-')
        exp_band = plt.Rectangle((0,0),1,1, facecolor=cfg['exp_color'], alpha=cfg['exp_alpha'])
        fig.legend([posterior_patch, gauss_line, exp_line, exp_band], ['Posterior (histogram)', 'Gaussian fit', 'Experiment central', 'Experiment ±1σ'], loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False, fontsize=cfg['font']['size'])
        fig.subplots_adjust(top=0.82, bottom=0.20)
        plt.tight_layout()
        outname = f'{outdir}/plots/angles_panel_{style}.png'
        plt.savefig(outname, dpi=300, bbox_inches='tight')
        if DISPLAY_PLOTS: plt.show()
        plt.close()

    plot_angles_with_overlay_panel(idata)

    # dm21 & dm31 panels (match NO style)
    dm21_obs = 7.50e-5; dm21_sigma = 0.18e-5
    dm31_obs = 2.50e-3; dm31_sigma = 0.04e-3
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    col1, col2 = cfg.get('mass_colors', ['#1f77b4','#17becf'])
    if np.nanstd(dm21_vals) > 0:
        ax1.hist(dm21_vals, bins=80, density=True, alpha=cfg['alpha'], color=col1, zorder=3)
    else:
        ax1.axvline(float(np.nanmedian(dm21_vals)), color=col1, linewidth=2, alpha=cfg['alpha'])
    mu21 = np.mean(dm21_vals); std21 = np.std(dm21_vals)
    x21 = np.linspace(np.percentile(dm21_vals,0.5), np.percentile(dm21_vals,99.5), 400)
    if std21>0:
        pdf21 = 1.0/(std21*np.sqrt(2*np.pi)) * np.exp(-0.5*((x21-mu21)/std21)**2)
        ax1.plot(x21, pdf21, linestyle='--', color=cfg['gauss_color'], label=f'Gaussian fit μ={mu21:.2e}, σ={std21:.2e}', zorder=4)
    ax1.axvspan(dm21_obs-dm21_sigma, dm21_obs+dm21_sigma, color=cfg['exp_color'], alpha=cfg['exp_alpha'], zorder=0)
    ax1.axvline(dm21_obs, color=cfg['exp_color'], linestyle='-', zorder=5)
    ax1.set_xlabel(r'$\Delta m^2_{21}$ (eV$^2$)'); ax1.set_title('Posterior of Δm²_21')
    caption1 = f"Exp={dm21_obs:.2e} ±{dm21_sigma:.2e}  |  Post={mu21:.2e} ±{std21:.2e}"
    ax1.text(0.5, -0.25, caption1, ha='center', va='top', transform=ax1.transAxes, fontsize=cfg['font']['size']-1, bbox=cfg['box'])
    ax1.legend()

    ax2 = plt.subplot(1,2,2)
    if np.nanstd(dm31_vals) > 0:
        ax2.hist(dm31_vals, bins=80, density=True, alpha=cfg['alpha'], color=col2, zorder=3)
    else:
        ax2.axvline(float(np.nanmedian(dm31_vals)), color=col2, linewidth=2, alpha=cfg['alpha'])
    mu31 = np.mean(dm31_vals); std31 = np.std(dm31_vals)
    x31 = np.linspace(np.percentile(dm31_vals,0.5), np.percentile(dm31_vals,99.5), 400)
    if std31>0:
        pdf31 = 1.0/(std31*np.sqrt(2*np.pi)) * np.exp(-0.5*((x31-mu31)/std31)**2)
        ax2.plot(x31, pdf31, linestyle='--', color=cfg['gauss_color'], label=f'Gaussian fit μ={mu31:.2e}, σ={std31:.2e}', zorder=4)
    ax2.axvspan(dm31_obs-dm31_sigma, dm31_obs+dm31_sigma, color=cfg['exp_color'], alpha=cfg['exp_alpha'], zorder=0)
    ax2.axvline(dm31_obs, color=cfg['exp_color'], linestyle='-', zorder=5)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x*1e3:.2f}"))
    ax2.set_xlabel(r'$\Delta m^2_{31}$ (10$^{-3}$ eV$^2$)'); ax2.set_title('Posterior of Δm²_31')
    caption2 = f"Exp={dm31_obs*1e3:.3f} ±{dm31_sigma*1e3:.3f}  |  Post={mu31*1e3:.3f} ±{std31*1e3:.3f} (×10^-3)"
    ax2.text(0.5, -0.25, caption2, ha='center', va='top', transform=ax2.transAxes, fontsize=cfg['font']['size']-1, bbox=cfg['box'])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{outdir}/plots/dm_splittings_with_exp.png', dpi=300)
    if DISPLAY_PLOTS: plt.show()
    plt.close()

# -----------------------
# Runner
# -----------------------
def run_and_save(num_chains=NUM_CHAINS, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, seed=SEED):
    print("jax devices:", jax.device_count(), "local_device_count:", jax.local_device_count())
    print(f"Running MCMC: chains={num_chains}, warmup={num_warmup}, samples={num_samples}, target_accept={TARGET_ACCEPT} (IO eig-inv)")

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

    az.to_netcdf(idata, f"{outdir}/posterior_idata.nc")
    df = az.summary(idata, kind="all", round_to=4)
    df.to_csv(f"{outdir}/summary_with_rhat_ess.csv")

    posterior = idata.posterior
    dm21_vals = posterior['dm21'].values.reshape(-1)
    dm31_vals = posterior['dm31'].values.reshape(-1)
    eps_vals = posterior['epsilon'].values.reshape(-1)
    mr_min_abs = posterior['mr_min_abs_eig'].values.reshape(-1) if 'mr_min_abs_eig' in posterior else None
    mr_n_clipped = posterior['mr_n_clipped'].values.reshape(-1) if 'mr_n_clipped' in posterior else None

    summary_df = pd.DataFrame({
        'dm21_eV2': dm21_vals,
        'dm31_eV2': dm31_vals,
        'epsilon': eps_vals,
        'mr_min_abs_eig': (mr_min_abs if mr_min_abs is not None else np.nan),
        'mr_n_clipped': (mr_n_clipped if mr_n_clipped is not None else np.nan)
    })
    summary_df.to_csv(f'{outdir}/dm_summary_samples.csv', index=False)

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
    pvals_df.to_csv('resultsIO_eig/pvalues_summary.csv', index=False)
    print("Saved p-values to resultsIO_eig/pvalues_summary.csv")

    print("\n--- Summary of means & std devs ---")
    print('θ12 mean,std (deg):', np.degrees(mean_t12), np.degrees(np.std(theta12_post)))
    print('θ13 mean,std (deg):', np.degrees(mean_t13), np.degrees(np.std(theta13_post)))
    print('θ23 mean,std (deg):', np.degrees(mean_t23), np.degrees(np.std(theta23_post)))

    try:
        plot_posterior_publication(idata)
    except Exception as e:
        print('Plotting failed:', e)

    return mcmc, idata

# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    mcmc, idata = run_and_save()
