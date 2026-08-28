"""
Inverted Hierarchy: Type-I seesaw with FN U(1)_X symmetry.

Majorana:   a12, a22, a23, a33  (no texture zeros)
NDIM = 15: log10_M0, epsilon, 9 Yukawa, 4 Majorana

Parameter layout (16 total):
  x[0]     = log10(M0)          in [13, 17]
  x[1]     = epsilon            in [0.1, 0.5]
  x[2:10]  = y11,y12,y13,y21,y23,y31,y32,y33   (8 Yukawa, y22=0)
  x[10:14] = a12,a22,a23,a33            (6 Majorana, no zeros)

NuFIT 6.0 NH with SK atmospheric data:
  dm21 = 7.49e-5  ± 0.19e-5  eV^2
  dm31 = 2.513e-3 ± 0.021e-3 eV^2
  s12  = 0.308    ± 0.0115
  s13  = 0.02215  ± 0.00057
  s23  = 0.470    ± 0.015    (lower octant NH best fit)

Naturalness: r = max|y_ij| / min|y_ij| over x[2:10] (8 Yukawa only).
             y22=0 is excluded by construction.

"""

import numpy as np
import os, json, time, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG — edit here only
# =============================================================================

# Script must live in the same folder as Run_00/, Run_01/, Run_02/
BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "IH_UltraNest")
N_RUNS = 3

# UltraNest live points.
MIN_LIVE     = 50_000

CLUSTER_LIVE = 300

DLOGZ        = 0.2
NSTEPS       = 50   # pass to SliceSampler

# =============================================================================
# PHYSICS
# =============================================================================
V_EW   = 246.22   # GeV (electroweak VEV)
GEV2EV = 1e9

# NuFIT 5.2 IH central values
DATA_IH = {
    "dm21": (7.49e-5,   0.19e-5),    # eV^2,  dm21 = m2^2 - m1^2 > 0
    "dm31": (-2.484e-3, 0.020e-3),   # eV^2,  dm32 = m3^2 - m2^2 < 0  (IH)
    "s12":  (0.308,     0.0115),     # sin^2(theta_12)
    "s13":  (0.02225,   0.0006),     # sin^2(theta_13)
    "s23":  (0.550,     0.0135),     # sin^2(theta_23)
}

# Parameter layout (NDIM=15):
#   x[0]     = log10(M0)          in [13, 17]
#   x[1]     = epsilon            in [0.1, 0.5]
#   x[2:11]  = y_ij  9 Yukawa     in [-3, 3]  (O(1) coefficients)
#   x[11:15] = a_kl  4 Majorana   in [-3, 3]  (a11=a13=0 by IH texture)

NDIM = 15
PARAM_NAMES = [
    'log10_M0', 'epsilon',
    'y11', 'y12', 'y13',
    'y21', 'y22', 'y23',
    'y31', 'y32', 'y33',
    'a12', 'a22', 'a23', 'a33',
]


def loglike(x):
    M0 = 10**x[0]
    e  = x[1]
    y11, y12, y13, y21, y22, y23, y31, y32, y33 = x[2:11]
    a12, a22, a23, a33 = x[11:15]

    Y = np.array([
        [y11,    e*y12,     e*y13    ],
        [e*y21,  y22,       e**2*y23 ],
        [e*y31,  e**2*y32,  y33      ],
    ], dtype=complex)

    MR = M0 * np.array([
        [0,          e**3*a12,  0        ],
        [e**3*a12,   e**4*a22,  e**2*a23 ],
        [0,          e**2*a23,  a33      ],
    ], dtype=complex)

    if abs(np.linalg.det(MR)) < 1e-290:
        return -1e300

    try:
        MRi = np.linalg.inv(MR)
    except np.linalg.LinAlgError:
        return -1e300

    mn = -0.5 * V_EW**2 * (Y @ MRi @ Y.T) * GEV2EV
    mn = 0.5 * (mn + mn.T)

    H = mn.conj().T @ mn
    try:
        msq, U = np.linalg.eigh(0.5*(H + H.conj().T))
    except np.linalg.LinAlgError:
        return -1e300

    if np.any(msq < 0):
        return -1e300

    u3 = U[:, 0]
    u2 = U[:, 2]
    s13   = float(abs(u3[0])**2)
    denom = max(1.0 - s13, 1e-12)
    s12   = float(np.clip(abs(u2[0])**2 / denom, 0.0, 1.0))
    s23   = float(np.clip(abs(u3[1])**2 / denom, 0.0, 1.0))
    dm21  = float(msq[2] - msq[1])
    dm32  = float(msq[0] - msq[2])

    chi2 = (
        ((dm21 - DATA_IH['dm21'][0]) / DATA_IH['dm21'][1])**2 +
        ((dm32 - DATA_IH['dm31'][0]) / DATA_IH['dm31'][1])**2 +
        ((s12  - DATA_IH['s12'][0])  / DATA_IH['s12'][1] )**2 +
        ((s13  - DATA_IH['s13'][0])  / DATA_IH['s13'][1] )**2 +
        ((s23  - DATA_IH['s23'][0])  / DATA_IH['s23'][1] )**2
    )
    return -0.5 * chi2

def ptform(u):
    """
    Maps u ~ Uniform[0,1]^15 to physical parameter space.
    Full prior — no restrictions, no octant splitting.
    UltraNest explores everything including all sign combinations.
    """
    x       = np.empty(NDIM)
    x[0]    = 13.0 + 4.0 * u[0]    # log10_M0 in [13, 17]
    x[1]    = 0.1  + 0.4 * u[1]    # epsilon   in [0.1, 0.5]
    x[2:15] = -3.0 + 6.0 * u[2:15] # y_ij, a_kl in [-3, 3]
    return x


def naturalness(samples):
    """r = max|y_ij| / min|y_ij| over x[2:11] (9 Yukawa coefficients only)."""
    y  = np.abs(samples[:, 2:11])
    ym = np.where(y > 1e-10, y, np.inf)
    return np.where(
        np.isfinite(ym.min(1)) & (ym.min(1) > 0),
        y.max(1) / ym.min(1),
        np.inf,
    )


# =============================================================================
# SINGLE RUN
# =============================================================================
def run_single(run_i):
    """
    Run one UltraNest ReactiveNestedSampler on the full IH prior.
    Resumes automatically if log_dir already exists from a previous attempt.
    Skips if metadata.json already complete.
    """
    run_dir   = os.path.join(OUTDIR, f"Run_{run_i:02d}")
    log_dir   = os.path.join(run_dir, "log")
    meta_path = os.path.join(run_dir, "metadata.json")
    os.makedirs(run_dir, exist_ok=True)

    # Skip if already complete
    if os.path.exists(meta_path):
        m = json.load(open(meta_path))
        print(f"  Run {run_i}: already done — "
              f"logZ={m['logZ']:.4f}±{m['logZerr']:.4f}  "
              f"P(r<30)={m['p30']:.2f}%")
        return m

    try:
        import ultranest
        import ultranest.stepsampler as ss
    except ImportError:
        print("ERROR: ultranest not installed.")
        print("Fix: pip install --user --force-reinstall --no-binary ultranest ultranest h5py")
        return None
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            print("ERROR: numpy ABI mismatch.")
            print("Fix: pip install --user --force-reinstall --no-binary ultranest ultranest")
            print(f"  (full error: {e})")
            return None
        raise

    print(f"\n{'='*65}")
    print(f"  RUN {run_i}  UltraNest  min_live={MIN_LIVE:,}  dlogz={DLOGZ}")
    print(f"  cluster_min={CLUSTER_LIVE}  nsteps={2*NDIM} per new point")
    print(f"  log_dir = {log_dir}")
    print(f"  (resumes from checkpoint if interrupted)")
    print(f"{'='*65}", flush=True)

    t0 = time.time()

    # ReactiveNestedSampler:
    #   - MLFriends region: union of balls around live points (not ellipsoids)
    #   - Cluster tracking: separate compression chains per detected mode
    #   - resume='resume': reads checkpoint from log_dir if it exists
    sampler = ultranest.ReactiveNestedSampler(
        PARAM_NAMES,
        loglike,
        ptform,
        log_dir    = log_dir,
        resume     = 'resume',
        vectorized = False,       # call loglike one point at a time
    )

    # SliceSampler: takes nsteps decorrelating slice steps per new live point.
    # nsteps = 2*NDIM = 30 is the standard recommendation.
    # generate_mixture_random_direction: mixes axis-aligned and random directions
    # for better coverage of elongated posteriors.
    sampler.stepsampler = ss.SliceSampler(
        nsteps             = NSTEPS,
        generate_direction = ss.generate_region_random_direction,
    )

    # run() blocks until convergence. UltraNest prints progress to stdout
    # including cluster detection ("Have N modes") as it runs.
    result = sampler.run(
    min_num_live_points     = MIN_LIVE,
    cluster_num_live_points = CLUSTER_LIVE,
    dlogz                   = DLOGZ,
    min_ess                 = 20_000,
    max_ncalls              = 2_000_000_000,
    show_status             = True,   # print progress including cluster count
    )

    # Extract results
    logZ    = float(result['logz'])
    logZerr = float(result['logzerr'])
    ncall   = int(result.get('ncall', 0))
    niter   = int(result.get('niter', 0))
    ess     = float(result.get('ess', 0))

    # Equal-weight posterior samples
    # result['samples'] is shape (N, 15) — already equal-weight
    eq_s = np.array(result['samples'], dtype=np.float64)
    r    = naturalness(eq_s)
    p30  = float(np.mean(r < 30) * 100)

    # P_spike: fraction of posterior with all |y_ij| > 0.5
    y_min_arr  = np.min(np.where(np.abs(eq_s[:,2:11])>1e-10,
                                  np.abs(eq_s[:,2:11]), np.inf), axis=1)
    spike_frac = float(np.mean(y_min_arr > 0.5) * 100)

    elapsed = (time.time() - t0) / 3600

    print(f"\n  RUN {run_i} COMPLETE")
    print(f"  logZ         = {logZ:.4f}  ±  {logZerr:.4f}")
    print(f"  P(r<30)      = {p30:.2f}%")
    print(f"  spike_frac   = {spike_frac:.1f}%  (Y_min > 0.5; expected ~12.5%)")
    print(f"  ncall        = {ncall:,}")
    print(f"  niter        = {niter:,}")
    print(f"  ESS          = {ess:.0f}")
    print(f"  Time         = {elapsed:.2f}h", flush=True)

    # Save
    np.savez_compressed(
        os.path.join(run_dir, "results.npz"),
        eq_samples = eq_s.astype(np.float32),
        logZ       = np.float64(logZ),
        logZerr    = np.float64(logZerr),
        p30        = np.float64(p30),
        spike_frac = np.float64(spike_frac),
        ncall      = np.int64(ncall),
        niter      = np.int64(niter),
        ess        = np.float64(ess),
    )
    meta = dict(
        run_i    = run_i,
        logZ     = logZ,
        logZerr  = logZerr,
        p30      = p30,
        spike_frac = spike_frac,
        ncall    = ncall,
        niter    = niter,
        ess      = ess,
        elapsed  = elapsed,
        min_live = MIN_LIVE,
        dlogz    = DLOGZ,
        sampler  = 'ultranest',
    )
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


# =============================================================================
# CONVERGENCE CHECK
# =============================================================================
def convergence_check(results):
    logZs      = np.array([m['logZ'] for m in results])
    p30s       = np.array([m['p30']  for m in results])
    spike_fracs = np.array([m['spike_frac'] for m in results])
    logZerrs   = np.array([m['logZerr'] for m in results])

    print(f"\n{'='*65}")
    print(f"CONVERGENCE SUMMARY ({len(results)} runs)")
    print(f"{'='*65}")
    print(f"\n  {'Run':>4} {'logZ':>10} {'logZerr':>9} {'P(r<30)':>9} "
          f"{'spike%':>8} {'ncall':>12}")
    for m in results:
        print(f"  {m['run_i']:>4} {m['logZ']:>10.4f} {m['logZerr']:>9.4f} "
              f"{m['p30']:>8.2f}% {m['spike_frac']:>7.1f}% "
              f"{m['ncall']:>12,}")

    sd_logZ  = float(np.std(logZs, ddof=1)) if len(results) > 1 else 0
    mean_logZ = float(np.mean(logZs))
    sd_p30    = float(np.std(p30s,  ddof=1)) if len(results) > 1 else 0

    H_est     = 35.0   # from dynesty N_dead/N_live history
    skilling  = np.sqrt(H_est / MIN_LIVE)
    ratio     = sd_logZ / skilling if skilling > 0 else 0

    print(f"\n  logZ:    mean={mean_logZ:.4f}  SD={sd_logZ:.4f}")
    print(f"  P(r<30): mean={np.mean(p30s):.2f}%   SD={sd_p30:.2f}%")
    print(f"  spike%:  mean={np.mean(spike_fracs):.1f}%   SD={np.std(spike_fracs,ddof=1) if len(results)>1 else 0:.1f}%")
    print(f"\n  Skilling sigma (H≈{H_est}, N={MIN_LIVE:,}): {skilling:.4f} nats")
    print(f"  Observed SD(logZ):                     {sd_logZ:.4f} nats")
    print(f"  Excess factor:                         {ratio:.1f}x")

    # Sanity check: spike fraction should be ~12.5% (from Run 2 IS estimate)
    # and all runs should agree to within ~5%
    print(f"\n  SANITY CHECKS:")
    print(f"  spike_frac mean = {np.mean(spike_fracs):.1f}%  "
          f"(from Run 2 dynesty IS estimate: ~12.5%)")
    if np.mean(spike_fracs) < 1.0:
        print(f"  *** WARNING: spike_frac << 12.5% — UltraNest may still be")
        print(f"      missing the spike. Increase MIN_LIVE to 50,000.")
    elif np.mean(spike_fracs) > 25.0:
        print(f"  *** WARNING: spike_frac > 25% — possible spike oversampling.")

    print(f"\n  CONVERGENCE VERDICT:")
    if ratio < 3:
        print(f"  ✓ CONVERGED — SD(logZ) = {sd_logZ:.3f} < 3x Skilling = "
              f"{3*skilling:.3f}")
        print(f"  Final result: P(r<30) = {np.mean(p30s):.2f}% ± {sd_p30:.2f}%")
    elif ratio < 10:
        print(f"  ~ MARGINAL — SD(logZ) = {sd_logZ:.3f} = {ratio:.1f}x Skilling")
        print(f"  Re-run with MIN_LIVE = {MIN_LIVE*3:,} for better convergence.")
    else:
        print(f"  ✗ NOT CONVERGED — SD(logZ) = {sd_logZ:.3f} = {ratio:.1f}x Skilling")
        print(f"  Mode-switching is still occurring within UltraNest.")
        print(f"  Options:")
        print(f"    1. Increase MIN_LIVE to 100,000")
        print(f"    2. Increase CLUSTER_LIVE to 500")
        print(f"    3. The posterior may have too many modes for nested sampling")

    # Save summary
    out = os.path.join(OUTDIR, "summary.json")
    with open(out, 'w') as f:
        json.dump({
            'n_runs':       len(results),
            'min_live':     MIN_LIVE,
            'logZ_mean':    mean_logZ,
            'logZ_SD':      sd_logZ,
            'p30_mean':     float(np.mean(p30s)),
            'p30_SD':       sd_p30,
            'spike_frac_mean': float(np.mean(spike_fracs)),
            'skilling_sigma': skilling,
            'excess_factor': ratio,
            'converged':    bool(ratio < 3),
        }, f, indent=2)
    print(f"\n  Saved → {out}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\n{'#'*65}")
    print(f"  IH NESTED SAMPLING — UltraNest")
    print(f"  Full prior. No octant splitting.")
    print(f"  {N_RUNS} runs × min_live={MIN_LIVE:,}  dlogz={DLOGZ}")
    print(f"  Output → {OUTDIR}/")
    print(f"{'#'*65}\n")

    results = []
    for i in range(N_RUNS):
        m = run_single(i)
        if m is not None:
            results.append(m)

    if len(results) >= 2:
        convergence_check(results)
    elif len(results) == 1:
        print(f"\n  Only 1 run complete. Run again to get convergence statistics.")
    else:
        print(f"\n  No runs completed successfully.")


if __name__ == '__main__':
    main()