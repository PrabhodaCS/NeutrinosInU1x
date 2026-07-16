"""
mcmc_NH_ultranest.py  —  NH Nested Sampling with UltraNest
===========================================================
Normal Hierarchy: Type-I seesaw with FN U(1)_X symmetry.

NH texture: y22 = 0  (one Yukawa texture zero)
Majorana:   a11, a12, a13, a22, a23, a33  (no texture zeros)
NDIM = 16: log10_M0, epsilon, 8 Yukawa, 6 Majorana

Parameter layout (16 total):
  x[0]     = log10(M0)          in [13, 17]
  x[1]     = epsilon            in [0.1, 0.5]
  x[2:10]  = y11,y12,y13,y21,y23,y31,y32,y33   (8 Yukawa, y22=0)
  x[10:16] = a11,a12,a13,a22,a23,a33            (6 Majorana, no zeros)

NuFIT 6.0 NH with SK atmospheric data:
  dm21 = 7.49e-5  ± 0.19e-5  eV^2
  dm31 = 2.513e-3 ± 0.021e-3 eV^2
  s12  = 0.308    ± 0.0115
  s13  = 0.02215  ± 0.00057
  s23  = 0.470    ± 0.015    (lower octant NH best fit)

Naturalness: r = max|y_ij| / min|y_ij| over x[2:10] (8 Yukawa only).
             y22=0 is excluded by construction.

Run: python3 mcmc_NH_ultranest.py
"""

import numpy as np
import os, json, time, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG — edit here only
# =============================================================================
BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "NH_UltraNest")
N_RUNS = 3

MIN_LIVE     = 50_000   # live points — matches IH for methodological consistency
CLUSTER_LIVE = 300      # min live pts per detected cluster
DLOGZ        = 0.2      # convergence criterion
NSTEPS       = 50       # slice sampler steps per new live point

# =============================================================================
# PHYSICS — NuFIT 6.0 NH with SK atmospheric data
# =============================================================================
V_EW   = 246.22   # GeV
GEV2EV = 1e9

DATA_NH = {
    "dm21": (7.49e-5,   0.19e-5),    # eV^2  dm21 = m2^2 - m1^2 > 0
    "dm31": (2.513e-3,  0.021e-3),   # eV^2  dm31 = m3^2 - m1^2 > 0 (NH)
    "s12":  (0.308,     0.0115),
    "s13":  (0.02215,   0.00057),
    "s23":  (0.470,     0.015),      # lower octant NH best fit
}

NDIM = 16
PARAM_NAMES = [
    'log10_M0', 'epsilon',
    'y11', 'y12', 'y13',
    'y21', 'y23', 'y31', 'y32', 'y33',   # y22=0 excluded
    'a11', 'a12', 'a13', 'a22', 'a23', 'a33',
]


def loglike(x):
    """
    NH Type-I seesaw loglikelihood.

    Y matrix (y22 = 0 texture zero):
        [[y11,    e*y12,     e*y13  ],
         [e*y21,  0,         e^2*y23],
         [e*y31,  e^2*y32,   y33    ]]

    MR matrix (no texture zeros for NH):
        M0 * [[a11,        e^3*a12,  e*a13   ],
              [e^3*a12,    e^4*a22,  e^2*a23 ],
              [e*a13,      e^2*a23,  a33     ]]

    NH eigenvalue ordering from np.linalg.eigh (ascending m^2):
        [m1, m2, m3]
    PMNS columns: U[:,1] = nu_2, U[:,2] = nu_3
      s13 = |U[0,2]|^2
      s12 = |U[0,1]|^2 / (1 - s13)
      s23 = |U[1,2]|^2 / (1 - s13)
      dm21 = msq[1] - msq[0] > 0
      dm31 = msq[2] - msq[0] > 0

    Returns -0.5 * chi2, or -1e300 on numerical failure.
    """
    M0 = 10**x[0]
    e  = x[1]
    y11, y12, y13, y21, y23, y31, y32, y33 = x[2:10]
    a11, a12, a13, a22, a23, a33 = x[10:16]

    Y = np.array([
        [y11,    e*y12,     e*y13    ],
        [e*y21,  0.0,       e**2*y23 ],
        [e*y31,  e**2*y32,  y33      ],
    ], dtype=complex)

    MR = M0 * np.array([
        [a11,        e**3*a12,  e*a13    ],
        [e**3*a12,   e**4*a22,  e**2*a23 ],
        [e*a13,      e**2*a23,  a33      ],
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
        msq, U = np.linalg.eigh(0.5 * (H + H.conj().T))
    except np.linalg.LinAlgError:
        return -1e300

    if np.any(msq < 0):
        return -1e300

    # NH: eigh returns [m1^2, m2^2, m3^2] ascending
    u2 = U[:, 1]   # nu_2 column (solar)
    u3 = U[:, 2]   # nu_3 column (atmospheric)

    s13   = float(abs(u3[0])**2)
    denom = max(1.0 - s13, 1e-12)
    s12   = float(np.clip(abs(u2[0])**2 / denom, 0.0, 1.0))
    s23   = float(np.clip(abs(u3[1])**2 / denom, 0.0, 1.0))
    dm21  = float(msq[1] - msq[0])   # > 0 for NH
    dm31  = float(msq[2] - msq[0])   # > 0 for NH

    chi2 = (
        ((dm21 - DATA_NH['dm21'][0]) / DATA_NH['dm21'][1])**2 +
        ((dm31 - DATA_NH['dm31'][0]) / DATA_NH['dm31'][1])**2 +
        ((s12  - DATA_NH['s12'][0])  / DATA_NH['s12'][1] )**2 +
        ((s13  - DATA_NH['s13'][0])  / DATA_NH['s13'][1] )**2 +
        ((s23  - DATA_NH['s23'][0])  / DATA_NH['s23'][1] )**2
    )
    return -0.5 * chi2


def ptform(u):
    """
    Maps u ~ Uniform[0,1]^16 to physical parameter space.
    All parameters uniform — no restrictions.
    """
    u = np.clip(u, 0.0, 1.0)   # defensive clip against slice sampler overstepping
    x       = np.empty(NDIM)
    x[0]    = 13.0 + 4.0 * u[0]    # log10_M0 in [13, 17]
    x[1]    = 0.1  + 0.4 * u[1]    # epsilon   in [0.1, 0.5]
    x[2:16] = -3.0 + 6.0 * u[2:16] # y_ij, a_kl in [-3, 3]
    return x


def naturalness(samples):
    """
    r = max|y_ij| / min|y_ij| over x[2:10] (8 Yukawa only).
    y22=0 is excluded by the index range — never include it.
    Never include Majorana a_kl (x[10:16]).
    """
    y  = np.abs(samples[:, 2:10])
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
    run_dir   = os.path.join(OUTDIR, f"Run_{run_i:02d}")
    log_dir   = os.path.join(run_dir, "log")
    meta_path = os.path.join(run_dir, "metadata.json")
    os.makedirs(run_dir, exist_ok=True)

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
            return None
        raise

    print(f"\n{'='*65}")
    print(f"  RUN {run_i}  NH UltraNest  min_live={MIN_LIVE:,}  dlogz={DLOGZ}")
    print(f"  cluster_min={CLUSTER_LIVE}  nsteps={NSTEPS}")
    print(f"  log_dir = {log_dir}")
    print(f"  (resumes from checkpoint if interrupted)")
    print(f"{'='*65}", flush=True)

    t0 = time.time()

    sampler = ultranest.ReactiveNestedSampler(
        PARAM_NAMES,
        loglike,
        ptform,
        log_dir    = log_dir,
        resume     = 'resume',
        vectorized = False,
    )

    sampler.stepsampler = ss.SliceSampler(
        nsteps             = NSTEPS,
        generate_direction = ss.generate_region_random_direction,
    )

    result = sampler.run(
        min_num_live_points     = MIN_LIVE,
        cluster_num_live_points = CLUSTER_LIVE,
        dlogz                   = DLOGZ,
        min_ess                 = 20_000,
        max_ncalls              = 2_000_000_000,
        show_status             = True,
    )

    logZ    = float(result['logz'])
    logZerr = float(result['logzerr'])
    ncall   = int(result.get('ncall', 0))
    niter   = int(result.get('niter', 0))
    ess     = float(result.get('ess', 0))

    eq_s = np.array(result['samples'], dtype=np.float64)
    r    = naturalness(eq_s)
    p30  = float(np.mean(r < 30) * 100)

    elapsed = (time.time() - t0) / 3600

    print(f"\n  RUN {run_i} COMPLETE")
    print(f"  logZ         = {logZ:.4f}  ±  {logZerr:.4f}")
    print(f"  P(r<30)      = {p30:.2f}%")
    print(f"  ncall        = {ncall:,}")
    print(f"  niter        = {niter:,}")
    print(f"  ESS          = {ess:.0f}")
    print(f"  Time         = {elapsed:.2f}h", flush=True)

    np.savez_compressed(
        os.path.join(run_dir, "results.npz"),
        eq_samples = eq_s.astype(np.float32),
        logZ       = np.float64(logZ),
        logZerr    = np.float64(logZerr),
        p30        = np.float64(p30),
        ncall      = np.int64(ncall),
        niter      = np.int64(niter),
        ess        = np.float64(ess),
    )

    meta = dict(
        run_i    = run_i,
        logZ     = logZ,
        logZerr  = logZerr,
        p30      = p30,
        ncall    = ncall,
        niter    = niter,
        ess      = ess,
        elapsed  = elapsed,
        min_live = MIN_LIVE,
        dlogz    = DLOGZ,
        sampler  = 'ultranest',
        hierarchy = 'NH',
        nufit    = 'NuFIT 6.0 NH with SK',
    )
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


# =============================================================================
# CONVERGENCE CHECK
# =============================================================================
def convergence_check(results):
    logZs = np.array([m['logZ'] for m in results])
    p30s  = np.array([m['p30']  for m in results])

    print(f"\n{'='*65}")
    print(f"CONVERGENCE SUMMARY  NH UltraNest ({len(results)} runs)")
    print(f"{'='*65}")
    print(f"\n  {'Run':>4} {'logZ':>10} {'logZerr':>9} {'P(r<30)':>9} {'ncall':>12}")
    for m in results:
        print(f"  {m['run_i']:>4} {m['logZ']:>10.4f} {m['logZerr']:>9.4f} "
              f"{m['p30']:>8.2f}% {m['ncall']:>12,}")

    sd_logZ   = float(np.std(logZs, ddof=1)) if len(results) > 1 else 0.0
    sd_p30    = float(np.std(p30s,  ddof=1)) if len(results) > 1 else 0.0
    mean_logZ = float(np.mean(logZs))

    # H_NH from dynesty: N_dead/N_live ≈ 3.36M/100k ≈ 33.6 nats
    H_est    = 33.6
    skilling = np.sqrt(H_est / MIN_LIVE)
    ratio    = sd_logZ / skilling if skilling > 0 else 0.0

    print(f"\n  logZ:    mean={mean_logZ:.4f}  SD={sd_logZ:.4f}")
    print(f"  P(r<30): mean={np.mean(p30s):.2f}%  SD={sd_p30:.2f}%")
    print(f"\n  Skilling sigma (H≈{H_est}, N={MIN_LIVE:,}): {skilling:.4f} nats")
    print(f"  Observed SD(logZ):                           {sd_logZ:.4f} nats")
    print(f"  Excess factor:                               {ratio:.1f}x")

    print(f"\n  CONVERGENCE VERDICT:")
    if ratio < 3:
        print(f"  CONVERGED — SD(logZ)={sd_logZ:.3f} < 3x Skilling={3*skilling:.3f}")
        print(f"  Final: P(r<30) = {np.mean(p30s):.2f}% ± {sd_p30:.2f}%")
    elif ratio < 10:
        print(f"  MARGINAL — {ratio:.1f}x Skilling. Consider re-running.")
    else:
        print(f"  NOT CONVERGED — {ratio:.1f}x Skilling.")

    out = os.path.join(OUTDIR, "summary.json")
    with open(out, 'w') as f:
        json.dump({
            'hierarchy':      'NH',
            'n_runs':         len(results),
            'min_live':       MIN_LIVE,
            'logZ_mean':      mean_logZ,
            'logZ_SD':        sd_logZ,
            'p30_mean':       float(np.mean(p30s)),
            'p30_SD':         sd_p30,
            'skilling_sigma': float(skilling),
            'excess_factor':  float(ratio),
            'converged':      bool(ratio < 3),
        }, f, indent=2)
    print(f"  Saved → {out}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"\n{'#'*65}")
    print(f"  NH NESTED SAMPLING — UltraNest")
    print(f"  NuFIT 6.0 NH (with SK atmospheric)")
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
        print("\n  Only 1 run complete. Run again for convergence statistics.")
    else:
        print("\n  No runs completed.")


if __name__ == '__main__':
    main()