import argparse
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
RUN_OUTDIR = BASE / "NH_UltraNest1"
PLOTS_OUT = BASE / "Plots2"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_RUNS_TO_INCLUDE = [0, 1, 2]

DATA_NH = {
    "dm21": (7.49e-5, 0.19e-5),
    "dm31": (2.513e-3, 0.021e-3),
    "s12": (0.308, 0.0115),
    "s13": (0.02215, 0.00057),
    "s23": (0.470, 0.015),
}

V_EW = 246.22
GEV2EV = 1e9
LambdaFN_M0 = 1e-12

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

parser = argparse.ArgumentParser(description="Plot NH UltraNest diagnostics for selected runs.")
parser.add_argument(
    "--runs",
    nargs="+",
    type=int,
    default=None,
    help="Run indices to include, e.g. --runs 0 1 or --runs 0 1 2. Defaults to the configured list above.",
)
args = parser.parse_args()
RUN_IDS = list(dict.fromkeys(args.runs if args.runs is not None else DEFAULT_RUNS_TO_INCLUDE))


def physics_builder(x):
    """NH vectorized physics builder matching Unsuppressed Run layout."""
    N = len(x)
    M0 = 10 ** x[:, 0]
    ep = x[:, 1]

    y11, y12, y13, y21, y22, y23, y31, y32, y33 = x[:, 2:11].T
    a11, a12, a13, a22, a23, a33 = x[:, 11:17].T

    Y = np.zeros((N, 3, 3), dtype=complex)
    Y[:, 0, 0] = y11
    Y[:, 0, 1] = ep * y12
    Y[:, 0, 2] = ep * y13
    Y[:, 1, 0] = ep * y21
    Y[:, 1, 1] = y22
    Y[:, 1, 2] = (ep ** 2) * y23
    Y[:, 2, 0] = ep * y31
    Y[:, 2, 1] = (ep ** 2) * y32
    Y[:, 2, 2] = y33

    MR = np.zeros((N, 3, 3), dtype=complex)
    MR[:, 0, 0] = a11
    MR[:, 0, 1] = a12 * (ep ** 3)
    MR[:, 0, 2] = a13 * ep
    MR[:, 1, 0] = a12 * (ep ** 3)
    MR[:, 1, 1] = a22 * (ep ** 4)
    MR[:, 1, 2] = a23 * (ep ** 2)
    MR[:, 2, 0] = a13 * ep
    MR[:, 2, 1] = a23 * (ep ** 2)
    MR[:, 2, 2] = a33

    MR *= M0[:, None, None]

    detM = np.linalg.det(MR)
    valid = np.abs(detM) > 1e-300

    InvMR = np.zeros_like(MR)
    if valid.any():
        InvMR[valid] = np.linalg.inv(MR[valid])

    mnu = -0.5 * GEV2EV * V_EW ** 2 * (Y @ InvMR @ np.transpose(Y, (0, 2, 1)))
    mnu = 0.5 * (mnu + np.transpose(mnu, (0, 2, 1)))

    H = mnu.conj().transpose((0, 2, 1)) @ mnu
    H = 0.5 * (H + H.conj().transpose((0, 2, 1)))

    try:
        msq, U = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        valid[:] = False
        return {}, valid

    valid &= np.all(msq >= 0, axis=1)

    u2 = U[:, :, 1]
    u3 = U[:, :, 2]

    s13 = np.abs(u3[:, 0]) ** 2
    denom = np.maximum(1.0 - s13, 1e-12)
    s12 = np.clip(np.abs(u2[:, 0]) ** 2 / denom, 0, 1)
    s23 = np.clip(np.abs(u3[:, 1]) ** 2 / denom, 0, 1)

    dm21 = msq[:, 1] - msq[:, 0]
    dm31 = msq[:, 2] - msq[:, 0]

    # Lambda_FN = 1e-12 * M0, so log10 Lambda_FN = log10 M0 - 12.
    log10_LambdaFN = x[:, 0] + np.log10(LambdaFN_M0)

    return dict(
        s12=s12,
        s13=s13,
        s23=s23,
        dm21=dm21,
        dm31=dm31,
        epsilon=x[:, 1],
        log10_LambdaFN=log10_LambdaFN,
    ), valid


def naturality(x):
    """Return the naturalness ratio r = Ymax / Ymin for the Yukawa block."""
    absY = np.abs(x[:, 2:11])
    Ymin = np.where(absY > 1e-10, absY, np.inf).min(axis=1)
    Ymax = absY.max(axis=1)
    r = Ymax / Ymin
    return np.where(np.isfinite(Ymin) & (Ymin > 0), r, np.inf)


def load_run(run_idx):
    res_path = RUN_OUTDIR / f"Run_{run_idx:02d}" / "results.npz"
    if not res_path.exists():
        raise FileNotFoundError(f"Results file not found: {res_path}")

    data = np.load(res_path)
    samples = data.get("eq_samples")
    if samples is None:
        raise KeyError(f"eq_samples not found in {res_path}: files={list(data.files)}")

    s = samples.astype(np.float64)
    obs, valid = physics_builder(s)
    valid_mask = valid
    obs_valid = {k: v[valid_mask] for k, v in obs.items()}
    r = naturality(s)

    print(f"Loaded Run {run_idx}: samples.shape={s.shape}, valid={int(valid_mask.sum()):,}")
    return s, obs_valid, valid_mask, r


def make_corner(data_cols, col_labels, truths, title, fname, color):
    data = np.column_stack(data_cols)
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    if data.size == 0:
        raise ValueError(f"Empty data passed to corner plot for {title}")

    ranges = []
    for j in range(data.shape[1]):
        col = data[:, j]
        if col.size == 0:
            ranges.append((0.0, 1.0))
            continue
        mn = float(np.nanmin(col))
        mx = float(np.nanmax(col))
        if np.isclose(mn, mx):
            span = max(1e-6, abs(mn) * 1e-3, 1e-6)
            ranges.append((mn - span, mn + span))
        else:
            ranges.append((mn, mx))

    fig = corner.corner(
        data,
        labels=col_labels,
        truths=truths,
        range=ranges,
        truth_color="#e63946",
        color=color,
        bins=50,
        smooth=1.2,
        levels=(0.68, 0.95),
        fill_contours=True,
        plot_datapoints=False,
        plot_density=False,
        max_n_ticks=4,
        label_kwargs={"fontsize": 13},
        hist_kwargs={"linewidth": 2},
        contour_kwargs={"linewidths": 1.5},
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11},
        title_fmt=".3g",
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(PLOTS_OUT / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


print(f"\nIncluding runs: {RUN_IDS}")

OBS_BY_RUN = []
R_BY_RUN = []
S_BY_RUN = []
for run_idx in RUN_IDS:
    s, obs_valid, valid_mask, r = load_run(run_idx)
    OBS_BY_RUN.append(obs_valid)
    R_BY_RUN.append(r)
    S_BY_RUN.append(s)

OBS_ALL = {k: np.concatenate([obs[k] for obs in OBS_BY_RUN]) for k in OBS_BY_RUN[0]}
R_ALL = np.concatenate(R_BY_RUN)

RUN_COLORS = ["#457b9d", "#2a9d8f", "#e76f51"]
COLOR_MAP = {run_idx: RUN_COLORS[i % len(RUN_COLORS)] for i, run_idx in enumerate(RUN_IDS)}

print("\nPlot: CDF of naturalness ratio...")
fig, ax = plt.subplots(figsize=(10, 7))

for run_idx, r, col in zip(RUN_IDS, R_BY_RUN, [COLOR_MAP[r] for r in RUN_IDS]):
    rf = np.sort(r[np.isfinite(r)])
    cdf = np.arange(1, len(rf) + 1) / len(rf)
    p30 = float(np.mean(rf < 30) * 100)
    ax.plot(
        rf,
        cdf,
        lw=2.0,
        color=col,
        label=f"Run {run_idx:02d}: P(r<30)={p30:.1f}%  (N={len(rf):,})",
    )
    ax.fill_between(rf, cdf, where=(rf <= 30), color=col, alpha=0.07)

rf_all = np.sort(R_ALL[np.isfinite(R_ALL)])
cdf_all = np.arange(1, len(rf_all) + 1) / len(rf_all)
p30_all = float(np.mean(rf_all < 30) * 100)
ax.plot(
    rf_all,
    cdf_all,
    lw=3,
    color="black",
    ls="--",
    label=f"Combined: P(r<30)={p30_all:.1f}%  (N={len(rf_all):,})",
)

ax.axvline(30, color="black", ls=":", lw=1.5, label="r = 30")
ax.set_xscale("log")
ax.set_xlim(1, 3000)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"Naturalness ratio  $r = |Y_\mathrm{max}|/|Y_\mathrm{min}|$", fontsize=14)
ax.set_ylabel("CDF", fontsize=14)
ax.set_title("NH Naturalness CDF - Selected Runs", fontsize=14, fontweight="bold")
ax.legend(framealpha=0.9, loc="lower right")
ax.grid(True, alpha=0.25, which="both")
plt.tight_layout()
plt.savefig(PLOTS_OUT / "CDF_naturalness.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: CDF_naturalness.png")

print("Plot: Y_min / Y_max PDFs...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for run_idx, s, col in zip(RUN_IDS, S_BY_RUN, [COLOR_MAP[r] for r in RUN_IDS]):
    absY = np.abs(s[:, 2:11])
    yminv = np.where(absY > 1e-10, absY, np.inf).min(axis=1)
    ymaxv = absY.max(axis=1)

    lmin = np.log10(yminv[np.isfinite(yminv) & (yminv > 0)])
    lmax = np.log10(ymaxv[ymaxv > 0])

    kw = dict(bins=80, density=True, histtype="step", lw=2, color=col, label=f"Run {run_idx:02d}", alpha=0.9)
    axes[0].hist(lmin, **kw)
    axes[1].hist(lmax, **kw)

for ax, title, xl in zip(
    axes,
    [r"PDF of $\log_{10}|Y_\mathrm{min}|$", r"PDF of $\log_{10}|Y_\mathrm{max}|$"],
    [r"$\log_{10}|Y_\mathrm{min}|$", r"$\log_{10}|Y_\mathrm{max}|$"],
):
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xl, fontsize=13)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.25)

fig.suptitle("NH UltraNest — Yukawa Extreme-Value Distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_OUT / "PDF_Ymin_Ymax.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: PDF_Ymin_Ymax.png")

print("Plot: Corner plots...")
plot_loop = []
for i, run_idx in enumerate(RUN_IDS):
    plot_loop.append((OBS_BY_RUN[i], f"Run {run_idx:02d}", COLOR_MAP[run_idx], f"Run{run_idx:02d}"))
plot_loop.append((OBS_ALL, "Combined", "#333333", "Combined"))

for obs, lab, col, tag in plot_loop:
    make_corner(
        [obs["epsilon"], obs["log10_LambdaFN"], obs["s12"], obs["s13"], obs["s23"]],
        [r"$\epsilon$", r"$\log_{10}(\Lambda_{\mathrm{FN}}/\mathrm{GeV})$", r"$\sin^2\theta_{12}$", r"$\sin^2\theta_{13}$", r"$\sin^2\theta_{23}$"],
        [None, None, DATA_NH["s12"][0], DATA_NH["s13"][0], DATA_NH["s23"][0]],
        f"NH Mixing Angles — {lab}",
        f"Corner_Angles_{tag}.png",
        col,
    )

    make_corner(
        [obs["epsilon"], obs["log10_LambdaFN"], obs["dm21"] * 1e5, obs["dm31"] * 1e3],
        [r"$\epsilon$", r"$\log_{10}(\Lambda_{\mathrm{FN}}/\mathrm{GeV})$", r"$\Delta m_{12}^2 [10^{-5} \mathrm{eV}^2]$", r"$\Delta m_{31}^2 [10^{-3} \mathrm{eV}^2]$"],
        [None, None, DATA_NH["dm21"][0] * 1e5, DATA_NH["dm31"][0] * 1e3],
        f"NH Masses — {lab}",
        f"Corner_Masses_{tag}.png",
        col,
    )

print(f"\nAll done → {PLOTS_OUT}/")
