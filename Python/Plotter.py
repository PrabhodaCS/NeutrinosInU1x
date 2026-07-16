import os, json
import numpy as np
import matplotlib.pyplot as plt
import corner

BASE = os.path.dirname(__file__)
OUTDIR = os.path.join(BASE, "Plots2")

os.makedirs(OUTDIR, exist_ok=True)

npypaths = [
    os.path.join(BASE, "Run_00", "results.npz"),
    os.path.join(BASE, "Run_01", "results.npz"),
    os.path.join(BASE, "Run_02", "results.npz"),
    ]

#TRIAL LOL
data = np.load(npypaths[2])
print(data)                             #Printing path (For me)

samples = data["eq_samples"]
print(samples.shape)                    #Printing shape (For me)

out = {k: data[k] for k in data.files}
s = out.get('eq_samples', [])
print(s)                                #Checking where samples stored

## PHYSICS DEFNS
v   = 246.22   # GeV (electroweak VEV)
GEV2EV = 1e9

# NuFIT 5.2 IH central values
DATA_IH = {
    "dm21": (7.49e-5,   0.19e-5),    # eV^2,  dm21 = m2^2 - m1^2 > 0
    "dm32": (-2.484e-3, 0.020e-3),   # eV^2,  dm32 = m3^2 - m2^2 < 0  (IH)
    "s12":  (0.308,     0.0115),     # sin^2(theta_12)
    "s13":  (0.02225,   0.0006),     # sin^2(theta_13)
    "s23":  (0.550,     0.0135),     # sin^2(theta_23)
}

def physics_builder(x):
    """
    For my own sanity, I am unpacking N vectors (M0, ep, Y_ij, A_ij), but for this, i need to take transpose
    """
    N = len(x)
    M0 = 10**x[: , 0]
    ep = x[: , 1]
    y11, y12, y13, y21, y22, y23, y31, y32, y33 = x[: , 2:11].T
    a12, a22, a23, a33 = x[: , 11:].T


    # Build Y (N,3,3)
    Y = np.zeros((N, 3, 3), dtype=complex)

    Y[:, 0, 0] = y11    ; Y[:,0,1] = y12*ep    ; Y[:,0,2] = y13*ep
    Y[:, 1, 0] = y21*ep ; Y[:,1,1] = y22       ; Y[:,1,2] = y23*ep**2
    Y[:, 2, 0] = y31*ep ; Y[:,2,1] = y32*ep**2 ; Y[:,2,2] = y33
        
    MR = np.zeros((N, 3, 3), dtype = complex)
    MR[:, 0, 1] = a12 * ep**3
    MR[:, 1, 0] = a12 * ep**3; MR[:, 1, 1] = a22 * ep**4; MR[:, 1, 2] = a23 * ep**2
    MR[:, 2, 1] = a23 * ep**2; MR[:, 2, 2] = a33
    
    MR *= M0[:, None, None]

    detM = np.linalg.det(MR)
    valid = np.abs(detM) > 1e-300

    InvMR = np.zeros_like(MR)
    InvMR[valid] = np.linalg.inv(MR[valid])

    mnu = -0.5 * v**2 * (Y @ InvMR @ np.transpose(Y, (0, 2, 1)))
    mnu = 0.5 * (mnu + np.transpose(mnu, (0, 2, 1)))

    H = np.transpose(mnu.conj(), (0, 2, 1)) @ mnu
    H = 0.5 * (H + np.transpose(H, (0, 2, 1)))

    try:
        msq, U = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        valid[:] = False
        return {}, valid

    valid &= np.all(msq >= 0, axis=1)

    # IH eigenvalue ordering from eigh (ascending): [m3, m1, m2]
    # indices: 0->m3, 1->m1, 2->m2
    u3 = U[:, :, 0]   # nu_3 column
    u2 = U[:, :, 2]   # nu_2 column

    s13  = np.abs(u3[:, 0])**2
    denom = np.maximum(1.0 - s13, 1e-12)
    s12  = np.clip(np.abs(u2[:, 0])**2 / denom, 0, 1)
    s23  = np.clip(np.abs(u3[:, 1])**2 / denom, 0, 1)

    dm21 = msq[:, 2] - msq[:, 1]   # m2^2 - m1^2  > 0
    dm32 = msq[:, 0] - msq[:, 2]   # m3^2 - m2^2  < 0 for IH

    return dict(s12=s12, s13=s13, s23=s23,
                dm21=dm21, dm32=dm32,
                epsilon=samples[:,1],
                log10_M0=samples[:,0]), valid

def naturality(x):
    absY = np.abs(x[:, 2:11])
    Yval = np.where(absY > 1e-10, absY, np.inf)
    Ymin = np.min(absY, axis = 1)
    Ymax = np.max(absY, axis = 1)
    r = Ymax/Ymin
    return np.where(np.isfinite(Ymin) & (Ymin > 0), r, np.inf)

OBS, VALID = [], []
R = []

obs, valid = physics_builder(s)
OBS.append({k: v[valid] for k, v in obs.items()})
VALID.append(valid)
R.append(naturality(s))

print("\nPlot: CDF of naturalness ratio...")
fig, ax = plt.subplots(figsize=(10, 7))

print(type(R[0]))
rf     = np.sort(R[0])
cdf    = np.arange(1, len(rf)+1) / len(rf)
p30    = float(np.mean(rf < 30) * 100)
ax.plot(rf, cdf, lw=2.0, color='#457b9d',
        label=f"{'Run 2'}: P(r<30)={p30:.1f}%  (N={len(rf):,})")
ax.fill_between(rf, cdf, where=(rf <= 30), color='#457b9d', alpha=0.07)

ax.set_xscale('log')
ax.set_xlim(1, 3000)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r'Naturalness ratio  $r = |Y_\mathrm{max}|/|Y_\mathrm{min}|$', fontsize=14)
ax.set_ylabel('CDF', fontsize=14)
ax.set_title('IH Naturalness CDF - All Runs', fontsize=14, fontweight='bold')
ax.legend(framealpha=0.9, loc='lower right')
ax.grid(True, alpha=0.25, which='both')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "CDF_naturalness.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: CDF_naturalness.png")


print("Plot: Y_min / Y_max PDFs...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

absY = np.abs(s[:,2:11])
yminv = np.where(absY > 1e-10, absY, np.inf).min(1)
ymaxv = absY.max(1)

lmin  = np.log10(yminv[np.isfinite(yminv) & (yminv > 0)])
lmax  = np.log10(ymaxv[ymaxv > 0])

kw = dict(bins=80, density=True, histtype='step', lw=2,
              color='#457b9d', label='Run 0', alpha=0.9)
axes[0].hist(lmin, **kw)
axes[1].hist(lmax, **kw)

for ax, title, xl in zip(axes,
    [r'PDF of $\log_{10}|Y_\mathrm{min}|$',
     r'PDF of $\log_{10}|Y_\mathrm{max}|$'],
    [r'$\log_{10}|Y_\mathrm{min}|$',
     r'$\log_{10}|Y_\mathrm{max}|$']):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xl, fontsize=13)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.25)

fig.suptitle('IH UltraNest — Yukawa Extreme-Value Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'PDF_Ymin_Ymax.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: PDF_Ymin_Ymax.png")

def make_corner(data_cols, col_labels, truths, title, fname, color):
    data = np.column_stack(data_cols)
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    fig = corner.corner(
        data,
        labels=col_labels,
        truths=truths,
        truth_color='#e63946',
        color='#457b9d',
        bins=50,
        smooth=1.2,
        levels=(0.68, 0.95),
        fill_contours=True,
        plot_datapoints=False,
        plot_density=False,
        max_n_ticks=4,
        label_kwargs={'fontsize': 13},
        hist_kwargs={'linewidth': 2},
        contour_kwargs={'linewidths': 1.5},
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={'fontsize': 11},
        title_fmt='.3g',
    )
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

make_corner(
    [obs['epsilon'], obs['log10_M0'],
        obs['s12'], obs['s13'], obs['s23']],
    [r'$\epsilon$', r'$\log_{10}(M_0/\mathrm{GeV})$',
        r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\sin^2\theta_{23}$'],
    [None, None, DATA_IH['s12'][0], DATA_IH['s13'][0], DATA_IH['s23'][0]],
    f'IH Mixing Angles — Run 0',
    f'Corner_Angles_Run0.png',
    '#457b9d',
)

make_corner(
    [obs['epsilon'], obs['log10_M0'],
        obs['dm21'] * 1e5, np.abs(obs['dm32']) * 1e3],
    [r'$\epsilon$', r'$\log_{10}(M_0/\mathrm{GeV})$',
        r'$\delta m_{12}^2 [10^{-5} \text{eV}^2]$', r'$\delta m_{32}^2 [10^{-3} \text{eV}^2]$'],
    [None, None, DATA_IH['dm21'][0] * 1e5, abs(DATA_IH['dm32'][0]) * 1e3],
    f'IH Masses — Run 0',
    f'Corner_Masses_Run0.png',
    '#457b9d',
)