"""Regenerate Fig. 4 (overlay of the proposed reconstruction on the ground truth) for the
final version of JBHI-01557-2026.

The plotting code is the original `plot_overlay_figure` from COLAB_IEEE_FIGURES.ipynb
(cell 21), with the same IEEE style. The data are the saved arrays of the segment it
showed (ECGPCG0001_chunk181, subject S023, exercise). Only the metric line changes:
SSIM now follows the paper's own definition (Section V-B: each signal min-max scaled to
[0, 1], data range 1, 7-sample uniform window, K1 = 0.01, K2 = 0.03, sample covariance),
which gives 0.967 for this segment, matching the Trail-4 results file. The original
notebook computed SSIM on z-scored signals with data range = peak-to-peak (0.836).

Outputs (in the folder given as argv[1]): fig2_overlay_best.pdf (vector) and
fig2_overlay_best.png (600 dpi), same file names as in the manuscript source.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr

DATA = r"E:\NUST MS Thesis\Draft-1\Images\EUSIPCO_2026_Figures\raw_predictions"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---- original IEEE style (COLAB_IEEE_FIGURES.ipynb, cell 4) ----
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 600, 'font.size': 9, 'font.family': 'serif',
    # The original ran on Colab, which has no Times New Roman, so matplotlib fell back to
    # DejaVu Serif; Fig. 3 shows the same font. DejaVu Serif keeps Fig. 4 identical in look.
    'font.serif': ['DejaVu Serif'], 'axes.labelsize': 10,
    'axes.titlesize': 10, 'axes.linewidth': 1.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'xtick.major.size': 4,
    'ytick.major.size': 4, 'legend.fontsize': 8, 'legend.framealpha': 0.9,
    'lines.linewidth': 1.0, 'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'text.usetex': False,
    'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
IEEE_FULL_WIDTH = 7.16
COLORS = {'Ground Truth': '#2ca02c', 'Proposed': '#9467bd'}


def psnr_paper(true, pred):
    mse = np.mean((true - pred) ** 2)
    return 20 * np.log10(np.max(np.abs(true)) / np.sqrt(mse))


def ssim_paper(true, pred, win=7, k1=0.01, k2=0.03):
    """skimage.metrics.structural_similarity defaults on min-max scaled 1-D signals."""
    mm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-8)
    x, y = mm(true.astype(np.float64)), mm(pred.astype(np.float64))
    ux, uy = uniform_filter(x, win), uniform_filter(y, win)
    c = win / (win - 1)
    vx = c * (uniform_filter(x * x, win) - ux * ux)
    vy = c * (uniform_filter(y * y, win) - uy * uy)
    vxy = c * (uniform_filter(x * y, win) - ux * uy)
    c1, c2 = (k1 * 1.0) ** 2, (k2 * 1.0) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    p = (win - 1) // 2
    return float(s[p:-p].mean())


ecg = np.load(os.path.join(DATA, "ecg_ground_truth.npy")).astype(np.float64)
pred = np.load(os.path.join(DATA, "pred_Proposed.npy")).astype(np.float64)
met = {"psnr": psnr_paper(ecg, pred), "corr": pearsonr(ecg, pred)[0], "ssim": ssim_paper(ecg, pred)}

# ---- original plot_overlay_figure (cell 21); vector lines kept for the final version ----
fs = 8000
window_len = 2 * fs
best_start, best_peak = 0, 0
for start in range(0, len(ecg) - window_len, fs // 2):
    peak = np.max(ecg[start:start + window_len])
    if peak > best_peak:
        best_peak, best_start = peak, start
s, e_ = best_start, best_start + window_len
t = np.arange(window_len) / fs
rpeak_idx = np.argmax(ecg[s:e_])
z_half = int(0.2 * fs)
zs, ze = max(0, rpeak_idx - z_half), min(window_len, rpeak_idx + z_half)
t_zoom = np.arange(ze - zs) / fs * 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_FULL_WIDTH, 2.2), gridspec_kw={'width_ratios': [3, 1.5]})
ax1.plot(t, ecg[s:e_], color=COLORS['Ground Truth'], linewidth=0.8, label='Ground Truth')
ax1.plot(t, pred[s:e_], color=COLORS['Proposed'], linewidth=0.6, alpha=0.85, linestyle='--', label='Proposed')
ax1.axvspan(zs / fs, ze / fs, alpha=0.1, color='gray')
ax1.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Amplitude (normalized)', fontsize=9, fontweight='bold')
ax1.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax1.set_title('(a) 2-second segment', fontsize=9, fontweight='bold')
ax1.grid(True, alpha=0.2, linewidth=0.3)
ax2.plot(t_zoom, ecg[s:e_][zs:ze], color=COLORS['Ground Truth'], linewidth=1.2, label='Ground Truth')
ax2.plot(t_zoom, pred[s:e_][zs:ze], color=COLORS['Proposed'], linewidth=1.0, alpha=0.85, linestyle='--', label='Proposed')
ax2.set_xlabel('Time (ms)', fontsize=9, fontweight='bold')
ax2.set_title('(b) QRS complex', fontsize=9, fontweight='bold')
ax2.grid(True, alpha=0.2, linewidth=0.3)
for ax in (ax1, ax2):
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=7, width=1.2)
metric_text = f'PSNR={met["psnr"]:.1f} dB | r={met["corr"]:.3f} | SSIM={met["ssim"]:.3f}'
fig.text(0.5, -0.02, metric_text, ha='center', fontsize=8, fontstyle='italic')
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_overlay_best.png"), dpi=600, bbox_inches='tight', pad_inches=0.08)
fig.savefig(os.path.join(OUT, "fig2_overlay_best.pdf"), bbox_inches='tight', pad_inches=0.08)
print(json.dumps({"window_start_s": best_start / fs, **{k: round(float(v), 4) for k, v in met.items()},
                  "label": metric_text}))
