"""
IEEE double-column figure style for the JBHI revision.

Design goals (per IEEE Transactions / J-BHI figure rules):
  * Vector output (PDF) + high-DPI PNG (600) for every figure.
  * Serif face (Times) to match IEEE body text; DejaVu Serif fallback on Colab.
  * BOLD axis labels that always carry explicit units.
  * Thick spines and ticks so lines stay legible at column scale.
  * constrained_layout everywhere -> no text/label/legend overlaps.
  * Column widths fixed to IEEE spec: single = 3.5 in, double = 7.16 in.

Usage
-----
    from ieee_style import set_ieee_style, new_fig, finalize_axes, save_fig
    set_ieee_style()
    fig, ax = new_fig("single", height=2.2)
    ax.plot(t, y)
    finalize_axes(ax, xlabel="Time (s)", ylabel="Amplitude (a.u.)")
    save_fig(fig, out_dir, "fig_name")   # writes fig_name.pdf AND fig_name.png
"""

import matplotlib
matplotlib.use("Agg")            # headless-safe; remove if you want inline display
import matplotlib.pyplot as plt

# IEEE column widths (inches)
COL_SINGLE = 3.5
COL_DOUBLE = 7.16

# Colour-blind-safe palette (Okabe-Ito), stable role -> colour mapping
COLORS = {
    "truth":    "#009E73",   # bluish green  -> ground truth
    "proposed": "#0072B2",   # blue          -> proposed model
    "cnn":      "#D55E00",   # vermillion    -> CNN
    "unet":     "#CC79A7",   # reddish purple-> U-Net
    "wgan":     "#E69F00",   # orange        -> WGAN-GP
    "pcg":      "#666666",   # grey          -> PCG input
    "accent":   "#0072B2",
    "muted":    "#999999",
}


def set_ieee_style():
    """Apply the global rcParams. Call once at the top of every script."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        "font.size":         8,
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":  "dejavuserif",
        "text.usetex":       False,
        "axes.labelsize":    9,
        "axes.labelweight":  "bold",
        "axes.titlesize":    9,
        "axes.titleweight":  "bold",
        "axes.linewidth":    1.1,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.4,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.fontsize":   7.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "0.7",
        "legend.handlelength": 1.6,
        "lines.linewidth":   1.0,
        "lines.solid_capstyle": "round",
    })


def new_fig(width="single", height=2.2, nrows=1, ncols=1, **kw):
    """Create a figure sized to an IEEE column.

    width : 'single' (3.5 in), 'double' (7.16 in), or a float in inches.
    height: figure height in inches.
    Returns (fig, ax) or (fig, axes) exactly like plt.subplots.
    """
    w = {"single": COL_SINGLE, "double": COL_DOUBLE}.get(width, width)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, height), **kw)
    return fig, ax


def finalize_axes(ax, xlabel=None, ylabel=None, title=None,
                  legend=False, legend_loc="best", legend_ncol=1):
    """Apply bold labels with units, thicken spines, tidy ticks.

    Always pass xlabel/ylabel WITH units, e.g. 'Time (ms)', 'PSNR (dB)'.
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", width=1.0, labelsize=8)
    if legend:
        leg = ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=True)
        leg.get_frame().set_linewidth(0.6)
    return ax


def save_fig(fig, out_directory, name):
    """Save both a vector PDF (for LaTeX) and a 600-dpi PNG (for preview)."""
    import os
    pdf = os.path.join(out_directory, f"{name}.pdf")
    png = os.path.join(out_directory, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  saved  {pdf}")
    print(f"  saved  {png}")
    return pdf, png
