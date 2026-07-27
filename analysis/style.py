"""Matplotlib configuration for the manuscript's figures: 7pt type sized for a
journal column, Okabe-Ito colours, seaborn's ticks/paper theme.

`manuscript/figures/` is committed and regenerating it is expected to produce
no diff, so changing any value here changes tracked output.
"""
import matplotlib

matplotlib.use("Agg")
import seaborn as sns

FONT = "Montserrat"

# Okabe-Ito colourblind-safe palette.
COLORS = {
    "pb": "#0173B2",
    "xgb_matched": "#000000",
    "xgb_default": "#999999",
    "accent": "#D55E00",
    "highlight": "#CC79A7",
    "muted": "#999999",
}


def apply():
    sns.set_theme(style="ticks", context="paper", font=FONT,
                  rc={
                      "font.family": "sans-serif",
                      "font.sans-serif": [FONT],
                      "font.size": 7,
                      "axes.labelsize": 7,
                      "axes.titlesize": 8,
                      "xtick.labelsize": 7,
                      "ytick.labelsize": 7,
                      "legend.fontsize": 7,
                      "figure.dpi": 300,
                      "xtick.direction": "in",
                      "ytick.direction": "in",
                      "xtick.top": False,
                      "ytick.right": False,
                      "xtick.major.size": 3,
                      "ytick.major.size": 3,
                      "xtick.major.width": 0.5,
                      "ytick.major.width": 0.5,
                      "xtick.minor.visible": False,
                      "ytick.minor.visible": True,
                      "xtick.minor.size": 1.5,
                      "ytick.minor.size": 1.5,
                      "xtick.minor.width": 0.5,
                      "ytick.minor.width": 0.5,
                      "axes.linewidth": 0.5,
                      "grid.linewidth": 0.3,
                      "grid.alpha": 0.4,
                      "lines.linewidth": 1.0,
                      "lines.markersize": 3,
                      "legend.frameon": False,
                      "savefig.bbox": "tight",
                      "savefig.pad_inches": 0.05,
                  })
