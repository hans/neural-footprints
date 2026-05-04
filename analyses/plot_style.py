"""Shared plotting style for all figures."""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# Column widths (inches) for a two-column LaTeX paper
COL_WIDTH = 3.5
FULL_WIDTH = 7.0

# Semantic color palette used across all figures.
# Naming follows the SPEC.md glossary:
#   "render"  = full render slice (RGBA + depth + segmentation)
#   "pixels"  = RGBA only (a subset of render); used by the dynamics
#               pixel-forward model that predicts only RGBA.
COLORS = {
    "blue": "#4878CF",
    "red": "#D65F5F",
    "green": "#6ACC65",
    "gray": "#999999",
    # Semantic aliases
    "render": "#4878CF",
    "pixels": "#4878CF",
    "physics": "#D65F5F",
    "combined": "#D65F5F",
    "control": "#6ACC65",
    "neutral": "#999999",
}


@contextmanager
def paper_style():
    """Context manager that sets publication-quality matplotlib defaults.

    Tuned for single-column (3.5 in) figures in a two-column paper.

    Usage:
        with paper_style():
            fig, ax = plt.subplots(...)
            ...
            plt.savefig(...)
    """
    overrides = {
        # Font — sized for 3.5" column width at 300 dpi
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.format": "pdf",
    }

    with mpl.rc_context(overrides):
        yield
