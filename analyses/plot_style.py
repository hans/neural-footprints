"""Shared plotting style for all figures."""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# Semantic color palette used across all figures
COLORS = {
    "blue": "#4878CF",
    "red": "#D65F5F",
    "green": "#6ACC65",
    "gray": "#999999",
    # Semantic aliases
    "pixels": "#4878CF",
    "physics": "#D65F5F",
    "control": "#6ACC65",
    "neutral": "#999999",
}


@contextmanager
def paper_style():
    """Context manager that sets publication-quality matplotlib defaults.

    Usage:
        with paper_style():
            fig, ax = plt.subplots(...)
            ...
            plt.savefig(...)
    """
    overrides = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Lines
        "lines.linewidth": 1.5,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }

    with mpl.rc_context(overrides):
        yield
