"""Shared LaTeX-style rcParams: Latin Modern Roman text + Computer Modern math."""
import matplotlib.pyplot as plt

def use_latex_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "Latin Modern Roman",
        "mathtext.it": "Latin Modern Roman:italic",
        "mathtext.bf": "Latin Modern Roman:bold",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.5,
    })
