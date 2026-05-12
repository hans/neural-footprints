import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from analyses.plot_figures import plot_pp, plot_pp_frames

fig_dir = os.path.dirname(snakemake.output.figure)
os.makedirs(fig_dir, exist_ok=True)

plot_data = dict(np.load(snakemake.input.plot_data, allow_pickle=False))
plot_pp(plot_data, fig_dir=fig_dir)
plot_pp_frames(plot_data, fig_dir=fig_dir)
