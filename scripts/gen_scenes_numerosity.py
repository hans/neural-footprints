"""Generate static numerosity scenes for one regime.

Snakemake wildcard ``{regime}`` selects 'confounded' or 'area_controlled'.
Reads numerosity: block from config.yaml.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_config import load_config
from scene_generator_numerosity import (
    generate_numerosity_scenes,
    save_numerosity_scenes,
)


cfg = load_config()
n_cfg = cfg['numerosity']
regime = snakemake.wildcards.regime  # noqa: F821

print(f"\nGenerating numerosity scenes (regime={regime}) "
      f"-- {n_cfg['n_scenes_per_condition']} per condition")
print("=" * 60)

scenes = generate_numerosity_scenes(
    regime=regime,
    n_scenes_per_condition=n_cfg['n_scenes_per_condition'],
    n_low=n_cfg['n_low'],
    n_high=n_cfg['n_high'],
    base_radius=n_cfg['base_radius'],
    area_controlled_total_area=n_cfg['area_controlled_total_area'],
    xy_extent=tuple(n_cfg['scene_xy_extent']),
    z_height=n_cfg['z_height'],
    placement_max_attempts=n_cfg['placement_max_attempts'],
    base_color=n_cfg['base_color'],
    ground_color=n_cfg['ground_color'],
    seed=n_cfg['random_seed'] + (0 if regime == 'confounded' else 1),
)
save_numerosity_scenes(scenes, snakemake.output.scenes)  # noqa: F821
print(f"  Saved scenes -> {snakemake.output.scenes}")  # noqa: F821
