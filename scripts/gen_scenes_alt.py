"""
Alternative scene generator with overridable PP_EARLY_FRAME.

Standalone (no snakemake). Uses other config values from config.yaml.

Usage:
    uv run python scripts/gen_scenes_alt.py --early-frame 10 --output data/scenes_ef10.npz
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.io_utils import save_scenes
from scripts.load_config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--early-frame', type=int, required=True,
                    help="t at which the second render is captured")
    ap.add_argument('--output', required=True,
                    help="path to save the regenerated scenes .npz")
    ap.add_argument('--n-scenes', type=int, default=None,
                    help="override config.n_scenes (default: use config)")
    args = ap.parse_args()

    cfg = load_config()
    n_scenes = args.n_scenes if args.n_scenes is not None else cfg['n_scenes']

    # Monkey-patch the scene_generator module's PP_EARLY_FRAME constant
    # before generate_scenes uses it.
    import scene_generator
    scene_generator._CFG_PP_EARLY_FRAME = args.early_frame
    print(f"Set scene_generator._CFG_PP_EARLY_FRAME = {args.early_frame}")

    print(f"Generating {n_scenes} scenes (n_timesteps={cfg['n_timesteps']}, "
          f"early_frame={args.early_frame}) → {args.output}")
    t0 = time.time()
    scenes = scene_generator.generate_scenes(
        n_scenes, cfg['random_seed'],
        n_timesteps=cfg['n_timesteps'],
    )
    print(f"Generation time: {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_scenes(scenes, args.output)
    print(f"Saved → {args.output}")


if __name__ == '__main__':
    main()
