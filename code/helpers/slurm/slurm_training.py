import yaml
import argparse
import os
import sys
from pathlib import Path
from omegaconf import DictConfig
sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent.parent))
from training import ModularTaskTrainer  # noqa
# ? antipattern and not the cleanest way to do this, but it declutters the code/ directory


def main(args):
    path = os.path.join(args.results_dir, args.experiment_id,
                        "training", "slurm.yaml")
    with open(path, "r") as f:
        run_name = yaml.safe_load(f)[int(args.run_id)]
    output_directory = os.path.join(args.results_dir, args.experiment_id,
                                    "training", run_name)
    cfg_path = os.path.join(output_directory, ".hydra", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = DictConfig(cfg)

    trainer = ModularTaskTrainer(
        cfg=cfg,
        output_directory=output_directory,
    )
    trainer.train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-rd", "--results-dir",
        required=True,
        help="Path to Results Directory"
    )
    ap.add_argument(
        "-id", "--experiment-id",
        required=True,
        help="Grid Search Experiment ID"
    )
    ap.add_argument(
        "-r", "--run-id",
        required=True,
        help="Run ID"
    )

    args = ap.parse_args()
    main(args)
