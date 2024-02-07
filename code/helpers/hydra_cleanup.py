import os
import shutil
import argparse


def on_multirun_end(results_dir, experiment_id) -> None:
    dir = os.path.join(results_dir, experiment_id, "training")
    run_names = [a for a in os.listdir(dir)
                 if os.path.isdir(os.path.join(dir, a))]
    for run_name in run_names:
        metrics_file = "metrics.csv"
        p = os.path.join(dir, run_name, metrics_file)
        if not os.path.exists(p):
            shutil.rmtree(os.path.join(dir, run_name))


def on_curriculum_end(results_dir, experiment_id) -> None:
    dir = os.path.join(results_dir, experiment_id, "curriculum")
    for base_dir, _, files in os.walk(dir):
        if not os.path.exists(os.path.join(base_dir, "score.yaml")):
            continue
        if os.path.exists(os.path.join(base_dir, "scores.csv")):
            continue
        shutil.rmtree(base_dir)


def main(args):
    if args.run_type == "multirun":
        on_multirun_end(
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
        )
    elif args.run_type == "curriculum":
        on_curriculum_end(
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
        )


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
        "-rt", "--run-type",
        required=False,
        help="Type of run cleanup to perform",
        default="multirun",
        choices=["multirun", "curriculum"]
    )
    args = ap.parse_args()
    main(args)
