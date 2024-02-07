import os
from pathlib import Path
from omegaconf import DictConfig
import sys
import argparse
sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent.parent))  # noqa
from curriculums import CurriculumScoreManager
from postprocessing.postprocessing_utils import load_yaml
# ? antipattern and not the cleanest way to do this, but it declutters the code/ directory


def execute_slurm_job(results_dir: str, experiment_id: str, job: str):
    score_name, run_name = job.split("/")
    score_config = DictConfig(load_yaml(
        os.path.join(results_dir, experiment_id, "curriculum",
                     score_name, run_name, "score.yaml")
    ))
    run_config = DictConfig(load_yaml(
        os.path.join(results_dir, experiment_id, "curriculum",
                     score_name, run_name, "config.yaml")
    ))
    cs = CurriculumScoreManager(
        score_config,
        output_directory=os.path.join(
            results_dir, experiment_id, "curriculum", score_config.curriculum.scoring.name
        )
    )
    cs.run(run_config, run_name)


def postprocess_slurm_jobs(results_dir: str, experiment_id: str, slurm_posts: str):
    for post_yaml in slurm_posts:
        score_config = DictConfig(load_yaml(os.path.join(
            results_dir, experiment_id, "curriculum",
            "_slurm_postprocess", post_yaml
        )))
        cs = CurriculumScoreManager(
            score_config,
            output_directory=os.path.join(
                results_dir, experiment_id, "curriculum", score_config.curriculum.scoring.name
            )
        )
        cs.postprocess(score_config.curriculum.scoring.id, correlation=False)
    cs._correlation_matrix()


def main(args):
    slurm_job_path = os.path.join(
        args.results_dir, args.experiment_id, "curriculum", "slurm.yaml")
    if args.run_type == "curriculum" and os.path.exists(slurm_job_path):
        slurm_job_list = load_yaml(slurm_job_path)
        execute_slurm_job(
            args.results_dir,
            args.experiment_id,
            slurm_job_list[int(args.run_id)]
        )

    slurm_post_dir = os.path.join(
        args.results_dir, args.experiment_id, "curriculum", "_slurm_postprocess")
    if args.run_type == "postprocessing" and os.path.exists(slurm_post_dir):
        slurm_post_yamls = os.listdir(slurm_post_dir)
        postprocess_slurm_jobs(
            args.results_dir,
            args.experiment_id,
            slurm_post_yamls
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
        "-r", "--run-id",
        required=True,
        help="Run ID"
    )
    ap.add_argument(
        "-rt", "--run-type",
        required=False,
        default="curriculum",
        help="Type of run to perform",
        choices=["curriculum", "postprocessing"]
    )

    args = ap.parse_args()
    main(args)
