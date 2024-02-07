import argparse
import os
import yaml
import uuid
import subprocess
import time


def main(args):
    # ? Check if slurm jobs should be run
    if args.run_type == "curriculum":
        training_dir = "curriculum"
        job_prefix = "curr"
        script_name = "slurm_curriculum"
    else:
        training_dir = "training"
        job_prefix = "grid"
        script_name = "slurm_training"

    slurm_file = os.path.join(
        args.results_dir, args.experiment_id,
        training_dir, "slurm.yaml"
    )
    if not os.path.exists(slurm_file):
        return

    # ? Load slurm.yaml config file
    with open(slurm_file, "r") as f:
        slurm_run_list = yaml.safe_load(f)
    num_jobs = len(slurm_run_list)
    num_parallel = min(num_jobs, args.parallel)
    if num_jobs == 0:
        return

    # ? Launch slurm jobs
    os.makedirs("gpu_scripts/_slurm_logs", exist_ok=True)
    short_uuid = str(uuid.uuid4())[:8]
    unique_id = f"{job_prefix}_{short_uuid}"
    sbatch_cmd = (
        f"sbatch",
        f"--array=0-{num_jobs-1}%{num_parallel}",
        f"--job-name='{unique_id}'",
        f"gpu_scripts/slurm_array.sh '{args.results_dir}' '{args.experiment_id}' '{script_name}'",
    )

    subprocess.run(" ".join(sbatch_cmd), shell=True, env=os.environ)

    # ? Wait for slurm jobs to finish
    while True:
        time.sleep(30)
        squeue_cmd = f"squeue -u $USER -n {unique_id} -o %.{len(unique_id)}j"
        try:
            output_str = subprocess.check_output(
                squeue_cmd, shell=True, text=True).strip()
            if unique_id not in output_str:
                break
        except subprocess.CalledProcessError:
            break


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
        "-p", "--parallel",
        required=False,
        default=10,
        help="Number of parallel jobs to run"
    )
    ap.add_argument(
        "-rt", "--run-type",
        required=False,
        default="multirun",
        help="Type of run to perform with slurm",
        choices=["multirun", "curriculum"]
    )

    args = ap.parse_args()
    main(args)
