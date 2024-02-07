import argparse
from postprocessing import AggregateGrid, SummarizeGrid, SummarizeCurriculum


def main(args):
    sg = SummarizeGrid(
        results_dir=args.results_dir,
        experiment_id=args.experiment_id,
        max_runs_plot=args.max_runs
    )
    sg.summarize()
    sg.plot_aggregated_bars()
    sg.plot_metrics()
    if not args.aggregate:
        return
    for agg in args.aggregate:
        ag = AggregateGrid(
            results_dir=args.results_dir,
            experiment_id=args.experiment_id,
            aggregate_list=agg,
            max_runs_plot=args.max_runs
        )
        ag.aggregate()
        ag.summarize()

    sc = SummarizeCurriculum(
        results_dir=args.results_dir,
        experiment_id=args.experiment_id,
        max_runs_plot=args.max_runs
    )
    sc.summarize()
    sc.plot_pace()


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
        "-agg", "--aggregate",
        required=False,
        nargs="+",
        action="append",
        help="Aggregate Runs over Parameters"
    )
    ap.add_argument(
        "-mr", "--max-runs",
        required=False,
        type=int,
        help="Limit number of best runs to plot",
        default=None
    )
    args = ap.parse_args()
    main(args)
