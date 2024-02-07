import os
import shutil
import pickle
import warnings
import numpy as np
import pandas as pd
from abc import ABC
import seaborn as sns
from latex import escape
import matplotlib.pyplot as plt

from metrics import METRIC_REGISTRY


class PlotBase(ABC):
    def __init__(
        self,
        output_directory: str,
        training_type: str,
        figsize: tuple,
        latex: bool,
        filetypes: list,
        pickle: bool,
        context: str,
        palette: str,
        replace_none: bool,
        rcParams: dict,
    ) -> None:
        self.output_directory = output_directory
        self.training_type = training_type.lower()
        self.figsize = figsize
        self.latex = latex
        self.filetypes = filetypes
        self.pickle = pickle
        self.context = context
        self.palette = palette
        self.replace_none = replace_none
        sns.set_theme(
            context=self.context,
            style="whitegrid",
            palette=self.palette,
            rc=rcParams
        )
        if self.latex:
            self._enable_latex()

    def save_plot(
        self,
        fig: plt.Figure,
        name: str,
        path: str = "",
        close: bool = True
    ) -> None:
        if self.replace_none:
            self._replace_none(fig)
        if self.latex:
            self._escape_latex(fig)
        base_path = os.path.join(self.output_directory, path)
        os.makedirs(base_path, exist_ok=True)
        if self.pickle:
            pkl_path = os.path.join(base_path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(fig, f)
        for filetype in self.filetypes:
            fpath = os.path.join(base_path, f"{name}.{filetype}")
            fig.savefig(fpath, bbox_inches="tight", dpi=300)
        if close:
            plt.close(fig)

    def _enable_latex(self) -> None:
        if shutil.which("latex") is None:
            warnings.warn("LaTeX requested but not found.")
            return
        plt.rcParams["text.usetex"] = True

    def _apply_to_labels(self, fig: plt.Figure, func: callable) -> None:
        if not fig.axes or fig.axes[0].legend_ is None:
            return
        for text in fig.axes[0].legend_.texts:
            text.set_text(func(text.get_text()))

        # fig.axes[0].legend(fontsize=4)

    def _escape_latex(self, fig: plt.Figure) -> None:
        self._apply_to_labels(fig, escape)

    def _replace_none(self, fig: plt.Figure) -> None:
        def process_label(label):
            parts = label.split("_")
            parts = [part if part != "None" else "~" for part in parts]
            parts = "_".join(parts).replace("_~", "~").replace("~_", "~")
            return parts
        self._apply_to_labels(fig, process_label)


class PlotMetrics(PlotBase):
    def __init__(
        self,
        output_directory: str,
        training_type: str,
        figsize: tuple,
        latex: bool,
        filetypes: list,
        pickle: bool,
        context: str,
        palette: str,
        replace_none: bool,
        rcParams: dict,
    ) -> None:
        super().__init__(
            output_directory,
            training_type,
            figsize,
            latex,
            filetypes,
            pickle,
            context,
            palette,
            replace_none,
            rcParams
        )

    def plot_run(self, metrics: pd.DataFrame, std_scale: float = .1):
        std_columns = metrics.filter(regex="\\.std$").columns
        has_std = not std_columns.empty

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        sns.lineplot(
            data=metrics[["train_loss", "dev_loss"]], ax=ax, dashes=False)
        if has_std:
            for col in ["train_loss", "dev_loss"]:
                ax.fill_between(
                    metrics.index,
                    metrics[col] - std_scale * metrics[f"{col}.std"],
                    metrics[col] + std_scale * metrics[f"{col}.std"],
                    alpha=0.2
                )
        ax.set(xlabel=self.training_type)
        self.save_plot(fig, "loss", path="_plots")

        for key in metrics.columns:
            if ".std" in key or key in ["train_loss", "dev_loss", "iteration"]:
                continue
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            sns.lineplot(
                data=metrics[key], ax=ax, dashes=False)
            if has_std:
                ax.fill_between(
                    metrics.index,
                    metrics[key] - std_scale * metrics[f"{key}.std"],
                    metrics[key] + std_scale * metrics[f"{key}.std"],
                    alpha=0.2
                )
            ax.set(xlabel=self.training_type)
            self.save_plot(fig, key, path="_plots")

    def plot_metric(
        self,
        metrics: pd.DataFrame,
        metric: str,
        metrics_std: pd.DataFrame = None,
        std_scale: float = .1,
        max_runs: int = None,
    ):
        fig = plt.figure(figsize=self.figsize)

        if max_runs is None:
            max_runs = len(metrics.columns)
        metrics, metrics_std = self._select_top_runs(
            metric,
            metrics,
            metrics_std,
            max_runs
        )

        sns.lineplot(data=metrics, dashes=False)

        if metrics_std is not None:
            for col in metrics_std.columns:
                plt.fill_between(
                    metrics.index,
                    metrics[col] - std_scale * metrics_std[col],
                    metrics[col] + std_scale * metrics_std[col],
                    alpha=0.2
                )

        plt.xlabel(self.training_type)
        plt.ylabel(metric)
        path = os.path.join("plots", "training_plots")
        self.save_plot(fig, metric, path)

    def plot_aggregated_bars(
            self,
            metrics_df: pd.DataFrame,
            metric: str,
            subplots_by: int = 0,
            group_by: int = 1,
            split_subgroups: bool = True,
            metrics_std: pd.DataFrame = None,
            std_scale: float = 0.1):
        """
        Generate a bar plots from the metrics_df, which are divided
        by the "subplots_by" column, further grouped according to the 
        "group_by" column. If "split_subgroups" is set to true,
        each group is further split into subgroups based on what comes
        after a potential "-" in the "group_by" entry. Finally the 
        "metric" entries are averaged to create the bars and the std is 
        used as an error bar.
        """
        label_replacement_models = {
            "None": "scratch", "pret": "pretrained", "T": "transfer"}

        # Group metrics by the specified columns
        plot_metrics = metrics_df.groupby(metrics_df.columns[subplots_by])

        # Prepare data for plotting
        df_list = []
        for subplot, plot_dfs in plot_metrics:
            group_metrics = plot_dfs.groupby(plot_dfs.columns[group_by])
            for group, group_df in group_metrics:
                if split_subgroups:
                    # Use the shared prefix for subgroups
                    group_split = group.split("-")
                    group = group_split[0]
                    if len(group_split) > 1:
                        subgroup = group_split[1]
                    else:
                        subgroup = "None"
                values = group_df[metric].dropna().astype(float).values
                if values.size == 0:
                    continue
                m, s = np.mean(values), np.std(values)
                df_list.append({
                    "Subplot": subplot,
                    "Group": group,
                    "Subgroup": subgroup,
                    "Mean": m,
                    "Std": s
                })
        df = pd.DataFrame(df_list)
        num_subplots = len(df)
        fig, ax = plt.subplots(
            nrows=num_subplots,
            ncols=1,
            figsize=(self.figsize[0], .5 * self.figsize[1] * num_subplots)
        )
        for i, (subplot) in enumerate(df["Subplot"]):
            if num_subplots > 1:
                ax_obj = ax[i]
            else:
                ax_obj = ax
            plot_df = df[df["Subplot"] == subplot].reset_index(drop=True)
            bar_plot = sns.barplot(
                data=plot_df,
                x="Group",
                y="Mean",
                hue="Subgroup",
                errorbar=None,
                ax=ax_obj
            )
            for i, row in plot_df.iterrows():
                bar_plot.errorbar(
                    i,
                    row["Mean"],
                    yerr=row["Std"],
                    fmt="none",
                    c="black",
                    capsize=3
                )
            legend_labels = []
            for subgroup in df["Subgroup"].unique():
                if subgroup in label_replacement_models.keys():
                    legend_labels.append(label_replacement_models[subgroup])
                else:
                    legend_labels.append(subgroup)
            handles, _ = ax_obj.get_legend_handles_labels()
            ax_obj.legend(handles, legend_labels,
                          bbox_to_anchor=(0.9, 1), loc="upper left")
            ax_obj.set_xlabel("")
            ax_obj.set_ylabel(metric)
            ax_obj.set_title(subplot)
        plt.tight_layout()
        path = os.path.join("plots", "bar_plots")
        self.save_plot(fig, metric, path)

    def _select_top_runs(
        self,
        metric: str,
        metrics: pd.DataFrame,
        metrics_std: pd.DataFrame,
        max_runs: int
    ):
        if "loss" in metric:
            top_values = metrics.min()
            ascending_order = True
        else:
            m = METRIC_REGISTRY(**{"name": metric, "metric": metric})
            if m.suffix == "max":
                top_values = metrics.max()
                ascending_order = False
            else:
                top_values = metrics.min()
                ascending_order = True

        top_runs = top_values\
            .sort_values(ascending=ascending_order)\
            .head(max_runs)\
            .index

        metrics = metrics[top_runs]
        if metrics_std is not None:
            metrics_std = metrics_std[top_runs]

        return metrics, metrics_std
