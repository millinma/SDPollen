import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helpers.plot_utils import PlotBase


class CurriculumPlots(PlotBase):
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

    def plot_score(self, df: pd.DataFrame, score_id: str, num_bins=20) -> None:
        bin_width = 1 / num_bins
        bin_edges = np.arange(0, 1 + bin_width, bin_width)

        fg = sns.FacetGrid(
            df,
            col="decoded",
            col_wrap=5,
            height=self.figsize[1]/2,
            aspect=self.figsize[0]/(self.figsize[1]*2),
            palette=self.palette
        )
        fg.set_titles(col_template="{col_name}")
        fg.map_dataframe(sns.histplot, x="ranks", bins=bin_edges)
        fg.set_ylabels("count")
        fg.fig.suptitle(score_id, y=1.05)
        self.save_plot(fg.fig, score_id)


    
    def plot_scatter_distribution(self, df: pd.DataFrame, score_id: str) -> None:
        label_counts = df["decoded"].value_counts()
        avg_values = df.groupby("decoded")["ranks"].mean()
        avg_values = avg_values.reindex(label_counts.index)

        # Creating a new DataFrame for the scatter plot
        plot_data = pd.DataFrame({
            "decoded": label_counts.index,
            "label_count": label_counts,
            "avg_value": avg_values
        })
        # Creating the scatter plot
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1]))
        sns.regplot(data=plot_data, x='label_count', y='avg_value', logx=True, scatter=True)
        for i in range(plot_data.shape[0]):
            plt.text(plot_data.iloc[i]['label_count'], plot_data.iloc[i]['avg_value'], 
             plot_data.iloc[i]['decoded'], 
             horizontalalignment='right', size='xx-small', color='black', weight='semibold')
        plt.xlabel("Number of Samples")
        plt.ylabel("Average Difficulty Rank")
        plt.xscale("log") 
        self.save_plot(fig, score_id + "_scatter")

    def plot_score_balanced(self, df: pd.DataFrame, score_id: str, num_bins=20) -> None:
        bin_width = 1 / num_bins
        bin_edges = np.arange(0, 1 + bin_width, bin_width)
        label_counts = df["decoded"].value_counts()
        unique_labels = label_counts.index.tolist()
        counts = label_counts.tolist()
        # imbalance_ordered_values = 

        fg = sns.FacetGrid(
            df,
            col="decoded",
            col_wrap=5,
            col_order=unique_labels,
            height=self.figsize[1]/2,
            aspect=self.figsize[0]/(self.figsize[1]*2),
            palette=self.palette
        )

        fg.set_titles(col_template="{col_name}")
        fg.map_dataframe(sns.histplot, stat="percent", x="ranks", bins=bin_edges)
        fg.set_ylabels("ratio [%]")
        for col_name, ax in fg.axes_dict.items():
            count = df[df["decoded"] == col_name]["decoded"].count()
            ax.set_title(f"{col_name} ({count} samples)")
        fg.fig.suptitle(score_id, y=1.05)
        self.save_plot(fg.fig, score_id + "_balanced")

    def plot_correlation_matrix_custom(self, df: pd.DataFrame) -> None:
        name_mapping = {
            "Bootstrapping-P15-E0": "EfficientNet-B0", 
            "Bootstrapping-P15-E4": "EfficientNet-B4", 
            "Bootstrapping-P15-E0T": "EfficientNet-B0-pret", 
            "Bootstrapping-P15-E4T": "EfficientNet-B4-pret", 
            "Bootstrapping-P15-R": "ResNet50",
            "Bootstrapping-P15-RT": "ResNet50-pret" 
        }
        order = [
            "ResNet50",
            "EfficientNet-B0",
            "EfficientNet-B4",
            "ResNet50-pret", 
            "EfficientNet-B0-pret",
            "EfficientNet-B4-pret"
        ]
        # order = [
        #     "Bootstrapping-P15-R",
        #     "Bootstrapping-P15-E0",
        #     "Bootstrapping-P15-E4",
        #     "Bootstrapping-P15-RT", 
        #     "Bootstrapping-P15-E0T",
        #     "Bootstrapping-P15-E4T"
        # ]
        df.sort_index(axis="columns", inplace=True)
        fig_height = int(1 + 0.75 * len(df.columns))
        fig = plt.figure(figsize=(self.figsize[0], fig_height))
        # Reorder rows and columns based on the specified order
        # Replace column and row names based on the custom dictionary
        df_renamed = df.rename(columns=name_mapping, index=name_mapping)
        df_renamed = df_renamed[order]
        
        correlation_matrix = df_renamed.corr(method="spearman")
        percent_char = "\\%" if self.latex else "%"
        
        def escape_percent(x): return f"{(x*100):.2f}{percent_char}"
        
        sns.heatmap(
            correlation_matrix,
            annot=correlation_matrix.applymap(escape_percent),
            fmt="",
            cbar=False,
            cmap="crest",
        )
    
        fig.gca().set_xticklabels(df_renamed.columns, rotation=25)
        self.save_plot(fig, "correlation_matrix_custom")
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        df.sort_index(axis="columns", inplace=True)
        fig_height = int(1 + 0.75 * len(df.columns))
        fig = plt.figure(figsize=(self.figsize[0], fig_height))
        correlation_matrix = df.corr(method="spearman")
        percent_char = "\\%" if self.latex else "%"
        def escape_percent(x): return f"{(x*100):.2f}{percent_char}"
        sns.heatmap(
            correlation_matrix,
            annot=correlation_matrix.map(escape_percent),
            fmt="",
            cbar=False,
            cmap="crest",
        )
        fig.gca().set_xticklabels(df.columns, rotation=25)
        self.save_plot(fig, "correlation_matrix")

    def plot_run(self, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=self.figsize)
        sns.lineplot(
            data=df["dataset_size"],
            dashes=False,
            drawstyle="steps-post"
        )
        plt.xlabel(self.training_type)
        plt.ylabel("dataset size")
        self.save_plot(fig, "pace")

    def plot_pace(self, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=self.figsize)
        sns.lineplot(
            data=df,
            dashes=False,
            drawstyle="steps-post"
        )
        plt.xlabel(self.training_type)
        plt.ylabel("dataset size")
        self.save_plot(fig, "pace")
