from abc import ABC, abstractmethod
from omegaconf import DictConfig
import os
from typing import Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from shutil import copy


import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from helpers import set_seed
from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from datasets.abstract_dataset import AbstractDataset
from datasets.base_transforms import BASE_TRANSFORMS_REGISTRY
from postprocessing.postprocessing_utils import load_yaml
from os.path import join, basename, splitext
import yaml
from omegaconf import DictConfig


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data_and_model(
    cfg: DictConfig
) -> Tuple[AbstractDataset, torch.nn.Module]:
    seed = cfg.get("seed", 42)
    set_seed(seed)
    base_transform_args = {"name": cfg.model.name + "_" + cfg.dataset.name}
    model_base_transform_args = cfg.model.pop("base_transform", None)
    if model_base_transform_args:
        base_transform_args.update(model_base_transform_args)
    base_transform = BASE_TRANSFORMS_REGISTRY(**base_transform_args)

    dataset_args = {}
    model_dataset_args = cfg.model.pop("dataset", None)
    if model_dataset_args:
        dataset_args.update(model_dataset_args)

    cfg.criterion = cfg.dataset.pop("criterion", None)
    data: AbstractDataset = DATASET_REGISTRY(
        **cfg.dataset,
        base_transform=base_transform,
        seed=seed,
        **dataset_args,
    )

    cfg.model.pop("pretrained", None)
    cfg.model.output_dim = data.output_dim
    model = MODEL_REGISTRY(**cfg.model)

    return data, model


def copy_n_samples(df, N, data_folder, output_folder, file_column, class_column, score_column, per_class=False, quality="top"):
    os.makedirs(join(image_out_dir, quality), exist_ok=True)
    classes = df[class_column].drop_duplicates().to_list()
    files = []
    class_labels = []
    for c in classes:
        if per_class:
            df_file = df[df[class_column] == c]
        else:
            df_file = df
        if quality == "top":
            indices = df_file[score_column].nlargest(N).index
        elif quality == "bottom":
            indices = df_file[score_column].nsmallest(N).index
        files += list(df_file.loc[indices, file_column].values)
        class_labels += list(df_file.loc[indices, class_column].values)
        if not per_class:
            break

    for i, (f, c) in enumerate(zip(files, class_labels)):
        src_file_path = join(data_folder, f)
        if N == 1:
            out_file_name = join(output_folder, quality,
                                 quality + "_" + str(c) + splitext(f)[1])
        else:
            out_file_name = join(output_folder, quality, str(
                c) + str(i).zfill(2) + splitext(f)[1])

        copy(src_file_path, out_file_name)


def plot_images_from_directory(directory_path, output_path, target_size=(100, 100)):
    # Get a list of all PNG files in the directory
    image_files = [file for file in os.listdir(
        directory_path) if file.endswith('.png')]

    # Create a subplot
    fig, ax = plt.subplots((), figsize=(10, 6))

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(directory_path, image_file)
        img = Image.open(image_path)

        # Resize the image to the target size
        img = img.resize(target_size)

        # Create an AnnotationBbox to display the resized image
        imagebox = OffsetImage(img, zoom=1.0)
        ab = AnnotationBbox(imagebox, (0, 0), frameon=False,
                            xycoords='axes fraction', boxcoords="axes fraction", pad=0)

        # Add the image to the plot
        ax.add_artist(ab)

        # Display the filename as label beneath the image
        ax.text(0.5, -0.1, image_file, transform=ax.transAxes, ha='center')

    # Remove axes
    ax.axis('off')

    # Save or show the plot
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Example usage


base_path = "results/reproduce"
run_name = "Pollen.Augsburg15_ResNet50_#_64_Epoch_100_None_None_None_#"
d = "agg_optimizer_seed"
model = "Bootstrapping-P15-R"
data_folder = "results/reproduce/file_selection"
curriculum_dir = "curriculum/Bootstrapping"
score_realtive_path = join(curriculum_dir, model + ".csv")
image_out_dir = join(base_path, curriculum_dir, model)


score_file = join(base_path, score_realtive_path)


runs = load_yaml(os.path.join(
    base_path, d, run_name, "runs.yaml"))

configs = [DictConfig(load_yaml(os.path.join(
    base_path, "training", r, ".hydra", "config.yaml"))) for r in runs]
cfg = configs[0]

data, model = prepare_data_and_model(cfg)
score_df = pd.read_csv(score_file)
print("data", data.df_train.shape)
print("score", score_df.shape)
columns = data.df_train.columns.to_list() + score_df.columns.to_list()
print(columns)


data.df_train.reset_index(drop=True, inplace=True)
score_df.reset_index(drop=True, inplace=True)

merged_df = pd.concat([data.df_train, score_df], axis=1, ignore_index=True)
print("merged", merged_df.shape)
merged_df.columns = columns

# x = (merged_df["decoded"] != merged_df["Taxon"]).values
# print(np.mean(x))

print(merged_df.columns.to_list())

copy_n_samples(merged_df, 1, data_folder, image_out_dir, "Path",
               "decoded", "ranks", quality="top", per_class=True)
copy_n_samples(merged_df, 1, data_folder, image_out_dir, "Path",
               "decoded", "ranks", quality="bottom", per_class=True)

# plot_images_from_directory(join(image_out_dir, "top"), join(image_out_dir, "top.png"))
