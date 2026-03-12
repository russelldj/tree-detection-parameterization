import itertools
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tree_detection_framework.entrypoints.detect_geometric_two_stage import (
    detect_trees_two_stage,
)

CHM_FOLDER = "/ofo-share/project-data/species-prediction-project/intermediate/CHMs"
SHIFT_QUALITIES_FILE = (
    "/ofo-share/project-data/species-prediction-project/intermediate/shift_quality.csv"
)
SHIFT_PER_DATASET_FILE = "/ofo-share/project-data/species-prediction-project/intermediate/shift_per_dataset.json"
FIELD_TREES_FILE = "/ofo-share/project-data/species-prediction-project/raw/ground-reference/ofo_ground-reference_trees.gpkg"
OUTPUT_FOLDER = (
    "/ofo-share/repos/david/tree-detection-parameterization/data/tree_predictions"
)

RESOLUTION = 0.12
NUM_WORKERS = 64

# Grid search parameter values
RASTER_BLUR_SIGMAS = [0.25, 0.5, 1.0, 1.5]
B_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05]
C_VALUES = [0, 0.25, 0.5]


shift_qualities = pd.read_csv(SHIFT_QUALITIES_FILE)
high_quality_datasets = [
    s.split(".")[0]
    for s in shift_qualities[shift_qualities.Quality >= 3].Dataset.values
]

field_trees = gpd.read_file(FIELD_TREES_FILE)

trees_per_plot = field_trees.groupby("plot_id").size()
plots_with_enough_trees = trees_per_plot[trees_per_plot >= 10].index.tolist()
high_quality_datasets = [
    d for d in high_quality_datasets if (d[:4] in plots_with_enough_trees)
]


def run_detection(args):
    dataset, sigma, b, c = args
    chm_file = Path(CHM_FOLDER) / f"{dataset}.tif"
    if not chm_file.exists():
        print(f"CHM file not found, skipping: {chm_file}")
        return

    subfolder = f"sigma_{sigma}__b_{b}_c_{c}"
    output_dir = Path(OUTPUT_FOLDER) / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}.gpkg"

    if output_file.exists():
        print(f"Output already exists, skipping: {output_file}")
        return

    print(f"Running detection: {dataset}, sigma={sigma}, b={b}, c={c}")
    detect_trees_two_stage(
        CHM_file=chm_file,
        tree_tops_save_path=output_file,
        resolution=RESOLUTION,
        raster_blur_sigma=sigma,
        tree_top_detector_kwargs={"b": b, "c": c},
    )


if __name__ == "__main__":
    param_combinations = list(
        itertools.product(high_quality_datasets, RASTER_BLUR_SIGMAS, B_VALUES, C_VALUES)
    )
    print(
        f"Running {len(param_combinations)} detection jobs with {NUM_WORKERS} workers"
    )

    with Pool(NUM_WORKERS) as pool:
        pool.map(run_detection, param_combinations)
