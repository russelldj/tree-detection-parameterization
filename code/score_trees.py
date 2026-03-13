import json
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tree_registration_and_matching.eval import obj_mee_matching
from tree_registration_and_matching.utils import is_overstory

SHIFT_PER_DATASET_FILE = "/ofo-share/project-data/species-prediction-project/intermediate/shift_per_dataset.json"
SHIFT_QUALITIES_FILE = (
    "/ofo-share/project-data/species-prediction-project/intermediate/shift_quality.csv"
)
FIELD_TREES_FILE = "/ofo-share/project-data/species-prediction-project/raw/ground-reference/ofo_ground-reference_trees.gpkg"
FIELD_PLOTS_FILE = "/ofo-share/project-data/species-prediction-project/raw/ground-reference/ofo_ground-reference_plots.gpkg"
DETECTIONS_FOLDER = (
    "/ofo-share/repos/david/tree-detection-parameterization/data/tree_predictions"
)
OUTPUT_FILE = "/ofo-share/repos/david/tree-detection-parameterization/data/scores.csv"

NUM_WORKERS = 64

# Global state set by worker initializer
_field_trees = None
_plot_bounds = None
_shift_per_dataset = None


def init_worker():
    global _field_trees, _plot_bounds, _shift_per_dataset
    print("trying to init workers")
    _field_trees = gpd.read_file(FIELD_TREES_FILE).to_crs(3310)
    _plot_bounds = gpd.read_file(FIELD_PLOTS_FILE).to_crs(3310)
    _shift_per_dataset = json.load(open(SHIFT_PER_DATASET_FILE))
    _field_trees = cleanup_field_trees(_field_trees)
    print("finished initializing workers")


def cleanup_field_trees(ground_reference_trees: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Filter out trees with high decay class (likely to have broken stems)
    ground_reference_trees = ground_reference_trees[
        ~(ground_reference_trees.decay_class > 2)
    ]

    # First replace any missing height values with pre-computed allometric values
    nan_height = ground_reference_trees.height.isna()
    ground_reference_trees[nan_height].height = ground_reference_trees[
        nan_height
    ].height_allometric

    # For any remaining missing height values that have DBH, use an allometric equation
    nan_height = ground_reference_trees.height.isna()
    # These parameters were fit on paired height, DBH data from this dataset.
    allometric_height_func = lambda x: 1.3 + np.exp(
        -0.3136489123372108 + 0.84623571 * np.log(x)
    )
    allometric_height = allometric_height_func(
        ground_reference_trees[nan_height].dbh.to_numpy()
    )
    ground_reference_trees.loc[nan_height, "height"] = allometric_height

    # Filter out any trees that still don't have height
    ground_reference_trees = ground_reference_trees[
        ~ground_reference_trees.height.isna()
    ]

    return ground_reference_trees


def score_one(args):
    dataset, param_combo, detections_file = args
    print(f"Scoring {param_combo}")

    if dataset not in _shift_per_dataset:
        print(f"No shift found for {dataset}, skipping")
        return None

    shift = _shift_per_dataset[dataset][0]
    plot_id = dataset.split("_")[0]

    ground_plot_perim = _plot_bounds.query("plot_id == @plot_id").copy()
    ground_trees = _field_trees.query("plot_id == @plot_id").copy()

    if len(ground_plot_perim) == 0 or len(ground_trees) == 0:
        print(f"No field data for plot {plot_id}, skipping")
        return None

    # Apply the shift to align field data with drone data
    ground_trees.geometry = ground_trees.geometry.translate(
        xoff=shift[0], yoff=shift[1]
    )
    ground_plot_perim.geometry = ground_plot_perim.geometry.translate(
        xoff=shift[0], yoff=shift[1]
    )

    # Only include overstory trees
    ground_trees = ground_trees[is_overstory(ground_trees)]

    # Remove dead trees before scoring
    ground_trees = ground_trees[ground_trees.live_dead != "D"]

    # Drop trees shorter than 5m
    ground_trees = ground_trees[ground_trees["height"] > 5]

    # Load detections and drop very short trees
    drone_trees = gpd.read_file(detections_file).to_crs(3310)
    drone_trees = drone_trees[drone_trees["height"] > 5]

    try:
        _, metrics = obj_mee_matching(
            ground_trees,
            drone_trees=drone_trees,
            obs_bounds=ground_plot_perim,
            edge_buffer=2,
            min_height=10,
            return_all_metrics=True,
        )
    except Exception as e:
        print(f"Error scoring {dataset} / {param_combo}: {e}")
        return None

    # Parse parameters from the folder name: sigma_{sigma}__b_{b}_c_{c}
    parts = param_combo.split("_")
    sigma = float(parts[1])
    b = float(parts[4])
    c = float(parts[6])

    return {
        "param_combo": param_combo,
        "sigma": sigma,
        "b": b,
        "c": c,
        "dataset": dataset,
        **metrics,
    }


if __name__ == "__main__":
    detections_folder = Path(DETECTIONS_FOLDER)

    tasks = []
    for param_folder in sorted(detections_folder.iterdir()):
        if not param_folder.is_dir():
            continue
        param_combo = param_folder.name
        for dataset_file in sorted(param_folder.glob("*.gpkg")):
            tasks.append((dataset_file.stem, param_combo, dataset_file))

    print(
        f"Scoring {len(tasks)} (dataset, param_combo) pairs with {NUM_WORKERS} workers"
    )

    with Pool(NUM_WORKERS, initializer=init_worker) as pool:
        results = pool.map(score_one, tasks)

    all_results = [r for r in results if r is not None]
    scores_df = pd.DataFrame(all_results)
    scores_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(scores_df)} rows to {OUTPUT_FILE}")
