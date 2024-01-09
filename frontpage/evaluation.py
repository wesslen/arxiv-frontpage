import json
import os
import polars as pl
from pathlib import Path
import srsly

from sklearn.metrics import classification_report
from .datastream import DataStream
import numpy as np
from datetime import datetime

from .constants import THRESHOLDS
from .utils import console

import warnings
# remove for polars table
warnings.filterwarnings("ignore")

data_stream = DataStream()


def calc_stats(pred_valid, y_valid):
    return {**classification_report(pred_valid, y_valid, output_dict=True)["1"]}


def evaluate(label, model, output_path):
    res = {"label": label}
    train_examples, eval_examples = data_stream.get_combined_stream()

    current_date = datetime.now().strftime("%Y-%m-%d")
    stats_dir = Path(output_path, "stats")
    stats_dir.mkdir(exist_ok=True)

    X = [ex["text"] for ex in eval_examples if label in ex["cats"]]
    n_x_eval = len(X)
    console.log(f"Load {n_x_eval} X records for {label} evaluation")
    y = [ex["cats"][label] for ex in eval_examples if label in ex["cats"]]
    n_y_eval = len(y)
    console.log(f"Load {n_y_eval} Y records for {label} evaluation")
    if n_x_eval != n_y_eval:
        console.log(f"Warning: {label} evaluation has missing labels.", style="bold red")
    
    for p in [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.52,
        0.54,
        0.56,
        0.58,
        0.6,
        0.62,
        0.64,
        0.66,
        0.68,
        0.7,
        0.72,
        0.74,
        0.76,
        0.78,
        0.8,
        0.9,
        1,
    ]:
        prob_pred = model.predict(X)
        prediction = [1 if probability[label] > p else 0 for probability in prob_pred]
        stats = calc_stats(prediction, y)
        res = {**res, **stats, "p": p, "n_eval": n_y_eval}
        if p == THRESHOLDS[label]:
            console.log(f"Current {label} threshold {p}:")
            console.log(res)
            filename = f"overall-stats-{current_date}.jsonl"
            filepath = os.path.join(stats_dir, filename)

            with open(filepath, 'a') as f:
                f.write(json.dumps(res) + '\n')

            # Print to console
            console.log(f"Write {filepath}")

        yield res


def run_and_save_evaluation(label, model, output_path="evaluation"):
    # Create base output directory if it doesn't exist
    base_dir = Path(output_path)
    base_dir.mkdir(exist_ok=True)

    # Define file path for the specific label
    label_dir = base_dir / label
    label_dir.mkdir(exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = label_dir / f"{label}-stats-{current_date}.jsonl"
    # Generate and save stats

    label_dir = base_dir / label
    label_dir.mkdir(exist_ok=True)
    stats = evaluate(label, model, output_path)
    srsly.write_jsonl(file_path, stats)
    console.log(f"Save {file_path}")

    # Read and print stats
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(1000)
    print(pl.read_ndjson(file_path).sort("p"))
