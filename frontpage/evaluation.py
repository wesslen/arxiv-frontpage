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
warnings.filterwarnings('ignore') 

output_path = 'evaluation'

data_stream = DataStream()

def calc_stats(pred_valid, y_valid):
    return {**classification_report(pred_valid, y_valid, output_dict=True)['1']}

def evaluate(label, model):
    res = {"label": label}
    train_examples, eval_examples = data_stream.get_combined_stream()
    n_eval = len(eval_examples)
    X = [ex['text'] for ex in eval_examples if label in ex['cats']]
    y = [ex['cats'][label] for ex in eval_examples if label in ex['cats']]
    for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.9, 1]:
        prob_pred = model.predict(X)
        prediction = [1 if probability[label] > p else 0 for probability in prob_pred]
        stats = calc_stats(prediction, y)
        res = {**res, **stats, "p": p, "n_eval": n_eval}
        if p==THRESHOLDS[label]:
            console.log(f"Current {label} threshold {p}:")
            console.log(res)
        
        yield res

def run_and_save_evaluation(label, model, output_path):
    # Create base output directory if it doesn't exist
    base_dir = Path(output_path)
    base_dir.mkdir(exist_ok=True)

    # Define file path for the specific label
    label_dir = base_dir / label
    label_dir.mkdir(exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_path = label_dir / f"{label}-stats-{current_date}.jsonl"
    # Generate and save stats
    
    stats = evaluate(label, model)
    srsly.write_jsonl(file_path, stats)
    console.log(f"Save {file_path}")

    # Read and print stats
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(1000)
    print(
        pl.read_ndjson(file_path).sort("p")
    )
