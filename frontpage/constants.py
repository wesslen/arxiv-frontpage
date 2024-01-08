import srsly
from pathlib import Path 
from typing import Literal 

from .types import Config

# Paths and folders
DATA_FOLDER = Path("data")
ANNOT_FOLDER = DATA_FOLDER / "annot"
EVAL_FOLDER = DATA_FOLDER / "eval"
INDICES_FOLDER = Path("indices")
CLEAN_DOWNLOADS_FOLDER = DATA_FOLDER / Path("cleaned")
DOWNLOADS_FOLDER = DATA_FOLDER / "downloads"
ANNOT_PATH = ANNOT_FOLDER / "annotations.jsonl"
ACTIVE_LEARN_PATH = ANNOT_FOLDER / "active-learn.jsonl"
SECOND_OPINION_PATH = ANNOT_FOLDER / "second-opinion.jsonl"
TRAINED_FOLDER = Path("training")
TEMPLATE_PATH = Path("templates/home.html")
CONFIG_FILE = "config.yml"
PRETRAINED_FOLDER = TRAINED_FOLDER / "custom-sbert-emb"

# Cache paths
EMBETTER_CACHE = Path("cache") / "embetter"

# Possible values
DATA_PARTITION = 0.1
DATA_LEVELS = ["sentence", "abstract"]
DATA_LEVELS_TYPE = Literal["sentence", "abstract"]
DATA_TYPES = ["training", "evaluation"]
CONFIG = Config(**srsly.read_yaml(CONFIG_FILE))
LABELS = [s.label for s in CONFIG.sections]
THRESHOLDS = {s.label: s.threshold for s in CONFIG.sections}
SITE_PATH = Path("index.html")
INPUT_PATH = Path("data/input.jsonl")
