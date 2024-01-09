from typing import List, Dict
from pathlib import Path
from functools import cached_property

from wasabi import Printer
import numpy as np
from skops.io import dump, load
from embetter.utils import cached
from sklearn.linear_model import LogisticRegression

from .constants import TRAINED_FOLDER, LABELS, EMBETTER_CACHE, PRETRAINED_FOLDER
from .utils import console

msg = Printer()


class SentenceModel:
    def __init__(self, labels=LABELS) -> None:
        self.labels = labels
        self._models = {
            k: LogisticRegression(class_weight="balanced") for k in self.labels
        }

    def train(self, examples):
        X = self.featurizer.transform([ex["text"] for ex in examples])
        for task, model in self._models.items():
            xs = np.array([X[i] for i, ex in enumerate(examples) if task in ex["cats"]])
            ys = np.array(
                [ex["cats"][task] for ex in examples if task in ex["cats"]], dtype=int
            )
            model.fit(xs, ys)
            console.log(
                f"Trained the [bold]{task}[/bold] task, using {len(xs)} examples."
            )
        return self

    def __call__(self, text: str) -> Dict:
        result = {}
        X = self.featurizer.transform([text])
        for label in self.labels:
            proba = self._models[label].predict_proba(X)[0, 1]
            result[label] = float(proba)
        return result

    def predict(self, texts: List[str]) -> List[Dict]:
        X = self.featurizer.transform(texts)
        result = [{} for _ in texts]
        for label in self.labels:
            probas = self._models[label].predict_proba(X)[:, 1]
            for i, proba in enumerate(probas):
                result[i][label] = float(proba)
        return result

    @cached_property
    def encoder(self):
        from embetter.text import SentenceEncoder

        encoder = SentenceEncoder()
        encoder = cached(EMBETTER_CACHE / "sbert", encoder)
        return encoder

    @cached_property
    def featurizer(self):
        from embetter.text import SentenceEncoder

        if not Path(PRETRAINED_FOLDER).exists():
            console.log("Did not find pretrained model. Falling back.")
            return self.encoder
        console.log(f"Will use custom model found in {PRETRAINED_FOLDER}")
        return SentenceEncoder(PRETRAINED_FOLDER)

    def pretrain(self, examples):
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        from embetter.finetune._contrastive import generate_pairs_batch

        console.log("Starting pretraining sequence.")
        all_pairs = []
        for label in LABELS:
            subset = [ex for ex in examples if label in ex["cats"]]
            pairs = generate_pairs_batch([ex["cats"][label] for ex in subset], n_neg=1)
            all_pairs.extend(pairs)

        input_examples = []
        for pair in pairs:
            text1 = examples[pair.i1]["text"]
            text2 = examples[pair.i2]["text"]
            input_examples.append(
                InputExample(texts=[text1, text2], label=float(pair.label))
            )
        model = SentenceTransformer("all-MiniLM-L6-v2")

        train_dataloader = DataLoader(input_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)

        console.log("Pairs generated. About to tune model.")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            output_path=str(PRETRAINED_FOLDER),
        )
        console.log(f"New encoder saved at {PRETRAINED_FOLDER}")

    @cached_property
    def nlp(self):
        import spacy

        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    def to_disk(self, path: Path = TRAINED_FOLDER):
        if not Path(path).exists():
            Path(path).mkdir(exist_ok=True, parents=True)
        # Delete old files
        if Path(path).exists():
            for p in Path(path).glob("*.h5"):
                p.unlink()
        # Write new files
        for name, clf in self._models.items():
            dump(clf, Path(path) / f"{name}.h5")
        console.log(f"Model saved in folder: [bold]{path}[/bold].")

    @classmethod
    def from_disk(cls, path: Path = TRAINED_FOLDER):
        if not Path(path).exists():
            raise RuntimeError("You need to train a model beforehand.")
        models = {}
        for f in Path(path).glob("*.h5"):
            models[f.stem] = load(f, trusted=True)

        model = SentenceModel(labels=models.keys())
        model._models = models
        console.log(f"Model loaded from: [bold]{path}[/bold].")
        return model
