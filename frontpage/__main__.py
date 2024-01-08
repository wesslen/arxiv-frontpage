import os
import datetime as dt
from pathlib import Path
import srsly

from jinja2 import Template
from radicli import Radicli, Arg

from .utils import console
from .constants import TEMPLATE_PATH, TRAINED_FOLDER, SITE_PATH, INPUT_PATH

cli = Radicli()


@cli.command("download")
def download():
    """Download new data."""
    from .download import main as download_data

    download_data()


@cli.command(
    "index",
    kind=Arg(help="Can be lunr/simsity"),
    level=Arg(help="Can be sentence/abstract"),
)
def index_cli(kind: str, level: str):
    """Creates index for annotation."""
    from .datastream import DataStream

    DataStream().create_index(level=level, kind=kind)


@cli.command("preprocess")
def preprocess_cli():
    """Dedup and process data for faster processing."""
    from .datastream import DataStream

    DataStream().save_clean_download_stream()


@cli.command("annotate")
def annotate():
    """Annotate new examples."""

    def run_questions():
        import questionary
        from .constants import LABELS, DATA_LEVELS, DATA_TYPES

        results = {}

        results["data_type"] = questionary.select(
            "What type of data do you want to annotate?",
            choices=DATA_TYPES,
        ).ask()        

        results["label"] = questionary.select(
            "Which label do you want to annotate?",
            choices=LABELS,
        ).ask()

        results["level"] = questionary.select(
            "What view of the data do you want to take?",
            choices=DATA_LEVELS,
        ).ask()

        if results["level"] == "abstract":
            choices = ["second-opinion", "search-engine", "simsity", "random"]
        else:
            choices = ["simsity", "search-engine", "active-learning", "random"]

        results["tactic"] = questionary.select(
            "Which tactic do you want to apply?",
            choices=choices,
        ).ask()

        results["setting"] = ""
        if results["tactic"] in ["simsity", "search-engine"]:
            results["setting"] = questionary.text(
                "What query would you like to use?", ""
            ).ask()

        if results["tactic"] == "active-learning":
            results["setting"] = questionary.select(
                "What should the active learning method prefer?",
                choices=["positive class", "uncertainty", "negative class"],
            ).ask()
        return results

    results = run_questions()
    from .recipe import annotate_prodigy

    annotate_prodigy(results)


@cli.command("annotprep")
def annotprep():
    """Prepares data for training."""
    from .datastream import DataStream

    DataStream().save_train_stream()
    DataStream().save_eval_stream()


@cli.command("train")
def train():
    """Trains a new model on the data."""
    from .datastream import DataStream
    from .modelling import SentenceModel

    train_examples, eval_examples = DataStream().get_combined_stream()
    SentenceModel().train(examples=train_examples).to_disk()


@cli.command("pretrain")
def pretrain():
    """Trains a new featurizer, set-fit style."""
    from .datastream import DataStream
    from .modelling import SentenceModel

    train_examples, eval_examples = DataStream().get_combined_stream()
    SentenceModel().pretrain(examples=train_examples)


@cli.command("stats")
def stats():
    """Show annotation stats"""
    from .datastream import DataStream

    DataStream().show_annot_stats()


@cli.command(
    "build",
    retrain=Arg("--retrain", "-rt", help="Retrain model?"),
    prep=Arg("--preprocess", "-pr", help="Preprocess again?"),
)
def build(retrain: bool = False, prep: bool = False):
    """Build a new site"""
    from .datastream import DataStream

    if prep:
        preprocess_cli()
    if retrain:
        train()
    console.log("Starting site build process")
    sections, data = DataStream().get_site_content()
    template = Template(Path(TEMPLATE_PATH).read_text())
    rendered = template.render(
        sections=sections,
        today=dt.date.today(),
    )
    srsly.write_jsonl(INPUT_PATH, data)
    SITE_PATH.write_text(rendered)
    console.log("Site built.")


@cli.command(
    "artifact",
    action=Arg(help="Can be upload/download"),
)
def artifact(action: str):
    """Upload/download from wandb"""
    import wandb
    from dotenv import load_dotenv
    from frontpage.constants import PRETRAINED_FOLDER

    load_dotenv()
    run = wandb.init(os.getenv("WANDB_API_KEY"))
    if action == "upload":
        artifact = wandb.Artifact(name="custom-sbert-emb", type="model")
        artifact.add_dir(local_path=PRETRAINED_FOLDER)
        run = wandb.init(project="arxiv-frontpage", job_type="upload")
        run.log_artifact(artifact)
    if action == "download":
        if not PRETRAINED_FOLDER.exists():
            run = wandb.init(project="arxiv-frontpage", job_type="download")
            artifact = run.use_artifact("custom-sbert-emb:latest")
            console.log(
                f"Could not find {PRETRAINED_FOLDER}. So will download from wandb."
            )
            artifact.download(PRETRAINED_FOLDER)
        else:
            console.log(f"{PRETRAINED_FOLDER} already exists. Skip wandb download.")


@cli.command("search")
def search():
    """Annotate new examples."""
    import questionary
    from simsity import load_index
    from .modelling import SentenceModel

    enc = SentenceModel().encoder
    index = load_index("indices/simsity/sentence", encoder=enc)
    while True:
        query = questionary.text("Query:").ask()
        texts, dists = index.query([query], n=5)
        for t in texts:
            print(t)


if __name__ == "__main__":
    cli.run()
