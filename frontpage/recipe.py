import spacy

import prodigy

from .datastream import DataStream
from .utils import console, extract_before_second_hyphen
datastream = DataStream()



@prodigy.recipe("textcat.arxiv.sentence",
    dataset=("The dataset to save", "positional", None, str),
    label=("The label to annotate", "positional", None, str),
    data_type=("The data partition: training or evaluation", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_sentence(dataset, label, data_type, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    from prodigy import set_hashes
    if tactic == "simsity":
        console.log("Setting up simsity stream")
        stream = datastream.get_ann_stream(query=setting, level="sentence", data_type=data_type)
    elif tactic == "random":
        console.log("Setting up randomized stream")
        stream = datastream.get_random_stream(level="sentence", data_type=data_type)
    elif tactic == "active-learning":
        console.log("Setting up active learning")
        stream = datastream.get_active_learn_stream(label=label, preference=setting, data_type=data_type)
    elif tactic == "search-engine":
        console.log("Setting up lunr query")
        stream = datastream.get_lunr_stream(query=setting, level="sentence", data_type=data_type)
    else:
        raise ValueError("This should never happen.")
    
    exclude = []
    if "evaluation" in dataset:
        exclude.append(extract_before_second_hyphen(dataset))
    
    return {
        "dataset": dataset,
        "exclude": exclude,
        "stream": (set_hashes({**ex, "label": label}) for ex in stream),
        "view_id": "classification",
        "config":{
            "exclude_by": "input"
        }
    }


@prodigy.recipe("textcat.arxiv.abstract",
    dataset=("The dataset to save", "positional", None, str),
    label=("The label to annotate", "positional", None, str),
    data_type=("The data partition: training or evaluation", "positional", None, str),
    tactic=("The tactic to retreive relevant examples", "positional", None, str),
    setting=("Additional setting for the tactic", "positional", None, str),
)
def arxiv_abstract(dataset, label, data_type, tactic, setting):
    """Very general recipe to annotate sentences, using different ordering techniques."""
    from prodigy.components.preprocess import add_tokens
    from prodigy import set_hashes

    if tactic == "simsity":
        console.log("Setting up simsity stream")
        stream = datastream.get_ann_stream(query=setting, level="abstract", data_type=data_type)
    elif tactic == "random":
        console.log("Setting up randomized stream")
        stream = datastream.get_random_stream(level="abstract", data_type=data_type)
    elif tactic == "search-engine":
        console.log("Setting up lunr query")
        stream = datastream.get_lunr_stream(query=setting, level="abstract", data_type=data_type)
    elif tactic == "second-opinion":
        console.log("Setting up second opinion")
        stream = datastream.get_second_opinion_stream(label=label, min_sents=1, max_sents=2, data_type=data_type)
    else:
        raise ValueError("This should never happen.")
    
    exclude = []
    if "evaluation" in dataset:
        exclude.append(extract_before_second_hyphen(dataset))
    else:
        exclude.append(f"{dataset}-evaluation")
    
    nlp = spacy.blank("en")
    stream = ({**ex, "label": label} for ex in stream)
    stream = add_tokens(nlp, stream)
    return {
        "dataset": dataset,
        "exclude": exclude,
        "stream": (set_hashes(ex) for ex in stream),
        "view_id": "blocks",
        "config": {
            "labels": [label],
            "blocks": [
                {"view_id": "ner_manual"},
            ],
            "exclude_by": "input"
        }
    }


def annotate_prodigy(results):
    from prodigy.app import server 
    from prodigy.core import Controller

    dataset_name = datastream.get_dataset_name(results['label'], results['level'], results['data_type'])
    name = "textcat.arxiv.sentence" if results['level'] == 'sentence' else "textcat.arxiv.abstract"
    if results['level'] == 'sentence':
        ctrl_data = arxiv_sentence(dataset_name, results['label'], results['data_type'], results['tactic'], results['setting'])
    else:
        ctrl_data = arxiv_abstract(dataset_name, results['label'], results['data_type'], results['tactic'], results['setting'])
    controller = Controller.from_components(name, ctrl_data)
    server(controller, controller.config)
