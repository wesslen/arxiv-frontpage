import pytest
from frontpage.datastream import DataStream, dedup_stream

# You must replace `your_script` with the actual name of your script module


@pytest.fixture(scope="module")
def data_stream():
    return DataStream()


@pytest.fixture(scope="module")
def test_data_stream():
    # This should be mock data resembling the actual data the script would process
    return [
        {"text": "Example training title", "meta": {"data_type": "training"}},
        {"text": "Example evaluation title", "meta": {"data_type": "evaluation"}},
        # Add more mock data as needed
    ]


def test_retrieve_dataset_names(data_stream):
    dataset_names = data_stream.retreive_dataset_names()
    # Assuming you have a set of known dataset names
    known_dataset_names = [
        "prompt-engineering-sentence",
        "prompt-engineering-sentence-evaluation",
        "prompt-engineering-abstract",
        "prompt-engineering-abstract-evaluation",
        "robustness-sentence",
        "robustness-sentence-evaluation",
        "robustness-abstract",
        "robustness-abstract-evaluation",
        "security-sentence",
        "security-sentence-evaluation",
        "security-abstract",
        "security-abstract-evaluation",
        "hci-sentence",
        "hci-sentence-evaluation",
        "hci-abstract",
        "hci-abstract-evaluation",
        "social-sciences-sentence",
        "social-sciences-sentence-evaluation",
        "social-sciences-abstract",
        "social-sciences-abstract-evaluation",
        "education-sentence",
        "education-sentence-evaluation",
        "education-abstract",
        "education-abstract-evaluation",
        "recommender-sentence",
        "recommender-sentence-evaluation",
        "recommender-abstract",
        "recommender-abstract-evaluation",
        "production-sentence",
        "production-sentence-evaluation",
        "production-abstract",
        "production-abstract-evaluation",
        "architectures-sentence",
        "architectures-sentence-evaluation",
        "architectures-abstract",
        "architectures-abstract-evaluation",
        "programming-sentence",
        "programming-sentence-evaluation",
        "programming-abstract",
        "programming-abstract-evaluation",
    ]
    assert set(dataset_names).issubset(
        set(known_dataset_names)
    ), "Dataset names not as expected"


def test_no_duplicates_between_eval_and_train_streams(data_stream, test_data_stream):
    train_stream = list(data_stream._filter_datatype(test_data_stream, "training"))
    eval_stream = list(data_stream._filter_datatype(test_data_stream, "evaluation"))

    # Assuming each example has a unique 'text' field for comparison
    train_texts = set(ex["text"] for ex in train_stream)
    eval_texts = set(ex["text"] for ex in eval_stream)

    duplicates = train_texts.intersection(eval_texts)
    assert (
        not duplicates
    ), f"Found duplicates across training and evaluation streams: {duplicates}"


def test_dedup_stream(data_stream, test_data_stream):
    # Adding duplicates intentionally
    test_data_stream.extend(test_data_stream)
    deduped_stream = list(dedup_stream(test_data_stream, key="text"))
    unique_texts = set(ex["text"] for ex in deduped_stream)
    assert len(deduped_stream) == len(
        unique_texts
    ), "Duplicates were not removed from the stream"


# You can continue to add more tests for each function you'd like to test
