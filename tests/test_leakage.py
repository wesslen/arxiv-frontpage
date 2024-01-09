from typing import Dict, List
from frontpage.constants import LABELS, ANNOT_FOLDER, EVAL_FOLDER
from frontpage.utils import dedup_stream
import srsly


def test_for_dedup_stream():
    def dedup_two_stream(combined_stream, original_streams, key="text"):
        uniq = {}
        separated_streams = {stream: [] for stream in original_streams}

        for ex in combined_stream:
            hash_key = hash(ex[key])
            if hash_key not in uniq:
                uniq[hash_key] = ex
                # Determine which stream the example came from and add it there
                for stream_name, stream_examples in original_streams.items():
                    if ex in stream_examples:
                        separated_streams[stream_name].append(ex)
                        break

        return separated_streams["train"], separated_streams["eval"]

    def get_stream(folder_path) -> List[Dict]:
        examples = []
        for label in LABELS:
            path = folder_path / f"{label}.jsonl"
            examples.extend(list(srsly.read_jsonl(path)))
        return examples

    train_examples = get_stream(ANNOT_FOLDER)
    eval_examples = get_stream(EVAL_FOLDER)

    # Dedup individual streams
    train_examples = dedup_stream(train_examples)
    eval_examples = dedup_stream(eval_examples)
    # Combine for global deduplication
    combined_examples = list(train_examples) + list(eval_examples)

    # Perform deduplication and separate the streams
    train_stream, eval_stream = dedup_two_stream(
        combined_examples, {"train": train_examples, "eval": eval_examples}
    )

    assert len(list(eval_examples)) == len(list(eval_stream))
    assert len(list(train_examples)) == len(list(train_stream))
