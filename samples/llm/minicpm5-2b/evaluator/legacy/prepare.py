"""Export the pinned WikiText2 TEST token stream without S600 SDK dependencies."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    """Validate the dataset and token stream, then write a new evaluation bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.test_data.read_bytes()).hexdigest()
    if digest != "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91":
        raise ValueError("WikiText2 TEST parquet does not match the pinned dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dataset = load_dataset(
        "parquet", data_files={"test": str(args.test_data)}, split="test"
    )
    tokens = tokenizer("\n\n".join(dataset["text"]))["input_ids"]
    ids = np.asarray([tokens], dtype=np.int64)
    if ids.shape != (1, 288009):
        raise ValueError(f"Unexpected token stream shape: {ids.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "input_ids.npy"
    np.save(output, ids)
    if hashlib.sha256(output.read_bytes()).hexdigest() != (
        "a82d4dedc5f60009e026d2cc8f96513054831d9b0bea407cf73b43d3411e2653"
    ):
        raise ValueError("Token IDs differ from the shared S600 evaluation input")
    print("EVALUATION_INPUT_READY", output)


if __name__ == "__main__":
    main()
