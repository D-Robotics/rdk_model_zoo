"""Create the recorded 50 x 256-token calibration set from WikiText2 TRAIN."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    """Verify the source split and write SDK text records; never use TEST."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Downloaded TRAIN parquet")
    parser.add_argument("--model-dir", type=Path, required=True, help="Original checkpoint tokenizer")
    parser.add_argument("--output", type=Path, required=True, help="Calibration JSON output")
    args = parser.parse_args()
    with args.train.open("rb") as stream:
        checksum = hashlib.sha256(stream.read()).hexdigest()
    if checksum != "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7":
        raise ValueError("Expected pinned WikiText2 TRAIN parquet")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    text = "\n\n".join(pq.read_table(args.train)["text"].to_pylist())
    tokens = tokenizer.encode(text, add_special_tokens=False)
    records = [{"text": tokenizer.decode(tokens[i*256:(i+1)*256])} for i in range(50)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")


if __name__ == "__main__":
    main()
