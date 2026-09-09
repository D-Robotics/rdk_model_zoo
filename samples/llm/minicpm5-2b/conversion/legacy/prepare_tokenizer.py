"""Prepare a separate, text-only tokenizer for OELLM S100 SDK 1.0.0."""
import argparse
import json
import shutil
from pathlib import Path


def prepare(source, output):
    """Copy tokenizer metadata only, adapting merges and the existing chat EOS.

    Args:
        source: Original Hugging Face checkpoint directory.
        output: New directory; must not already exist.
    """
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "config.json", "generation_config.json", "chat_template.jinja")
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    data = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    merges = data["model"]["merges"]
    if not all(isinstance(pair, list) and len(pair) == 2 and
               all(isinstance(part, str) and " " not in part for part in pair) for pair in merges):
        raise ValueError("Expected original two-string BPE merges")
    data["model"]["merges"] = [" ".join(pair) for pair in merges]
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copy2(source / name, output / name)
    (output / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = output / name
        config = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(config["eos_token"], dict):
            config["eos_token"]["content"] = "<|im_end|>"
        else:
            config["eos_token"] = "<|im_end|>"
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    template = Path(__file__).with_name("legacy-chat.jinja").read_text(encoding="utf-8").rstrip("\n")
    (output / "simple-chat.jinja").write_text(template, encoding="utf-8")


def main():
    """Parse checkpoint and output paths without modifying the checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original checkpoint directory")
    parser.add_argument("output", type=Path, help="New deployment tokenizer directory")
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
