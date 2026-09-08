"""Create a deployment tokenizer bundle without changing the source checkpoint."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    """Validate command-line inputs and create the requested deployment files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Deployment output must differ from source checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'generation_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'special_tokens_map.json'):
        shutil.copy2(args.source / name, args.output / name)
    template = (args.source / 'chat_template.jinja').read_text(encoding='utf-8')
    # Runtime versions load the template from tokenizer_config.json. Pin the
    # same non-thinking mode used for the official generation baseline.
    deployment_template = '{%- set enable_thinking = false -%}' + template
    (args.output / 'chat_template.jinja').write_text(deployment_template, encoding='utf-8')
    config_path = args.output / 'tokenizer_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['chat_template'] = deployment_template
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    from transformers import AutoTokenizer
    reference = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    deployed = AutoTokenizer.from_pretrained(args.output, local_files_only=True)
    cases = [[{'role': 'user', 'content': text}] for text in
             ('What is 1+1? Give a short answer.', '请用一句话介绍你自己。')]
    cases.append([{'role': 'system', 'content': 'Be concise.'},
                  {'role': 'user', 'content': 'Hi'},
                  {'role': 'assistant', 'content': 'Hello!'},
                  {'role': 'user', 'content': 'What is 2+2?'}])
    for messages in cases:
        expected = reference.apply_chat_template(messages, tokenize=True,
                      add_generation_prompt=True, enable_thinking=False)
        actual = deployed.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        assert actual == expected, 'Deployment template changed token sequence'
    print('TOKENIZER_BUNDLE_PASS: exact token equality for English, Chinese and multi-turn')


if __name__ == '__main__':
    main()
