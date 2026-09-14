"""Check complete WikiText2 evidence and the relative-PPL acceptance target."""
import argparse
import json
import math
from pathlib import Path


def main():
    """Reject partial, inconsistent, nonfinite or out-of-target results."""
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path, help='JSON from the complete board evaluator')
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    assert result['num_samples'] == 140 and result['seq_len'] == 2048
    assert result['chunk_size'] == 256 and result['predicted_tokens'] == 286580
    samples = result['samples']
    assert [item['index'] for item in samples] == list(range(140))
    assert all(item['predicted_tokens'] == 2047 and math.isfinite(item['nll']) for item in samples)
    nll = sum(item['nll'] for item in samples)
    assert abs(nll-result['total_nll']) < 1e-6
    assert abs(math.exp(nll/286580)-result['perplexity']) < 1e-9
    increase = result['perplexity']/14.0184-1
    assert increase <= 0.03, increase
    print(f"FULL_RESULT_PASS ppl={result['perplexity']:.8f} relative_increase={increase:.6%}")


if __name__ == '__main__':
    main()
