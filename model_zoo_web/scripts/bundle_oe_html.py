#!/usr/bin/env python3
"""Keep all original compiler subgraph HTMLs inside one companion report."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


def bundle(paths, model_sha256, title):
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError('Provide one or more distinct original compiler reports')
    if len(model_sha256) != 64 or any(c not in '0123456789abcdef' for c in model_sha256):
        raise ValueError('Provide the exact model SHA256')
    originals = []
    sections = []
    for path in paths:
        data = path.read_bytes()
        if b'<html' not in data[:1024].lower() and b'<!doctype html' not in data[:1024].lower():
            raise ValueError('Compiler report is not HTML: ' + path.name)
        encoded = base64.b64encode(data).decode('ascii')
        originals.append({'name': path.name, 'sha256': hashlib.sha256(data).hexdigest(), 'base64': encoded})
        sections.append('<section><h2>' + html.escape(path.name) + '</h2><iframe title="' + html.escape(path.name, quote=True)
                        + '" src="data:text/html;base64,' + encoded + '"></iframe></section>')
    identity = json.dumps({'kind': 'original-oe-subgraph-bundle-v1', 'artifact_sha256': model_sha256,
                           'reports': originals}, ensure_ascii=True, separators=(',', ':'))
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>' + html.escape(title) + '</title><style>body{margin:24px;font:16px system-ui;color:#1d2735;background:#f7f9fc}h1{font-size:22px}h2{font-size:16px}p{max-width:960px;line-height:1.6}iframe{width:100%;height:850px;border:1px solid #dce2ea;background:white}section{margin-top:28px}code{overflow-wrap:anywhere}</style>'
    document += '<h1>' + html.escape(title) + '</h1><p>Original compiler reports for every BPU subgraph of this model. Values are static compiler estimates; they are not board measurements.</p><p>Model SHA256: <code>' + model_sha256 + '</code></p>'
    document += ''.join(sections) + '<script type="application/json" id="original-oe-reports">' + identity + '</script></html>\n'
    return document.encode('utf-8'), originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('reports', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new companion path; original reports must stay intact')
    payload, reports = bundle(args.reports, args.model_sha256, args.title)
    args.output.write_bytes(payload)
    print(json.dumps({'status': 'bundled-original-reports', 'artifact_sha256': args.model_sha256,
                      'reports': [{k: v for k, v in r.items() if k != 'base64'} for r in reports],
                      'sha256': hashlib.sha256(payload).hexdigest()}))


if __name__ == '__main__':
    main()
