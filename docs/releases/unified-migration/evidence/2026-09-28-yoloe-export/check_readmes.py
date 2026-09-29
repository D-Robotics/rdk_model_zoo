"""Re-run existing API examples and verify every canonical YOLOE README link."""
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote
import io
import json
import re
import runpy
ROOT = Path(__file__).resolve().parents[5]
prior = Path(__file__).resolve().parent.parent / "2026-09-28-yoloe-python/check_readmes.py"
with redirect_stdout(io.StringIO()) as captured:
    runpy.run_path(str(prior), run_name="__main__")
results = json.loads(captured.getvalue())
links = []
for page in (ROOT / "samples/vision/yoloe").rglob("README*.md"):
    for href in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", page.read_text()):
        if href.startswith(("https:", "http:", "mailto:")):
            continue
        raw, _, anchor = href.partition("#")
        destination = (page.parent / unquote(raw)).resolve() if raw else page
        assert destination.exists(), (page, href)
        if anchor:
            assert 'id="' + anchor + '"' in destination.read_text(), (page, href)
        links.append({"page": str(page.relative_to(ROOT)), "href": href})
results["yoloe_local_links"] = links
print(json.dumps(results, indent=2))
