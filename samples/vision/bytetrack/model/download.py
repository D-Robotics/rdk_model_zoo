"""Explicit ByteTrack detector preparation, without video or dependency downloads."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples._shared.assets import download_asset
from samples.vision.bytetrack.runtime.python.model_binding import resolve_selection


def download_target(target,output_dir=None,*,asset_id=None):
    selection=resolve_selection(target,asset_id=asset_id)
    path=Path(output_dir).expanduser()/selection.asset.filename if output_dir else selection.model_path
    digest=download_asset(selection.asset,path)
    return selection.asset,path,digest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',required=True,choices=('s100','s100p','s600'));p.add_argument('--asset-id');p.add_argument('--output-dir')
    a=p.parse_args(argv)
    try:
        asset,path,digest=download_target(a.target,a.output_dir,asset_id=a.asset_id)
        print(f'{asset.reference}\nSaved: {path}\nObserved SHA-256: {digest}\nPublisher SHA-256: {asset.sha256 or "unknown"}');return 0
    except (ValueError,OSError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
