"""Explicit YOLOv5 artifact preparation from the active manifest."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from samples._shared.assets import download_asset
from samples.vision.yolov5.runtime.python.model_binding import resolve_selection,SAMPLE_DIR


def download_target(target,output_dir=None,*,variant=None,asset_id=None):
    selection=resolve_selection(target,variant=variant,asset_id=asset_id)
    path=Path(output_dir).expanduser()/selection.asset.filename if output_dir is not None else selection.model_path
    digest=download_asset(selection.asset,path)
    return selection.asset,path,digest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True,choices=('x5','s100','s100p','s600'))
    parser.add_argument('--variant');parser.add_argument('--asset-id');parser.add_argument('--output-dir')
    args=parser.parse_args(argv)
    try:
        asset,path,digest=download_target(args.target,args.output_dir,variant=args.variant,asset_id=args.asset_id)
        print(f'{asset.reference}\nSaved: {path}\nObserved SHA-256: {digest}\nPublisher SHA-256: {asset.sha256 or "unknown"}')
        return 0
    except (ValueError,OSError) as exc:print(f'Error: {exc}',file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
