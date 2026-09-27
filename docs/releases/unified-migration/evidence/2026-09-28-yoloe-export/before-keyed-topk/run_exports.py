"""Capture real checkpoint exports and exact code/input bindings; no compiler or board."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
CHECKPOINTS=ROOT.parent/".coordination/yoloe-checkpoints"
BUILD=ROOT.parent/".coordination/yoloe-export-final-v2-20260928"
BUILD.mkdir(exist_ok=False)
source_files=["samples/vision/yoloe/conversion/export.py","samples/vision/yoloe/conversion/export_heads.py","samples/vision/yoloe/conversion/contract.py","samples/vision/yoloe/conversion/calibration.py"]
code={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
records=[]
for variant in ("11s","11m","11l","26n","26s","26m","26l","26x"):
    checkpoint=CHECKPOINTS/f"yoloe-{variant}-seg-pf.pt"
    release=json.loads(checkpoint.with_suffix(".pt.release.json").read_text())
    assert checkpoint.stat().st_size==release["size"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==release["observed_sha256"]
    argv=[sys.executable,"samples/vision/yoloe/conversion/export.py","--weights",str(checkpoint),"--variant",variant,"--output-dir",str(BUILD/variant),"--test-image","samples/vision/yoloe/test_data/office_desk.jpg"]
    record={"variant":variant,"argv":argv,"cwd":str(ROOT),"started_utc":datetime.now(timezone.utc).isoformat(),"checkpoint":release,"code_sha256":code,"log":f"{variant}.log"}
    with (OUT/record["log"]).open("w") as log:
        process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=process.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    result=BUILD/variant/"export.json"
    if result.exists():
        shutil.copyfile(result,OUT/f"{variant}-export.json")
        record["result"]=f"{variant}-export.json"
    records.append(record)
    (OUT/"real-export-results.json").write_text(json.dumps(records,indent=2)+"\n")
    print(variant,process.returncode,flush=True)


sys.exit(any(record["returncode"] for record in records))
