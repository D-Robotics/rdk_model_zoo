#!/usr/bin/env python3
"""从已有 convert.bc 编译 Vision HBO/HBM（续编用）"""
import os

from hbdk4.compiler.apis import compile, link, load

OUT_DIR = os.environ.get(
    "COMPILE_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "output", "gemma4_e2b_vision"),
)
OUT_DIR = os.path.abspath(OUT_DIR)

JOBS = int(os.environ.get("HBDK_JOBS", "25"))
OPT = int(os.environ.get("HBDK_OPT", "0"))
CACHE_MODE = os.environ.get("HBDK_CACHE_MODE", "enable")

BC_PATH = os.path.join(OUT_DIR, "gemma4-e2b_vit_ptq.convert.bc")
HBO_PATH = os.path.join(OUT_DIR, "gemma4-e2b_vit_ptq.hbo")
HBM_PATH = os.path.join(OUT_DIR, "gemma4-e2b_vit_ptq.hbm")
CACHE_PATH = os.path.join(OUT_DIR, "compile_cache")


def log(msg):
    print(msg, flush=True)


log(f"[config] jobs={JOBS} opt={OPT} cache_mode={CACHE_MODE} march=nash-m")
log(f"[1/3] Loading {BC_PATH} ...")
module = load(BC_PATH)
log(f"      Functions: {[f.name for f in module.functions]}")

os.makedirs(CACHE_PATH, exist_ok=True)
log(f"[2/3] Compiling HBO (jobs={JOBS}, opt={OPT}) ...")
hbo = compile(
    module,
    HBO_PATH,
    march="nash-m",
    opt=OPT,
    jobs=JOBS,
    core_num=1,
    progress_bar=True,
    cache_mode=CACHE_MODE,
    cache_path=CACHE_PATH,
)
log(f"      HBO done: {HBO_PATH}")

log(f"[3/3] Linking HBM -> {HBM_PATH} ...")
link([hbo], HBM_PATH)
log(f"DONE: {HBM_PATH}")
