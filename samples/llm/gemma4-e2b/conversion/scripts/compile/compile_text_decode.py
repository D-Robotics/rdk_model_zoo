#!/usr/bin/env python3
"""从已有 decode convert_removed.bc 续编 HBO，并与 prefill.hbo link 成 HBM。"""
import os
import sys

from hbdk4.compiler.apis import compile, link, load
from hbdk4.compiler.hbm import Hbo

OUT_DIR = os.environ.get(
    "COMPILE_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "output", "gemma4_e2b_text"),
)
OUT_DIR = os.path.abspath(OUT_DIR)
PREFIX = os.path.join(OUT_DIR, "gemma4-e2b_lm_chunk_256_cache_4096_ptq")

DECODE_BC = f"{PREFIX}.decode_convert_removed.bc"
PREFILL_HBO = f"{PREFIX}.prefill.hbo"
DECODE_HBO = f"{PREFIX}.decode.hbo"
HBM_PATH = f"{PREFIX}.hbm"
CACHE_PATH = os.path.join(OUT_DIR, "compile_cache_decode")

JOBS = int(os.environ.get("HBDK_JOBS", "29"))
OPT = int(os.environ.get("HBDK_OPT", "0"))
CACHE_MODE = os.environ.get("HBDK_CACHE_MODE", "enable")
MARCH = os.environ.get("HBDK_MARCH", "nash-m")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    for path, name in [
        (DECODE_BC, "decode BC"),
        (PREFILL_HBO, "prefill HBO"),
    ]:
        if not os.path.isfile(path):
            log(f"ERROR: missing {name}: {path}")
            return 1

    log(f"[config] jobs={JOBS} opt={OPT} cache_mode={CACHE_MODE} march={MARCH}")
    if os.path.isfile(PREFILL_HBO) and os.path.isfile(DECODE_HBO):
        log("[fast-path] Both HBO files exist, linking only ...")
        link([Hbo(PREFILL_HBO), Hbo(DECODE_HBO)], HBM_PATH)
        log(f"DONE: {HBM_PATH}")
        return 0

    log(f"[1/3] Loading decode BC: {DECODE_BC}")
    decode_module = load(DECODE_BC)
    log(f"      Functions: {[f.name for f in decode_module.functions]}")

    os.makedirs(CACHE_PATH, exist_ok=True)
    if os.path.isfile(DECODE_HBO) and os.path.getsize(DECODE_HBO) > 0:
        log(f"[2/3] Skip compile: decode HBO already exists ({DECODE_HBO})")
    else:
        log(f"[2/3] Compiling decode HBO -> {DECODE_HBO}")
        decode_hbo = compile(
            decode_module,
            DECODE_HBO,
            march=MARCH,
            opt=OPT,
            jobs=JOBS,
            core_num=1,
            progress_bar=True,
            cache_mode=CACHE_MODE,
            cache_path=CACHE_PATH,
        )
        log(f"      decode HBO done: {DECODE_HBO}")

    log(f"[3/3] Linking prefill + decode -> {HBM_PATH}")
    link([Hbo(PREFILL_HBO), Hbo(DECODE_HBO)], HBM_PATH)
    log(f"DONE: {HBM_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
