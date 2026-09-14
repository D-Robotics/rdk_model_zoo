#!/usr/bin/env python3
"""Resume Text decode HBO compilation and link it with the prefill HBO."""

from __future__ import annotations

import os

from hbdk4.compiler.apis import compile, link, load
from hbdk4.compiler.hbm import Hbo

TARGET_DEFAULTS = {
    "s100": {"march": "nash-e", "decode_cores": 1, "jobs": 29, "opt": 0},
    "s100p": {"march": "nash-m", "decode_cores": 1, "jobs": 29, "opt": 0},
    "s600": {"march": "nash-p", "decode_cores": 2, "jobs": 22, "opt": 1},
}

TARGET_SOC = os.environ.get("TARGET_SOC", "s100p").lower()
if TARGET_SOC not in TARGET_DEFAULTS:
    raise ValueError("TARGET_SOC must be s100, s100p, or s600")
DEFAULTS = TARGET_DEFAULTS[TARGET_SOC]

OUT_DIR = os.environ.get(
    "COMPILE_OUTPUT_DIR",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "output",
        f"gemma4_e2b_text_{TARGET_SOC}",
    ),
)
OUT_DIR = os.path.abspath(OUT_DIR)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "256"))
CACHE_LEN = int(os.environ.get("CACHE_LEN", "4096"))
PREFIX = os.path.join(
    OUT_DIR,
    f"gemma4-e2b_lm_chunk_{CHUNK_SIZE}_cache_{CACHE_LEN}_ptq",
)

DECODE_BC = f"{PREFIX}.decode_convert_removed.bc"
PREFILL_HBO = f"{PREFIX}.prefill.hbo"
DECODE_HBO = f"{PREFIX}.decode.hbo"
HBM_PATH = f"{PREFIX}.hbm"
CACHE_PATH = os.path.join(OUT_DIR, "compile_cache_decode")

MARCH = os.environ.get("HBDK_MARCH", DEFAULTS["march"])
DECODE_CORE_NUM = int(
    os.environ.get("DECODE_CORE_NUM", str(DEFAULTS["decode_cores"]))
)
JOBS = int(os.environ.get("HBDK_JOBS", str(DEFAULTS["jobs"])))
OPT = int(os.environ.get("HBDK_OPT", str(DEFAULTS["opt"])))
CACHE_MODE = os.environ.get("HBDK_CACHE_MODE", "enable")
MAX_L2M_SIZE = int(os.environ.get("GEMMA4_MAX_L2M_SIZE", "25165824"))


def log(message: str) -> None:
    """Print a flush-enabled progress message."""
    print(message, flush=True)


def main() -> int:
    """Compile the decode graph and link the selected Text HBM."""
    for path, name in ((DECODE_BC, "decode BC"), (PREFILL_HBO, "prefill HBO")):
        if not os.path.isfile(path):
            log(f"ERROR: missing {name}: {path}")
            return 1

    log(
        f"[config] target={TARGET_SOC} march={MARCH} cores={DECODE_CORE_NUM} "
        f"jobs={JOBS} opt={OPT} cache_mode={CACHE_MODE}"
    )
    if os.path.isfile(DECODE_HBO) and os.path.getsize(DECODE_HBO) > 0:
        log("[fast-path] Decode HBO exists; linking prefill + decode only ...")
        link([Hbo(PREFILL_HBO), Hbo(DECODE_HBO)], HBM_PATH)
        log(f"DONE: {HBM_PATH}")
        return 0

    log(f"[1/3] Loading decode BC: {DECODE_BC}")
    decode_module = load(DECODE_BC)
    log(f"      Functions: {[function.name for function in decode_module.functions]}")

    os.makedirs(CACHE_PATH, exist_ok=True)
    compile_kwargs = {
        "march": MARCH,
        "opt": OPT,
        "jobs": JOBS,
        "core_num": DECODE_CORE_NUM,
        "progress_bar": True,
        "cache_mode": CACHE_MODE,
        "cache_path": CACHE_PATH,
    }
    if DECODE_CORE_NUM > 1:
        compile_kwargs["max_l2m_size"] = MAX_L2M_SIZE
    if MARCH == "nash-p":
        compile_kwargs.update(
            enable_hpc=True,
            input_no_padding=True,
            output_no_padding=True,
        )

    log(f"[2/3] Compiling decode HBO -> {DECODE_HBO}")
    compile(decode_module, DECODE_HBO, **compile_kwargs)
    log(f"      Decode HBO done: {DECODE_HBO}")

    log(f"[3/3] Linking prefill + decode -> {HBM_PATH}")
    link([Hbo(PREFILL_HBO), Hbo(DECODE_HBO)], HBM_PATH)
    log(f"DONE: {HBM_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
