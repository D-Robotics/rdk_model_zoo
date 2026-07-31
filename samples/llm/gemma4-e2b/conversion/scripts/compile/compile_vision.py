#!/usr/bin/env python3
"""Resume Vision HBO/HBM compilation from an existing converted BC."""

from __future__ import annotations

import os

from hbdk4.compiler.apis import compile, link, load

TARGET_DEFAULTS = {
    "s100": {"march": "nash-e", "cores": 1, "jobs": 25, "opt": 0},
    "s100p": {"march": "nash-m", "cores": 1, "jobs": 25, "opt": 0},
    "s600": {"march": "nash-p", "cores": 4, "jobs": 22, "opt": 1},
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
        f"gemma4_e2b_vision_{TARGET_SOC}",
    ),
)
OUT_DIR = os.path.abspath(OUT_DIR)

MARCH = os.environ.get("HBDK_MARCH", DEFAULTS["march"])
CORE_NUM = int(os.environ.get("VIT_CORE_NUM", str(DEFAULTS["cores"])))
JOBS = int(os.environ.get("HBDK_JOBS", str(DEFAULTS["jobs"])))
OPT = int(os.environ.get("HBDK_OPT", str(DEFAULTS["opt"])))
CACHE_MODE = os.environ.get("HBDK_CACHE_MODE", "enable")
MAX_L2M_SIZE = int(os.environ.get("GEMMA4_MAX_L2M_SIZE", "25165824"))

BC_PATH = os.path.join(OUT_DIR, "gemma4-e2b_vit_ptq.convert.bc")
HBO_PATH = os.path.join(OUT_DIR, "gemma4-e2b_vit_ptq.hbo")
HBM_PATH = os.path.join(OUT_DIR, "gemma4-e2b_vit_ptq.hbm")
CACHE_PATH = os.path.join(OUT_DIR, "compile_cache")


def log(message: str) -> None:
    """Print a flush-enabled progress message."""
    print(message, flush=True)


def main() -> int:
    """Compile and link the Vision model for the selected SoC."""
    if not os.path.isfile(BC_PATH):
        log(f"ERROR: missing converted Vision BC: {BC_PATH}")
        return 1

    log(
        f"[config] target={TARGET_SOC} march={MARCH} cores={CORE_NUM} "
        f"jobs={JOBS} opt={OPT} cache_mode={CACHE_MODE}"
    )
    log(f"[1/3] Loading {BC_PATH} ...")
    module = load(BC_PATH)
    log(f"      Functions: {[function.name for function in module.functions]}")

    os.makedirs(CACHE_PATH, exist_ok=True)
    compile_kwargs = {
        "march": MARCH,
        "opt": OPT,
        "jobs": JOBS,
        "core_num": CORE_NUM,
        "progress_bar": True,
        "cache_mode": CACHE_MODE,
        "cache_path": CACHE_PATH,
    }
    if CORE_NUM > 1:
        compile_kwargs["max_l2m_size"] = MAX_L2M_SIZE

    log(f"[2/3] Compiling HBO -> {HBO_PATH}")
    hbo = compile(module, HBO_PATH, **compile_kwargs)
    log(f"      HBO done: {HBO_PATH}")

    log(f"[3/3] Linking HBM -> {HBM_PATH}")
    link([hbo], HBM_PATH)
    log(f"DONE: {HBM_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
