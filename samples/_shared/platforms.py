# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Read concrete target identities; preparation is not evidence of local hardware.

Only explicitly recorded aliases are recognized. Artifact compatibility and runtime
requirements remain the responsibility of each reviewed model binding.
"""
from functools import lru_cache
import json
from pathlib import Path
from typing import Optional

SOC_NAME_PATH = Path('/sys/class/boardinfo/soc_name')
SOCINFO_NAME_PATH = Path('/sys/class/socinfo/soc_name')
BOARD_TYPE_PATH = Path('/sys/class/boardinfo/board_type')
DEVICE_TREE_MODEL_PATH = Path('/proc/device-tree/model')
_REGISTRY = Path(__file__).resolve().parents[2] / 'platforms/registry.json'


@lru_cache(maxsize=1)
def _targets() -> tuple:
    with _REGISTRY.open(encoding='utf-8') as handle:
        registry = json.load(handle)
    if registry.get('schema_version') != 1:
        raise ValueError('Unsupported target registry schema.')
    return tuple(registry['targets'])


def match_target(soc_name: str, board_type: Optional[str] = None) -> Optional[str]:
    """Resolve exact OS identity strings, including the known S100P refinement."""
    soc = (soc_name or '').strip().lower()
    board = (board_type or '').strip().lower()
    for target in _targets():
        if soc == target.get('base_soc') and board in target.get('board_types', ()):
            return target['id']
    for target in _targets():
        if soc in target['soc_names']:
            return target['id']
    return None


def _read_identity(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding='utf-8').strip('\x00 \t\r\n') or None
    except OSError:
        return None


def detect_target() -> Optional[str]:
    """Read local board information once per call, returning None if unknown."""
    soc = _read_identity(SOC_NAME_PATH)
    if soc:
        return match_target(soc, _read_identity(BOARD_TYPE_PATH))
    socinfo = _read_identity(SOCINFO_NAME_PATH)
    if socinfo:
        for target in _targets():
            if socinfo.lower() in target.get('socinfo_names', ()):
                return target['id']
        return None
    model = _read_identity(DEVICE_TREE_MODEL_PATH)
    for target in _targets():
        if model in target.get('device_tree_models', ()):
            return target['id']
    return None


def resolve_target(requested: str = 'auto', *, soc_name: Optional[str] = None,
                   board_type: Optional[str] = None) -> str:
    """Select a concrete target for preparation without claiming execution support.

    Args:
        requested: auto, x5, s100, s100p or s600; None is a legacy auto alias.
        soc_name: Optional already-observed identity, used only for auto.
        board_type: Optional observed variant, used with soc_name.

    Raises:
        ValueError: If the requested or automatically detected target is unknown.
    """
    key = (requested or 'auto').strip().lower()
    if key != 'auto':
        if key not in {target['id'] for target in _targets()}:
            raise ValueError(f'Unknown concrete target {requested!r}; use x5/s100/s100p/s600 or auto.')
        return key
    detected = match_target(soc_name, board_type) if soc_name is not None else detect_target()
    if detected is None:
        raise ValueError('Cannot identify this board. Select an explicit target for preparation only.')
    return detected


def require_execution_target(requested: str) -> str:
    """Reject unknown local hardware or a requested/observed target mismatch."""
    actual = detect_target()
    if actual is None:
        raise ValueError('Local execution requires recognized board identity; an explicit target is not hardware evidence.')
    expected = actual if requested in (None, 'auto') else resolve_target(requested)
    if expected != actual:
        raise ValueError(f'Target mismatch: requested {expected}, detected {actual}.')
    return actual
