# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Read existing asset facts without a second URL registry or publisher runtime.

The current manifests have model IDs and asset filenames, not standalone asset
IDs. A reference qualifies those existing fields as group:sample:filename;
it does not rename any published identity or certify hardware support.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Asset:
    """Immutable publication facts; unknown publisher hashes remain None."""
    group: str
    sample_id: str
    filename: str
    format: str
    url: Optional[str]
    sha256: Optional[str]

    @property
    def reference(self) -> str:
        """Return the platform-qualified reference to this exact manifest row."""
        return f'{self.group}:{self.sample_id}:{self.filename}'

    @property
    def source_path(self) -> str:
        """Return the authority location relative to the source checkout."""
        return f'platforms/{self.group}/docs/release/models.yaml'


@lru_cache(maxsize=2)
def _models(group: str) -> tuple:
    if group not in ('x5', 's'):
        raise ValueError(f'Unsupported active manifest group {group!r}.')
    import yaml
    with (_ROOT / f'platforms/{group}/docs/release/models.yaml').open(encoding='utf-8') as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict) or not isinstance(content.get('models'), list):
        raise ValueError(f'Invalid models manifest for {group}.')
    return tuple(content['models'])


def list_assets(group: str, sample_id: str) -> tuple[Asset, ...]:
    """List one sample's existing manifest records; never infer target support."""
    models = [row for row in _models(group) if row['id'] == sample_id]
    if len(models) != 1:
        raise ValueError(f'Expected one manifest sample {group}:{sample_id}, found {len(models)}.')
    result = []
    filenames = set()
    for row in models[0].get('assets', []):
        filename = row['filename']
        path = PurePosixPath(filename)
        if path.is_absolute() or '..' in path.parts or '\\' in filename or ':' in filename:
            raise ValueError(f'Unsafe manifest asset filename: {filename!r}.')
        if filename in filenames:
            raise ValueError(f'Duplicate asset filename in {group}:{sample_id}: {filename}.')
        filenames.add(filename)
        result.append(Asset(group, sample_id, filename, row['format'], row.get('url'), row.get('sha256')))
    return tuple(result)


def resolve_asset(reference: str) -> Asset:
    """Resolve an exact qualified reference, rejecting unknown or ambiguous data."""
    parts = reference.split(':', 2)
    if len(parts) != 3:
        raise ValueError('Asset reference must be group:sample:filename.')
    group, sample, filename = parts
    matches = [asset for asset in list_assets(group, sample) if asset.filename == filename]
    if len(matches) != 1:
        raise ValueError(f'Unknown manifest asset {reference!r}.')
    return matches[0]


def verify_asset_file(asset: Asset, path: Path) -> str:
    """Check a local file against a recorded publisher hash when one exists.

    Returns the observed SHA-256. An observation without an expected publisher
    hash is byte identity only, not proof of official origin or compatibility.
    """
    import hashlib
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or empty model file: {path}.')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    observed = digest.hexdigest()
    if asset.sha256 is not None and observed != asset.sha256.lower():
        raise ValueError(f'SHA-256 mismatch for {asset.reference}: {path}.')
    return observed


def download_asset(asset: Asset, destination: Path) -> str:
    """Explicitly fetch one asset into a temporary file and atomically finish it.

    Existing files are verified, never overwritten. Failed/empty downloads and
    digest mismatches cannot leave a new complete-looking model file.
    """
    import logging
    import os
    import shutil
    import tempfile
    import urllib.request
    from urllib.parse import urlsplit

    destination = Path(destination)
    if destination.exists():
        return verify_asset_file(asset, destination)
    if not asset.url or urlsplit(asset.url).scheme not in ('https', 'http'):
        raise ValueError(f'No HTTP(S) model source for {asset.reference}.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            with urllib.request.urlopen(asset.url, timeout=60) as source:
                headers = getattr(source, 'headers', {})
                declared_length = headers.get('Content-Length')
                shutil.copyfileobj(source, output)
                if declared_length is not None and output.tell() != int(declared_length):
                    raise ValueError(f'Download length mismatch for {asset.reference}.')
        observed = verify_asset_file(asset, temporary_path)
        # Link installation is atomic and cannot replace an existing path.
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            return verify_asset_file(asset, destination)
        if asset.sha256 is None:
            logging.getLogger(__name__).warning('No publisher SHA-256 recorded for %s; observed digest does not verify origin.', asset.reference)
        return observed
    finally:
        temporary_path.unlink(missing_ok=True)
