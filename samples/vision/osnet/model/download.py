"""Prepare an OSNet model from a checked local candidate or release manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.osnet.model.artifacts import load_manifest, sha256


def _write_selected(read, selected):
    # Validate all selected bytes and existing destinations before writing.
    verified = []
    for source, target, expected in selected:
        data = read(source)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Archive entry hash mismatch: {source}")
        if target.exists() and sha256(target) != expected:
            raise FileExistsError(f"Refusing to replace different local content: {target}")
        verified.append((target, data))
    for target, data in verified:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        print(f"Verified {target}")


def install_bundle(bundle, destination, variant, *, fixtures=None):
    """Verify the pinned bundle and copy only the selected model/fixtures."""
    manifest = load_manifest()
    if sha256(bundle) != manifest["bundle"]["sha256"]:
        raise ValueError("Delivery ZIP SHA-256 differs from the recorded experiment")
    record = manifest["artifacts"][variant]
    with zipfile.ZipFile(bundle) as archive:
        checksums = dict((name, digest) for digest, name in
                         (line.split("  ", 1) for line in archive.read("SHA256SUMS").decode().splitlines()))
        selected = [(f"models/{record['filename']}", Path(destination) / record["filename"], record["sha256"])]
        if fixtures is not None:
            for name in ("images.dat", "manifest.json", f"{variant}.f32"):
                source = f"cpp-fixture/{name}"
                selected.append((source, Path(fixtures) / name, checksums[source]))
        _write_selected(archive.read, selected)


def install_readback(bundle, destination, variant, *, fixtures=None):
    """Import only the selected files from the verified read-only X5 capture."""
    manifest = load_manifest()
    if sha256(bundle) != manifest["readback"]["sha256"]:
        raise ValueError("Readback TAR SHA-256 differs from the recorded X5 copy")
    receipt_path = Path(__file__).resolve().parent / manifest["readback"]["receipt"]
    receipt = json.loads(receipt_path.read_text())
    record = manifest["artifacts"][variant]
    selected = [(receipt["models"][variant]["source_path"], Path(destination) / record["filename"], record["sha256"])]
    if fixtures is not None:
        for name in ("images.dat", "manifest.json", f"{variant}.f32"):
            item = receipt["fixtures"][name]
            selected.append((item["source_path"], Path(fixtures) / name, item["sha256"]))
    with tarfile.open(bundle) as archive:
        def read(name):
            entry = archive.getmember(name)
            if not entry.isfile():
                raise ValueError(f"Expected a regular archive entry: {name}")
            return archive.extractfile(entry).read()
        _write_selected(read, selected)


def install_candidate(candidate, destination, variant):
    """Copy one target-specific candidate model, verifying release and bytes.

    Args:
        candidate: A platform bundle directory or its parent delivery directory.
        destination: Model output directory, which may initially be absent.
        variant: Exact variant registered in artifacts.json.

    Raises:
        ValueError: The release platform, artifact identity or bytes differ.
        FileExistsError: A different model already occupies the destination.
    """
    record = load_manifest()["artifacts"][variant]
    target = record.get("target", "x5")
    folder = Path(candidate)
    if not (folder / "release.json").is_file():
        folder = folder / target
    release = json.loads((folder / "release.json").read_text(encoding="utf-8"))
    if release.get("platform") != target:
        raise ValueError(f"Candidate platform must be {target}")
    artifact = release.get("artifact", {})
    filename = artifact.get("filename", "")
    if not filename or Path(filename).name != filename or artifact.get("sha256") != record["sha256"]:
        raise ValueError("Candidate artifact differs from the registered model")
    _write_selected(lambda name: (folder / name).read_bytes(), [
        (filename, Path(destination) / record["filename"], record["sha256"]),
    ])


def download_published(destination, variant):
    """Download only the exact URL and checksum in the native release manifest."""
    from utils.py_utils.assets import download_asset, resolve_asset

    record = load_manifest()["artifacts"][variant]
    target = record.get("target", "x5")
    group = "x5" if target == "x5" else "s"
    filename = record["filename"] if target == "x5" else f"{target}/{record['filename']}"
    asset = resolve_asset(f"{group}:osnet:{filename}")
    if asset.sha256 != record["sha256"]:
        raise ValueError("Release manifest differs from the registered model hash")
    if not asset.url:
        raise ValueError("OSNet has no published model URL yet; use --candidate-dir with an existing delivery")
    output = Path(destination) / record["filename"]
    download_asset(asset, output)
    print(f"Verified {output}")


def main(argv=None):
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-dir", type=Path, help="Existing platform bundle or accepted-delivery directory; model only")
    source.add_argument("--download", action="store_true", help="Download from the native release manifest once its public URL is available")
    source.add_argument("--bundle", type=Path, help="Existing archive; select its type with --format")
    source.add_argument("--from-server", action="store_true", help="Explicitly fetch the selected archive with scp")
    parser.add_argument("--format", choices=("delivery", "readback"), default="delivery", help="Original delivery ZIP or read-only X5 TAR")
    parser.add_argument("--ssh-host", default=None, help="Override the recorded SSH source host")
    parser.add_argument("--remote-path", default=None, help="Override the recorded absolute archive path")
    parser.add_argument("--jump-host", default=None, help="Optional SSH ProxyJump host")
    parser.add_argument("--target", choices=tuple(manifest["targets"]), default="x5", help="Model target; preparation does not require board hardware")
    parser.add_argument("--variant", choices=tuple(manifest["artifacts"]), default=None, help="Model variant; defaults to the recommended variant for --target")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent, help="Local model directory")
    parser.add_argument("--fixtures", action="store_true", help="Also prepare the original fixed-input C++ fixtures")
    args = parser.parse_args(argv)
    if args.variant is None:
        args.variant = next(name for name, record in manifest["artifacts"].items()
                            if record.get("target", "x5") == args.target and record.get("recommended"))
    if manifest["artifacts"][args.variant].get("target", "x5") != args.target:
        parser.error("--variant belongs to a different --target")
    if args.candidate_dir or args.download:
        if args.fixtures:
            parser.error("Candidate and published preparation copy models only; prepare evaluation fixtures separately")
        try:
            if args.candidate_dir:
                install_candidate(args.candidate_dir, args.output_dir, args.variant)
            else:
                download_published(args.output_dir, args.variant)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return 0
    archive_record = manifest["readback" if args.format == "readback" else "bundle"]
    install = install_readback if args.format == "readback" else install_bundle
    ssh_host = args.ssh_host or archive_record["source_host"]
    remote_path = args.remote_path or archive_record["source_path"]
    fixtures = Path(__file__).resolve().parents[1] / "test_data" / "fixture" if args.fixtures else None
    if args.bundle:
        install(args.bundle, args.output_dir, args.variant, fixtures=fixtures)
        return 0
    if ssh_host.startswith("-") or not remote_path.startswith("/"):
        parser.error("Use an SSH host name and an absolute remote path")
    with tempfile.TemporaryDirectory(prefix="osnet-delivery-") as directory:
        bundle = Path(directory) / archive_record["filename"]
        command = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if args.jump_host:
            command += ["-J", args.jump_host]
        command += [f"{ssh_host}:{remote_path}", str(bundle)]
        subprocess.run(command, check=True)
        install(bundle, args.output_dir, args.variant, fixtures=fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
