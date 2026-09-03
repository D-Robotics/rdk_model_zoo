# Release manifest

`models.yaml` is the model artifact inventory for the release identified in its `release` section. It records the artifacts exposed by each sample's model download script; it does not replace the sample README or certify runtime behavior.

## Schema

- `schema_version`: manifest format version.
- `release`: platform version, tag, source branch, compatibility baseline, and validation scope.
- `models`: one entry for every sample included in the release's top-level model list.
- `download_scripts`: repository-relative scripts that obtain the listed artifacts.
- `availability`: `download` when the repository publishes a download path, or `manual` when users must provide the model themselves.
- `assets`: model files used by the sample.
- `sha256`: lowercase SHA256 digest. A YAML `null` means the current repository does not record a trusted digest for that artifact; it never means that integrity was verified.

## Maintenance rules

1. Update the manifest in the same pull request that adds, removes, renames, or relocates a downloadable model.
2. Keep URLs and filenames consistent with the referenced download scripts.
3. Record a SHA256 digest when the published artifact is stable and the digest has been verified by the maintainer.
4. Do not invent a digest or copy one from an untrusted source. Leave it as `null` and disclose incomplete checksum coverage in the release notes.
5. Freeze the manifest with the platform release tag. Later corrections require a patch release; published tags are not moved.

The X5 manifest includes every sample in the release's top-level model list, including samples with manual model provisioning and assets without a repository-maintained checksum.
