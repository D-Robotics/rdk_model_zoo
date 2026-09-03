# RDK Model Zoo Release Policy

This document defines the manual release process for the RDK Model Zoo. It is intentionally small: the current phase has no CI gate and no board test gate.

## 1. Version lines and names

The X5, S, and X3 lines are versioned independently. Each line uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`.

| Platform | Release branch | Stable tag | Release example |
| --- | --- | --- | --- |
| X5 | `rdk_x5` | `x5-vMAJOR.MINOR.PATCH` | `x5-v1.0.0` |
| S | `rdk_s` | `s-vMAJOR.MINOR.PATCH` | `s-v1.0.0` |
| X3 | `rdk_x3` | `x3-vMAJOR.MINOR.PATCH` | `x3-v1.0.0` |

Release candidates append `-rc.N`, for example `x5-v1.0.0-rc.1`. The `VERSION` file on a platform branch contains the unprefixed version, such as `1.0.0`.

Do not use a common repository-wide version for these platform lines. A tag identifies both the platform and the exact commit from which the release was built.

## 2. Required release files

Every release commit must update or verify all of the following files:

- `VERSION` — the platform version.
- `CHANGELOG.md` — user-visible changes and known limitations.
- `release/models.yaml` — the model manifest for the release.
- `docs/releases/<tag>.md` — the release notes used by GitHub Releases.

The manifest records the models and assets exposed by the release, their sample paths, download scripts or URLs, formats, and checksums when available. An unknown SHA-256 value must be written as `null`; it must not be guessed. The release notes must disclose that checksum coverage is incomplete whenever any manifest entry has `sha256: null`.

The manifest describes the published source inventory. It does not certify model accuracy, runtime behavior, or board compatibility.

## 3. Manual release procedure

1. Select one platform branch and confirm that the release scope belongs to that platform.
2. Update `VERSION`, `CHANGELOG.md`, `release/models.yaml`, and `docs/releases/<tag>.md`.
3. Review the manifest manually: sample paths and download scripts must exist, URLs must be correct, and unknown checksums must be `null`. Check that the tag, branch, platform, and version agree in the release files.
4. Review the source diff and run `git diff --check`. Confirm the working tree is clean and that the intended tag does not already exist.
5. Merge the reviewed release commit into the platform branch. The tag must point to that branch tip.
6. Create and push an annotated tag:

   ```bash
   git switch rdk_x5
   git pull --ff-only origin rdk_x5
   git tag -a x5-v1.0.0 -m "RDK Model Zoo X5 v1.0.0"
   git push origin x5-v1.0.0
   ```

   Replace the branch, tag, and message for the S or X3 line. Do not create a lightweight tag.

7. Create the GitHub Release from the pushed tag, use the matching release-notes file, and attach the manifest as `models.yaml`:

   ```bash
   gh release create x5-v1.0.0 "release/models.yaml#models.yaml" \
     --title "RDK Model Zoo X5 v1.0.0" \
     --notes-file docs/releases/x5-v1.0.0.md \
     --verify-tag
   ```

8. Verify that the GitHub Release, attached manifest, tag, branch commit, `VERSION`, and repository manifest all refer to the same platform version. Record the release URL and commit in the change log or release record when the project workflow requires it.

## 4. Tag and release immutability

After publication, an annotated tag must never be moved, force-pushed, or deleted. Do not reuse a released version for a different commit. If a release contains an error, keep the original tag, mark the GitHub Release as withdrawn or superseded, and publish a new patch version with corrected files and release notes.

An archive tag preserves a historical line or snapshot and uses `archive/<platform>-v<version>`, for example `archive/x5-v0.0.1`. Archive tags are annotated and immutable as well.

## 5. Patch, withdrawal, and prerequisites

A patch release fixes documentation, download metadata, scripts, or other backward-compatible release defects. It repeats the required-file review and receives a new patch tag, such as `x5-v1.0.1`.

To withdraw a release, explain the reason in the GitHub Release notes, retain the published tag for traceability, and publish a replacement version when users need a corrected artifact. A withdrawal does not silently rewrite history.

This simplified policy does not require automated CI, self-hosted runners, benchmark jobs, or RDK board testing before publication. Release notes must not claim that the full model set passed board testing. Any tests or manual checks that were actually performed may be listed explicitly with their scope.
