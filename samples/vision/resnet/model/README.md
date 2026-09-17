# ResNet18 model artifacts

The model directory contains no checked-in binaries. The canonical downloader
resolves the existing platform release manifests and writes one selected
artifact into this directory. It is the only maintained URL/format/hash path
for the canonical sample.

## Select and download

The supported references are:

| Target | Manifest reference | Destination under this directory |
| --- | --- | --- |
| X5 | `x5:resnet:resnet18_224x224_nv12.bin` | `resnet18_224x224_nv12.bin` |
| S100 | `s:resnet18:s100/resnet18_224x224_nv12.hbm` | `s100/resnet18_224x224_nv12.hbm` |
| S600 | `s:resnet18:s600/resnet18_224x224_nv12.hbm` | `s600/resnet18_224x224_nv12.hbm` |

From the repository root, download one target explicitly:

```bash
python3 samples/vision/resnet/model/download.py --target x5
python3 samples/vision/resnet/model/download.py --target s100
python3 samples/vision/resnet/model/download.py --target s600
```

The shell form is equivalent:

```bash
bash samples/vision/resnet/model/download.sh s100
```

The Python module is safe to import on a host without `hbm_runtime`; network
access starts only inside `download_target`. It calls
`samples._shared.assets.resolve_asset`, so the URL and file format are read
from `platforms/x5/docs/release/models.yaml` or
`platforms/s/docs/release/models.yaml` rather than duplicated here.

## Download and verification behavior

`download_target` creates the target subdirectory when needed, writes to a
same-directory temporary file, checks the content length and recorded
publisher SHA-256, and installs with an atomic non-overwriting link. Existing
files are verified and never silently replaced. A missing publisher hash is
reported as an origin limitation; the observed digest is still printed for
local evidence. Failed or empty downloads leave no complete-looking artifact.

The downloader accepts only the three target keys above. S100P has no ResNet18
row in the S manifest and cannot be made valid by reusing the S100 file. The
runtime also requires the exact qualified reference alongside a custom
`--model-path`, which prevents a `.bin` or `.hbm` basename from selecting the
wrong packed/split input protocol.

## Compatibility paths

Existing platform commands remain available and delegate here:

```bash
(cd platforms/x5/samples/vision/resnet/model && bash download.sh)
(cd platforms/s/samples/vision/resnet18/model && bash download_model.sh s100)
(cd platforms/s/samples/vision/resnet18/model && bash download_model.sh s600)
```

Their old output layouts are preserved so old scripts can be compared with the
canonical flow. They no longer carry a second hard-coded URL registry.

After downloading, use the exact reference and generated path in the runtime
command. For example:

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm
```

If a download fails, check the manifest row's URL and network access, remove
only an incomplete `.part` file if one remains, and retry. If an existing file
fails verification, preserve it for investigation and choose a new destination;
the downloader intentionally refuses to overwrite it. A hash warning is
expected for the current rows because their publisher SHA-256 fields are
unrecorded.
