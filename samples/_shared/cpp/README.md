# Shared native identity and digest helpers

[简体中文](README_cn.md)

These C++17 helpers are used by native samples without Python, OpenCV, board
SDK or crypto-library dependencies. Include this directory in your compiler's
search path. Link `platform_identity.cc` when reading actual local identity;
the SHA-256 helper and identity matching functions are header-only.

| API | Contract |
| --- | --- |
| `rdk::sha256_hex(data, size)` | SHA-256 of exactly `size` readable bytes; lowercase 64-character digest; input is borrowed only during the call |
| `rdk::sha256_file(path)` | Stream a regular file in 64 KiB blocks; empty string means open/read/type failure; an empty regular file has a valid nonempty digest |
| `rdk::identify_target(NativeIdentity)` | Match observed strings; returns `x5`, `s100`, `s100p`, `s600`, or empty when unknown |
| `rdk::read_native_identity` | Read fixed local sysfs/device-tree files into owned strings; no SSH, networking, environment or CLI identity override |

Identity matching follows [platforms.json](../../../docs/release/platforms.json)
and the shared [Python implementation](../platforms.py). A nonempty SoC name
has priority, including the S100 + S100P board-type refinement. Only when it is
missing does socinfo apply, then the exact device-tree model. Unknown higher
priority data does not fall back to a lower-priority X5 alias. SoC/board strings
are case-insensitive; the device-tree model match is case-sensitive. Identity
recognition does not establish asset compatibility or board-test results.

The streaming SHA implementation was extracted from YOLOv5's native run-evidence
writer, which delegates to it. Read failures and directory/device paths must
not be mistaken for the digest of empty content. A matching hash proves byte
identity relative to the expected digest; an observed hash without a publisher
checksum is not publisher authentication or conversion-toolchain provenance.
Sample preflight remains responsible for rejecting an empty model.

Host checks in [YOLOE tests](../../vision/yoloe/tests/test_cpp_preflight.py)
compare all registered aliases and precedence against Python, and compare
binary/padding/read-boundary hashes with `hashlib`. Native preflight additionally
checks published SHA test vectors, including one million `a` bytes. These are
host checks; neither helper performs inference or certifies a board SDK.
