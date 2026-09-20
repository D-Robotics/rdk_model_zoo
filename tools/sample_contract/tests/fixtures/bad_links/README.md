# bad_links (fixture)

Negative fixture: broken relative file links, a broken image, and a broken
intra-document fragment. The external https link is out of checker scope by
design (no network access).

<a id="overview"></a>
## Overview

Fixture overview.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | fixture1 | python | supported-not-run |

<a id="prerequisites"></a>
## Prerequisites

Host Python only.

<a id="quickstart"></a>
## Quick start

See the [missing model guide](model/README.md) and the
![broken photo](test_data/missing.png). The [external site](https://example.com)
is not checked. The [valid fragment](#overview) resolves. The
[broken fragment](#no-such-anchor) does not.

<a id="expected-results"></a>
## Expected results

Prints a list.

<a id="directory"></a>
## Directory

- `README.md` — this file

<a id="entry-points"></a>
## Entry points

This fixture has no sub-entries.

<a id="license"></a>
## License

Repository LICENSE.
