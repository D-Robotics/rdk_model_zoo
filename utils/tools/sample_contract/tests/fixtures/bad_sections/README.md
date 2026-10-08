# bad_sections (fixture)

Negative fixture: duplicate `overview` anchor, missing `quickstart`, and
`support-matrix` appended out of template order.

<a id="overview"></a>
## Overview

Fixture overview anchor, first occurrence.

<a id="overview"></a>
## Overview (again)

Duplicate anchor id, must be reported.

<a id="prerequisites"></a>
## Prerequisites

Host Python only.

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

<a id="support-matrix"></a>
## Support matrix

Appended at the tail instead of its template position; the required order
is violated and `quickstart` is missing entirely.
