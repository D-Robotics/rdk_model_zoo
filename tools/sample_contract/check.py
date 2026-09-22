#!/usr/bin/env python3
"""Static contract checker for the unified samples layout (Phase 0.5, Q3).

The checker reads source files and documentation only.  It never downloads,
never loads a board SDK, and never executes README code blocks.  The single
exception is ``--parser-mode import`` (the default): it imports the sample's
``runtime/python/main.py`` and calls its side-effect-free ``build_parser()``
to compare documented CLI defaults against the real parser.  That import is
allowed for trusted repository code only; any import failure is recorded as
a *skipped* check with the reason, never as a pass.

Rules (IDs are decoupled from section IDs; see readme-contract §8):

  R-README-PAIR       bilingual pairing per level (README.md + README_cn.md)
  R-README-SECTIONS   fixed anchor IDs present, unique, in template order
  R-README-LINKS      relative links/images resolve to existing files and
                      intra-document fragments resolve to explicit anchors
  R-CLI-DEFAULTS      README parameter tables match the real parser
  R-I18N-PARAMS       en/zh parameter tables agree with each other
  R-STAGE-PURITY      pre/forward/post-style functions contain no download,
                      save/write, subprocess, or destructive calls (AST)
  R-EXEMPTION         exemption bookkeeping (unknown/unused entries fail)

Canonical default-value forms compared between the README ``Default`` column
and ``build_parser``:

  parser ``None``      -> README ``null`` or ``none`` (case-insensitive)
  parser ``True``/``False`` -> ``true`` / ``false``
  parser list/tuple    -> JSON form, e.g. ``[0]`` or ``[0, 1]``
  parser scalars       -> literal string
  absolute paths under the repository (or cwd) are compared repo-relative

Backticks and surrounding quotes in README cells are stripped first.

What this tool deliberately does NOT decide (goes to semantic review per the
plan): whether prose answers the contract's must-answer questions, whether a
``predict`` implementation duplicates stage logic, whether appended sections
are top-level or sub-sections, board behavior, and conversion correctness.

Usage:
  python3 tools/sample_contract/check.py --sample samples/vision/resnet
  python3 tools/sample_contract/check.py --scope migration --report out.json
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Optional, Sequence
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]

RULE_PAIR = "R-README-PAIR"
RULE_SECTIONS = "R-README-SECTIONS"
RULE_LINKS = "R-README-LINKS"
RULE_CLI = "R-CLI-DEFAULTS"
RULE_I18N = "R-I18N-PARAMS"
RULE_PURITY = "R-STAGE-PURITY"
RULE_EXEMPTION = "R-EXEMPTION"
RULE_SCOPE = "R-SCOPE"

# Levels: (directory, template stem, required pairing)
README_LEVELS: tuple[tuple[str, str], ...] = (
    ("", "sample"),
    ("model", "model"),
    ("runtime/python", "runtime-python"),
    ("runtime/cpp", "runtime-cpp"),
    ("conversion", "conversion"),
    ("evaluator", "evaluator"),
)

ANCHOR_RE = re.compile(r"""<a\s+id=["']([a-z0-9-]+)["']\s*/?>""")
LINK_RE = re.compile(r"""(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)""")
OPTION_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

# Stage functions whose bodies are purity-checked (inference-contract §1/§3).
STAGE_EXACT = {"pre_process", "forward", "post_process", "predict"}
STAGE_PREFIXES = ("forward_", "pre_process_", "post_process_", "run_")

# Files inside runtime/python that are exempt by policy, with the recorded
# reason.  These are reported as skips, never silently ignored.
POLICY_SKIPPED_FILES = {
    "main.py": "CLI layer: saving output and argument handling live here",
    "legacy.py": "documented compatibility shim (inference-contract §4)",
}

DENY_CALLS: dict[str, set[str]] = {
    "download/network": {
        "urlretrieve", "urlopen", "urlretrieve", "requests.get",
        "requests.post", "requests.put", "requests.delete",
        "httpx.get", "httpx.post", "wget", "curl",
        "hf_hub_download", "snapshot_download",
        "socket.socket", "socket.create_connection",
    },
    "file write/save": {
        "cv2.imwrite", "np.save", "np.savez", "numpy.save", "numpy.savez",
        "json.dump", "yaml.dump", "open",
    },
    "subprocess": {
        "subprocess.run", "subprocess.call", "subprocess.check_call",
        "subprocess.check_output", "subprocess.Popen",
        "os.system", "os.popen", "os.execv",
    },
    "destructive fs": {
        "os.remove", "os.unlink", "os.rmdir", "shutil.rmtree",
        "shutil.move", "shutil.copy",
    },
    "dynamic execution": {"eval", "exec"},
}
DENY_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".imwrite", "file write/save"),
    (".savez", "file write/save"),
    (".savefig", "file write/save"),
    (".to_csv", "file write/save"),
    (".write_bytes", "file write/save"),
    (".write_text", "file write/save"),
    (".urlretrieve", "download/network"),
)
WRITE_MODE_CHARS = set("wax+")


@dataclass
class Finding:
    """One violation with a rule ID and a repo-relative location."""

    rule: str
    path: str
    line: int
    message: str

    def as_dict(self) -> dict:
        return {"rule": self.rule, "path": self.path, "line": self.line,
                "message": self.message}


@dataclass
class Skip:
    """A check that could not run, with the reason.  Never counts as pass."""

    rule: str
    path: str
    reason: str

    def as_dict(self) -> dict:
        return {"rule": self.rule, "path": self.path, "reason": self.reason}


@dataclass
class SampleReport:
    """Findings and skips for one sample directory."""

    path: str
    findings: list[Finding] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)
    exemptions_applied: list[dict] = field(default_factory=list)

    def add(self, rule: str, path: Path, line: int, message: str) -> None:
        self.findings.append(
            Finding(rule, display_path(path), line, message))

    def skip(self, rule: str, path: Path, reason: str) -> None:
        self.skips.append(Skip(rule, display_path(path), reason))


def display_path(path: Path) -> str:
    """Prefer repo-relative posix paths; fall back to the given form."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_template_anchors(templates_dir: Path) -> dict[str, tuple[str, ...]]:
    """Extract the ordered required anchor IDs from the en/zh templates."""

    anchors: dict[str, tuple[str, ...]] = {}
    for stem in {stem for _, stem in README_LEVELS}:
        ids: list[str] = []
        for suffix in ("en", "zh"):
            template = templates_dir / f"{stem}.{suffix}.md"
            if not template.is_file():
                raise SystemExit(
                    f"sample-contract: missing template {template}")
            text = template.read_text(encoding="utf-8")
            file_ids = [
                match.group(1)
                for match in ANCHOR_RE.finditer(text)
                # The literal "…" anchor inside the guidance blockquote is
                # an example, not a section; the ID charset filter drops it.
            ]
            if len(set(file_ids)) != len(file_ids):
                raise SystemExit(
                    f"sample-contract: duplicate anchor id in {template}")
            if suffix == "en":
                ids = file_ids
            elif tuple(file_ids) != tuple(ids):
                raise SystemExit(
                    f"sample-contract: en/zh template anchor mismatch for "
                    f"{template.name}: {ids} vs {file_ids}")
        anchors[stem] = tuple(ids)
    return anchors


def extract_anchors(text: str) -> list[tuple[str, int]]:
    """Return [(anchor id, 1-based line)] for one markdown file."""

    results: list[tuple[str, int]] = []
    for match in ANCHOR_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        results.append((match.group(1), line))
    return results


def check_sections(report: SampleReport, readme: Path,
                   required: Sequence[str]) -> bool:
    """R-README-SECTIONS: presence, uniqueness, and template order."""

    text = readme.read_text(encoding="utf-8")
    anchors = extract_anchors(text)
    seen: dict[str, int] = {}
    present_ids: list[str] = []
    for anchor_id, line in anchors:
        if anchor_id in seen:
            report.add(
                RULE_SECTIONS, readme, line,
                f"duplicate anchor id {anchor_id!r} "
                f"(first at line {seen[anchor_id]})")
            continue
        seen[anchor_id] = line
        present_ids.append(anchor_id)

    for anchor_id in required:
        if anchor_id not in seen:
            report.add(
                RULE_SECTIONS, readme, 0,
                f"missing required section anchor {anchor_id!r}")

    ordered = [anchor_id for anchor_id in present_ids
               if anchor_id in required]
    expected = [anchor_id for anchor_id in required
                if anchor_id in seen]
    if ordered != expected:
        first_line = seen.get(ordered[0], 1) if ordered else 1
        report.add(
            RULE_SECTIONS, readme, first_line,
            f"required section order deviates from template "
            f"(expected: {', '.join(expected)}; got: {', '.join(ordered)})")
    return bool(anchors)


def check_links(report: SampleReport, readme: Path,
                anchor_ids: set[str]) -> None:
    """R-README-LINKS: local targets and fragments must resolve."""

    text = readme.read_text(encoding="utf-8")
    for match in LINK_RE.finditer(text):
        target = unquote(match.group(2)).strip()
        line = text.count("\n", 0, match.start()) + 1
        if SCHEME_RE.match(target) or target.startswith("//"):
            continue  # external links are outside static scope
        file_part, _, fragment = target.partition("#")
        if not file_part:
            if fragment and fragment not in anchor_ids:
                report.add(
                    RULE_LINKS, readme, line,
                    f"intra-document link #{fragment} does not match any "
                    f"explicit anchor id")
            continue
        resolved = (readme.parent / file_part)
        if not resolved.exists():
            report.add(
                RULE_LINKS, readme, line,
                f"local link target does not exist: {file_part}")


def split_table_row(line: str) -> list[str]:
    """Split one markdown table row into trimmed cells."""

    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell) and any(
        cell for cell in cells)


DEFAULT_COLUMN_NAMES = {"default", "默认值", "默认"}


def extract_parameter_table(
    readme: Path,
) -> Optional[tuple[dict[str, str], dict[str, int], int]]:
    """Read the table following the ``parameters`` anchor.

    Returns ``(option -> canonical default, option -> line, header line)``,
    or ``None`` when the parameters section (and therefore its table) is
    absent — the section rule already covers that case.
    """

    text = readme.read_text(encoding="utf-8")
    position = text.find('<a id="parameters"></a>')
    if position < 0:
        return None
    anchor_line = text.count("\n", 0, position) + 1
    table_rows: list[tuple[int, list[str]]] = []
    in_table = False
    for offset, line in enumerate(text[position:].splitlines()):
        row_line = anchor_line + offset
        if line.strip().startswith("|"):
            in_table = True
            table_rows.append((row_line, split_table_row(line)))
            continue
        if in_table:
            break
    if len(table_rows) < 2:
        return {}, {}, 1

    header_line, header = table_rows[0]
    default_column: Optional[int] = None
    for index, cell in enumerate(header):
        if cell.lower() in DEFAULT_COLUMN_NAMES:
            default_column = index
            break
    options: dict[str, str] = {}
    option_lines: dict[str, int] = {}
    for row_line, cells in table_rows[1:]:
        if is_separator_row(cells):
            continue
        if not cells or not cells[0]:
            continue
        found = OPTION_RE.findall(cells[0])
        if not found:
            continue
        default_value = ""
        if default_column is not None and default_column < len(cells):
            default_value = cells[default_column]
        canonical = canon_readme_default(default_value)
        for option in found:
            options[option] = canonical
            option_lines[option] = row_line
    return options, option_lines, header_line


def canon_readme_default(cell: str) -> str:
    """Canonicalize a README Default cell for comparison."""

    value = cell.strip().strip("`").strip('"').strip()
    if value.lower() in {"null", "none"}:
        return "null"
    return value


def canon_parser_default(value: object) -> str:
    """Canonicalize a parser default for comparison with the README form."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("/"):
            absolute = Path(stripped)
            for base in (REPO_ROOT, Path.cwd()):
                try:
                    return absolute.relative_to(base).as_posix()
                except ValueError:
                    continue
        return stripped
    return str(value)


_IMPORT_FAILURES: dict[str, BaseException] = {}


def import_build_parser(main_py: Path) -> argparse.ArgumentParser:
    """Import ``main.py`` and call ``build_parser()``; return the parser.

    Failures propagate to the caller, which records them as skips.  A failed
    import is remembered so repeated checks (one per language) report the
    same original reason instead of secondary AttributeError noise.
    """

    key = str(main_py)
    if key in _IMPORT_FAILURES:
        raise _IMPORT_FAILURES[key]
    module_name = "_sample_contract_" + re.sub(
        r"[^a-z0-9_]", "_", key.lower())
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, main_py)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"cannot load spec for {main_py}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except BaseException as exc:
            _IMPORT_FAILURES[key] = exc
            sys.modules.pop(module_name, None)
            raise
        sys.modules[module_name] = module
    module = sys.modules[module_name]
    build = getattr(module, "build_parser", None)
    if not callable(build):
        raise AttributeError(f"{main_py} does not define build_parser()")
    return build()


def collect_parser_defaults(
    parser: argparse.ArgumentParser,
) -> dict[str, str]:
    """Flatten option strings to canonical defaults, subparsers included."""

    defaults: dict[str, str] = {}

    def walk(target: argparse.ArgumentParser) -> None:
        for action in target._actions:  # noqa: SLF001 - stable CPython API
            if isinstance(action, argparse._SubParsersAction):
                for subparser in action.choices.values():
                    walk(subparser)
                continue
            for option in action.option_strings:
                if option in {"-h", "--help"}:
                    continue
                defaults[option] = canon_parser_default(action.default)

    walk(parser)
    return defaults


def check_cli_defaults(report: SampleReport, sample_dir: Path,
                       readme: Path, parser_mode: str) -> Optional[dict]:
    """R-CLI-DEFAULTS: compare one language's table with the real parser."""

    table = extract_parameter_table(readme)
    if table is None:
        report.skip(RULE_CLI, readme,
                    "no parameters section (see R-README-SECTIONS)")
        return None
    options, option_lines, header_line = table
    if not options:
        report.add(
            RULE_CLI, readme, header_line,
            "parameters table could not be parsed (option column or rows)")
        return None

    main_py = sample_dir / "runtime" / "python" / "main.py"
    if parser_mode == "static":
        report.skip(
            RULE_CLI, readme,
            "static parser mode: build_parser not executed (no pass implied)")
        return options
    if not main_py.is_file():
        report.skip(RULE_CLI, readme,
                    f"no runtime/python/main.py to import ({main_py} absent)")
        return options
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        parser = import_build_parser(main_py)
        defaults = collect_parser_defaults(parser)
    except BaseException as exc:  # noqa: BLE001 - recorded, not raised
        report.skip(
            RULE_CLI, readme,
            f"parser import failed ({type(exc).__name__}: {exc}); "
            f"defaults not verified")
        return options

    language = "zh" if readme.name == "README_cn.md" else "en"
    for option, documented in sorted(options.items()):
        if option not in defaults:
            report.add(
                RULE_CLI, readme, option_lines.get(option, header_line),
                f"[{language}] documents option {option} absent from the "
                f"parser (default written as {documented!r})")
    for option, actual in sorted(defaults.items()):
        if option not in options:
            report.add(
                RULE_CLI, readme, header_line,
                f"[{language}] parser option {option} (default {actual!r}) "
                f"is not documented")
            continue
        if options[option] != actual:
            report.add(
                RULE_CLI, readme, option_lines.get(option, header_line),
                f"[{language}] default drift for {option}: parser "
                f"{actual!r} vs README {options[option]!r}")
    return options


def check_i18n_params(report: SampleReport, en_readme: Path,
                      zh_readme: Path, en_table: Optional[dict],
                      zh_table: Optional[dict]) -> None:
    """R-I18N-PARAMS: the two languages must document the same parameters."""

    if en_table is None or zh_table is None:
        return
    for option in sorted(set(en_table) - set(zh_table)):
        report.add(
            RULE_I18N, en_readme, 1,
            f"option {option} documented only in the English table")
    for option in sorted(set(zh_table) - set(en_table)):
        report.add(
            RULE_I18N, zh_readme, 1,
            f"option {option} documented only in the Chinese table")
    for option in sorted(set(en_table) & set(zh_table)):
        if en_table[option] != zh_table[option]:
            report.add(
                RULE_I18N, en_readme, 1,
                f"default mismatch for {option}: en {en_table[option]!r} "
                f"vs zh {zh_table[option]!r}")


def dotted_name(node: ast.AST) -> Optional[str]:
    """Reconstruct a dotted name like ``cv2.imwrite`` from an AST call."""

    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def classify_call(call: ast.Call) -> Optional[tuple[str, str]]:
    """Return ``(dotted, category)`` when a call crosses a purity boundary."""

    name = dotted_name(call.func) or ""
    if not name:
        return None
    for dotted in (name, name.rsplit(".", 1)[-1]):
        for category, denied in DENY_CALLS.items():
            if dotted in denied and dotted != "open":
                return name, category
    for suffix, category in DENY_SUFFIXES:
        if name.endswith(suffix):
            return name, category
    if name == "open" or name.endswith(".open"):
        mode: Optional[str] = None
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) \
                and isinstance(call.args[1].value, str):
            mode = call.args[1].value
        for keyword in call.keywords:
            if keyword.arg == "mode" and isinstance(keyword.value,
                                                    ast.Constant):
                mode = keyword.value.value
        if mode is not None and any(char in mode for char in WRITE_MODE_CHARS):
            return name, "file write/save"
    return None


def check_stage_purity(report: SampleReport, sample_dir: Path) -> None:
    """R-STAGE-PURITY: AST scan of stage functions under runtime/python."""

    runtime = sample_dir / "runtime" / "python"
    if not runtime.is_dir():
        return

    def is_stage(name: str) -> bool:
        return name in STAGE_EXACT or name.startswith(STAGE_PREFIXES)

    for py_file in sorted(runtime.glob("*.py")):
        if py_file.name in POLICY_SKIPPED_FILES:
            report.skip(
                RULE_PURITY, py_file,
                f"policy skip: {POLICY_SKIPPED_FILES[py_file.name]}")
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"),
                             filename=str(py_file))
        except SyntaxError as exc:
            report.add(
                RULE_PURITY, py_file, exc.lineno or 0,
                f"cannot parse module ({exc})")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not is_stage(node.name):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                verdict = classify_call(child)
                if verdict is None:
                    continue
                name, category = verdict
                report.add(
                    RULE_PURITY, py_file, child.lineno,
                    f"stage function {node.name}() calls {name}() — "
                    f"{category} boundary (inference-contract §3)")


def run_sample(sample_dir: Path, templates: dict[str, tuple[str, ...]],
               parser_mode: str) -> SampleReport:
    """Run every applicable rule against one sample directory."""

    report = SampleReport(path=display_path(sample_dir))

    en_tables: dict[str, Optional[dict]] = {}
    zh_tables: dict[str, Optional[dict]] = {}
    runtime_en: Optional[Path] = None
    runtime_zh: Optional[Path] = None
    for directory, stem in README_LEVELS:
        level_dir = sample_dir / directory if directory else sample_dir
        if not level_dir.is_dir():
            continue
        en_readme = level_dir / "README.md"
        zh_readme = level_dir / "README_cn.md"
        for readme, label in ((en_readme, "README.md"),
                              (zh_readme, "README_cn.md")):
            if not readme.is_file():
                report.add(
                    RULE_PAIR, level_dir, 1,
                    f"{label} missing for this level "
                    f"(readme-contract §2 bilingual pairing)")
        required = templates[stem]
        for readme in (en_readme, zh_readme):
            if not readme.is_file():
                continue
            check_sections(report, readme, required)
            file_anchors = {
                anchor_id
                for anchor_id, _ in extract_anchors(
                    readme.read_text(encoding="utf-8"))
            }
            check_links(report, readme, file_anchors)
        if stem == "runtime-python":
            for readme, store in ((en_readme, en_tables),
                                  (zh_readme, zh_tables)):
                if readme.is_file():
                    store[stem] = check_cli_defaults(
                        report, sample_dir, readme, parser_mode)
            if en_readme.is_file():
                runtime_en = en_readme
            if zh_readme.is_file():
                runtime_zh = zh_readme
    if runtime_en is not None and runtime_zh is not None:
        check_i18n_params(
            report, runtime_en, runtime_zh,
            en_tables.get("runtime-python"), zh_tables.get("runtime-python"))
    check_stage_purity(report, sample_dir)
    return report


def parse_progress_region(map_path: Path) -> list[list[str]]:
    """Return the data rows of the current-round progress table."""

    rows: list[list[str]] = []
    in_progress = False
    for line in map_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and "进度区" in line:
            in_progress = True
            continue
        if in_progress and line.startswith("## "):
            break
        if not in_progress or not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] in {"Batch", "---", "—"}:
            continue
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            continue
        if len(cells) >= 6:
            rows.append(cells)
    return rows


def resolve_migration_scope(map_path: Path, samples_root: Path) -> tuple[
        list[Path], list[Finding]]:
    """Resolve sample directories from the Refactor column (>= in-progress).

    The historical P0 S/F/H columns are never read for scope decisions.
    """

    findings: list[Finding] = []
    sample_names: list[str] = []
    if not map_path.is_file():
        findings.append(Finding(
            RULE_SCOPE, display_path(map_path), 0,
            f"migration map not found at {map_path}"))
        return [], findings
    for cells in parse_progress_region(map_path):
        sample_cell = cells[1]
        refactor = re.sub(r"[（(].*$", "", cells[4]).strip().lower()
        if refactor not in {"in-progress", "done"}:
            continue
        if sample_cell.startswith("~"):
            # Infrastructure rows (leading '~', per the map legend) track
            # tooling/contract deliverables, not sample directories; they are
            # excluded from the sample scope instead of failing resolution.
            continue
        name = re.split(r"[（(]", sample_cell)[0].strip()
        name = re.sub(r"\s*/\s*.*$", "", name).strip()
        if name:
            sample_names.append(name)

    paths: list[Path] = []
    for name in dict.fromkeys(sample_names):
        matches = sorted(
            domain / name
            for domain in sorted(samples_root.iterdir())
            if (domain / name).is_dir()
        ) if samples_root.is_dir() else []
        if not matches:
            findings.append(Finding(
                RULE_SCOPE, display_path(map_path), 0,
                f"progress row sample {name!r} has no directory under "
                f"{samples_root}"))
        elif len(matches) > 1:
            findings.append(Finding(
                RULE_SCOPE, display_path(map_path), 0,
                f"progress row sample {name!r} is ambiguous: "
                f"{[display_path(p) for p in matches]}"))
        else:
            paths.append(matches[0])
    return paths, findings


@dataclass
class Exemption:
    """A reviewed exception to one finding; requires a non-empty reason.

    ``message`` optionally pins the exemption to one exact finding
    message.  Baselines over rules that report several findings at the
    same line (R-README-SECTIONS reports every missing anchor at line 0)
    must set it: without it one entry tolerates any finding of the rule
    at that path/line, so fixing one anchor and breaking another would
    keep the gate green instead of surfacing new debt.
    """

    rule: str
    path: str
    line: int
    reason: str
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: dict, source: Path) -> "Exemption":
        missing = [key for key in ("rule", "path", "line", "reason")
                   if key not in raw]
        if missing or not str(raw.get("reason", "")).strip():
            raise SystemExit(
                f"sample-contract: malformed exemption in {source}: {raw}")
        return cls(str(raw["rule"]), str(raw["path"]), int(raw["line"]),
                   str(raw["reason"]),
                   str(raw["message"]) if "message" in raw else None)


def load_exemptions(path: Optional[Path]) -> list[Exemption]:
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("exemptions", raw) if isinstance(raw, dict) else raw
    return [Exemption.from_dict(entry, path) for entry in entries]


def apply_exemptions(report: SampleReport,
                     exemptions: list[Exemption],
                     matched_indexes: Optional[set] = None) -> list[Finding]:
    """Drop exempted findings and record them; return unmatched leftovers.

    ``matched_indexes`` carries exemption indexes already consumed by
    earlier reports of the same run.  Without it, an exemption matched in
    one sample's report would resurface as "unused" against every other
    report and turn into phantom violations in multi-sample runs.
    """

    if matched_indexes is None:
        matched_indexes = set()
    remaining: list[Finding] = []
    for finding in report.findings:
        for index, exemption in enumerate(exemptions):
            if index in matched_indexes:
                continue
            if (exemption.rule == finding.rule
                    and exemption.path == finding.path
                    and exemption.line == finding.line
                    and (exemption.message is None
                         or exemption.message == finding.message)):
                matched_indexes.add(index)
                report.exemptions_applied.append({
                    "rule": finding.rule, "path": finding.path,
                    "line": finding.line, "reason": exemption.reason,
                })
                break
        else:
            remaining.append(finding)
    report.findings = remaining
    return [exemptions[i] for i in range(len(exemptions))
            if i not in matched_indexes]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the checker's own CLI parser (side-effect free)."""

    parser = argparse.ArgumentParser(
        prog="sample-contract",
        description="Static README/interface contract checker for unified "
                    "samples (Phase 0.5 Q3).")
    parser.add_argument(
        "--sample", action="append", default=[], metavar="PATH",
        help="sample directory to check (repeatable)")
    parser.add_argument(
        "--scope", choices=("migration",),
        help="resolve samples from the migration progress region")
    parser.add_argument(
        "--migration-map", type=Path,
        default=REPO_ROOT / "docs/releases/unified-migration/"
                            "x5-s-migration-map.md",
        help="progress-region markdown used by --scope migration")
    parser.add_argument(
        "--samples-root", type=Path, default=REPO_ROOT / "samples",
        help="root directory holding <domain>/<sample> directories")
    parser.add_argument(
        "--templates-dir", type=Path,
        default=REPO_ROOT / "docs/sample-standards/templates",
        help="directory holding the Q1 bilingual README templates")
    parser.add_argument(
        "--parser-mode", choices=("import", "static"), default="import",
        help="import mode calls the trusted sample's build_parser(); "
             "static mode never executes sample code and records skips")
    parser.add_argument(
        "--exemptions", type=Path, metavar="JSON",
        help="JSON file of reviewed exceptions "
             "(rule, path, line, reason; reason is mandatory)")
    parser.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="output format on stdout")
    parser.add_argument(
        "--report", type=Path, metavar="PATH",
        help="write the full JSON report to this file (evidence record)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns 0 when no violations, 1 when violations exist."""

    args = build_arg_parser().parse_args(argv)
    if not args.sample and not args.scope:
        print("sample-contract: pass --sample PATH or --scope migration",
              file=sys.stderr)
        return 2

    try:
        templates = load_template_anchors(args.templates_dir)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    sample_dirs = [Path(target) for target in args.sample]
    scope_findings: list[Finding] = []
    if args.scope == "migration":
        resolved, scope_findings = resolve_migration_scope(
            args.migration_map, args.samples_root)
        sample_dirs.extend(resolved)

    try:
        exemptions = load_exemptions(args.exemptions)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    reports: list[SampleReport] = []
    for sample_dir in dict.fromkeys(sample_dirs):
        if not sample_dir.is_dir():
            print(f"sample-contract: no such sample directory: "
                  f"{sample_dir}", file=sys.stderr)
            return 2
        reports.append(
            run_sample(sample_dir, templates, args.parser_mode))
    if not reports:
        # Scope resolution produced nothing; findings still need a home.
        reports.append(SampleReport(path=display_path(args.samples_root)))

    matched_indexes: set = set()
    for report in reports:
        apply_exemptions(report, exemptions, matched_indexes)
    # An exemption counts as unused only if no report in this run matched
    # it — per-report leftovers would flag yolo's baseline entries against
    # every other sample in scope runs.
    unused = [exemptions[i] for i in range(len(exemptions))
              if i not in matched_indexes]
    for exemption in unused:
        reports[0].findings.append(Finding(
            RULE_EXEMPTION, exemption.path, exemption.line,
            f"unused exemption ({exemption.reason[:60]})"))
    for finding in scope_findings:
        reports[0].findings.append(finding)

    payload = {
        "tool": "sample-contract",
        "parser_mode": args.parser_mode,
        "templates_dir": display_path(args.templates_dir),
        "samples": [
            {
                "path": report.path,
                "findings": [f.as_dict() for f in report.findings],
                "skips": [s.as_dict() for s in report.skips],
                "exemptions_applied": report.exemptions_applied,
            }
            for report in reports
        ],
        "summary": {
            "samples": len(reports),
            "violations": sum(len(r.findings) for r in reports),
            "skips": sum(len(r.skips) for r in reports),
            "exemptions_applied": sum(len(r.exemptions_applied)
                                      for r in reports),
        },
    }

    if args.format == "json":
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        for report in reports:
            print(f"sample: {report.path}")
            for finding in report.findings:
                location = (f"{finding.path}:{finding.line}"
                            if finding.line else finding.path)
                print(f"  {finding.rule} {location} {finding.message}")
            for skip in report.skips:
                print(f"  skipped {skip.rule} {skip.path}: {skip.reason}")
            for applied in report.exemptions_applied:
                print(f"  exempted {applied['rule']} {applied['path']}:"
                      f"{applied['line']} — {applied['reason']}")
        summary = payload["summary"]
        print(f"summary: {summary['samples']} samples, "
              f"{summary['violations']} violations, "
              f"{summary['skips']} skips, "
              f"{summary['exemptions_applied']} exemptions applied")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    return 1 if payload["summary"]["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
