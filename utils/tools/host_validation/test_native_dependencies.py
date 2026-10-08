"""Portable native dependency resolver tests.

Covers ``utils/tools/host_validation/native_dependencies.py``: explicit-override
validation, pkg-config and standard-system discovery, missing-dependency
errors, and macOS/Linux iconv link flags. Discovery branches are exercised
hermetically against temporary header/library trees with an injected
pkg-config double; two additional checks run against the real host
installation and skip explicitly when the dependency is absent (the CI runner
installs these dependencies and disallows the skips).

The consumer guards at the bottom keep the Gemma/MiniCPM/HIMLoco host tests
free of personal paths (``.coordination``), hard-coded Homebrew gflags
locations, hard-coded ``/usr/include`` assumptions and unconditional
``-liconv`` linking, so a fresh checkout runs from standard declared
dependencies only.
"""

import os
from pathlib import Path
import platform
import tempfile
import unittest
from unittest import mock

import native_dependencies as deps

REPO = Path(__file__).resolve().parents[3]
GEMMA_TEST = REPO / "samples/llm/gemma4-e2b/tests/test_cpp_chat_app.py"
MINICPM_TEST = REPO / "samples/llm/minicpm5-2b/tests/test_native_core.py"
HIMLOCO_TEST = REPO / "samples/robotics/himloco/tests/test_cpp_cli.py"

NO_PKG_CONFIG = lambda package, mode, environ: None  # noqa: E731


def make_json_include(root):
    include = root / "json-include"
    (include / "nlohmann").mkdir(parents=True)
    (include / "nlohmann" / "json.hpp").write_text("// fixture header\n")
    return include


def make_gflags_include(root):
    include = root / "gflags-include"
    (include / "gflags").mkdir(parents=True)
    (include / "gflags" / "gflags.h").write_text("// fixture header\n")
    return include


def make_gflags_lib(root, entries, name="gflags-lib"):
    """Create a gflags library directory from ``{name: kind}`` entries.

    ``kind`` is ``"file"``, ``"dir"``, or ``("symlink", target)``; a ``None``
    target produces a dangling symlink.
    """
    lib = root / name
    lib.mkdir()
    for library_name, kind in entries.items():
        path = lib / library_name
        if kind == "dir":
            path.mkdir()
        elif isinstance(kind, tuple):
            os.symlink(kind[1] or "missing-target", path)
        else:
            path.write_text("")
    return lib


def make_gflags_tree(root, library_name="libgflags.so"):
    return make_gflags_include(root), make_gflags_lib(root, {library_name: "file"})


class JsonIncludeDirTests(unittest.TestCase):
    """json_include_dir: override validation, pkg-config/standard discovery."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native-deps-json-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_explicit_override_returns_validated_directory(self):
        include = make_json_include(self.root)
        self.assertEqual(deps.json_include_dir(str(include)), include)

    def test_override_accepts_path_object_and_expands_user(self):
        include = make_json_include(self.root)
        with mock.patch.dict(
            "os.environ", {"HOME": str(self.root)}, clear=False
        ):
            self.assertEqual(
                deps.json_include_dir(Path("~/json-include")), include
            )

    def test_override_pointing_at_missing_directory_raises(self):
        with self.assertRaises(deps.NativeDependencyOverrideError) as raised:
            deps.json_include_dir(str(self.root / "absent"))
        self.assertIn("absent", str(raised.exception))

    def test_override_without_nlohmann_header_raises(self):
        empty = self.root / "empty-include"
        empty.mkdir()
        with self.assertRaises(deps.NativeDependencyOverrideError) as raised:
            deps.json_include_dir(str(empty))
        self.assertIn("nlohmann/json.hpp", str(raised.exception))

    def test_blank_override_falls_through_to_discovery(self):
        include = make_json_include(self.root)
        self.assertEqual(
            deps.json_include_dir(
                "", include_roots=(include,), pkg_config=NO_PKG_CONFIG
            ),
            include,
        )

    def test_pkg_config_include_wins_when_header_present(self):
        include = make_json_include(self.root)
        other = self.root / "other-root"
        other.mkdir()
        found = deps.json_include_dir(
            pkg_config=lambda package, mode, environ: f"-I{include}"
            if mode == "cflags"
            else None,
            include_roots=(other,),
        )
        self.assertEqual(found, include)

    def test_pkg_config_include_without_header_falls_through_to_roots(self):
        # A stale/misleading pkg-config entry must not mask a usable root.
        include = make_json_include(self.root)
        stale = self.root / "stale-include"
        stale.mkdir()
        found = deps.json_include_dir(
            pkg_config=lambda package, mode, environ: f"-I{stale}"
            if mode == "cflags"
            else None,
            include_roots=(include,),
        )
        self.assertEqual(found, include)

    def test_pkg_config_output_with_separate_i_argument_and_spaces(self):
        include = make_json_include(self.root)
        found = deps.json_include_dir(
            pkg_config=lambda package, mode, environ: f"-I {include} -DNDEBUG"
            if mode == "cflags"
            else None,
            include_roots=(),
        )
        self.assertEqual(found, include)

    def test_pkg_config_unavailable_falls_through_to_roots(self):
        include = make_json_include(self.root)
        self.assertEqual(
            deps.json_include_dir(
                pkg_config=NO_PKG_CONFIG, include_roots=(include,)
            ),
            include,
        )

    def test_missing_everywhere_raises_missing_with_guidance(self):
        with self.assertRaises(deps.NativeDependencyMissing) as raised:
            deps.json_include_dir(
                pkg_config=NO_PKG_CONFIG, include_roots=()
            )
        message = str(raised.exception)
        self.assertIn("nlohmann", message.lower())
        self.assertIn("GEMMA_JSON_INCLUDE", message)
        self.assertIn("MINICPM_JSON_INCLUDE", message)

    def test_no_personal_path_default_hides_behind_empty_discovery(self):
        # With no pkg-config and no roots, resolution must fail loudly; there
        # is never a silent fallback to a repo-external personal directory.
        for roots in ((), ("/definitely-not-a-standard-root",)):
            with self.subTest(roots=roots):
                self.assertRaises(
                    deps.NativeDependencyMissing,
                    deps.json_include_dir,
                    pkg_config=NO_PKG_CONFIG,
                    include_roots=roots,
                )
        for root in deps.STANDARD_INCLUDE_ROOTS:
            self.assertNotIn(".coordination", root)

    def test_real_host_resolution_returns_usable_include(self):
        try:
            include = deps.json_include_dir()
        except deps.NativeDependencyMissing as error:
            self.skipTest(
                f"host nlohmann-json headers not installed ({error}); "
                "standard discovery not exercised here"
            )
        self.assertTrue((include / "nlohmann" / "json.hpp").is_file())


class IconvLinkFlagsTests(unittest.TestCase):
    """iconv_link_flags: -liconv only on platforms that need it."""

    def test_darwin_links_libiconv(self):
        self.assertEqual(deps.iconv_link_flags("darwin"), ["-liconv"])

    def test_linux_uses_libc(self):
        self.assertEqual(deps.iconv_link_flags("linux"), [])

    def test_system_names_are_case_insensitive(self):
        self.assertEqual(deps.iconv_link_flags("Darwin"), ["-liconv"])
        self.assertEqual(deps.iconv_link_flags("LINUX"), [])

    def test_unsupported_system_raises_instead_of_guessing(self):
        self.assertRaises(deps.NativeDependencyError, deps.iconv_link_flags, "windows")

    def test_default_system_follows_platform(self):
        for reported, expected in (("Linux", []), ("Darwin", ["-liconv"])):
            with self.subTest(system=reported):
                with mock.patch.object(platform, "system", return_value=reported):
                    self.assertEqual(deps.iconv_link_flags(), expected)


class GflagsCompileFlagsTests(unittest.TestCase):
    """gflags_compile_flags: overrides, pkg-config, standard roots."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native-deps-gflags-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.include = make_gflags_include(self.root)

    def test_explicit_override_pair(self):
        lib = make_gflags_lib(self.root, {"libgflags.so": "file"})
        compile_flags, link_flags = deps.gflags_compile_flags(
            environ={
                "GFLAGS_INCLUDE_DIR": str(self.include),
                "GFLAGS_LIB_DIR": str(lib),
            },
            pkg_config=NO_PKG_CONFIG,
            include_roots=(),
            lib_roots=(),
        )
        self.assertEqual(compile_flags, ["-I", str(self.include)])
        self.assertEqual(link_flags, ["-L", str(lib), "-lgflags"])

    def test_partial_override_raises(self):
        lib = make_gflags_lib(self.root, {"libgflags.so": "file"})
        for environ in (
            {"GFLAGS_INCLUDE_DIR": str(self.include)},
            {"GFLAGS_LIB_DIR": str(lib)},
        ):
            with self.subTest(environ=environ):
                with self.assertRaises(deps.NativeDependencyOverrideError):
                    deps.gflags_compile_flags(
                        environ=environ,
                        pkg_config=NO_PKG_CONFIG,
                        include_roots=(),
                        lib_roots=(),
                    )

    def test_override_without_header_raises(self):
        lib = make_gflags_lib(self.root, {"libgflags.so": "file"})
        (self.include / "gflags" / "gflags.h").unlink()
        with self.assertRaises(deps.NativeDependencyOverrideError) as raised:
            deps.gflags_compile_flags(
                environ={
                    "GFLAGS_INCLUDE_DIR": str(self.include),
                    "GFLAGS_LIB_DIR": str(lib),
                },
                pkg_config=NO_PKG_CONFIG,
                include_roots=(),
                lib_roots=(),
            )
        self.assertIn("gflags/gflags.h", str(raised.exception))

    def test_override_without_library_raises(self):
        lib = make_gflags_lib(self.root, {})
        with self.assertRaises(deps.NativeDependencyOverrideError):
            deps.gflags_compile_flags(
                environ={
                    "GFLAGS_INCLUDE_DIR": str(self.include),
                    "GFLAGS_LIB_DIR": str(lib),
                },
                pkg_config=NO_PKG_CONFIG,
                include_roots=(),
                lib_roots=(),
            )

    def test_pkg_config_flags_are_preferred(self):
        found = deps.gflags_compile_flags(
            environ={},
            pkg_config=lambda package, mode, environ: {
                "cflags": "-I/usr/include",
                "libs": "-L/usr/lib -lgflags -lpthread",
            }[mode],
            include_roots=(),
            lib_roots=(),
        )
        self.assertEqual(
            found,
            (["-I", "/usr/include"], ["-L/usr/lib", "-lgflags", "-lpthread"]),
        )

    def test_standard_roots_discovery(self):
        lib = make_gflags_lib(self.root, {"libgflags.so": "file"})
        compile_flags, link_flags = deps.gflags_compile_flags(
            environ={},
            pkg_config=NO_PKG_CONFIG,
            include_roots=(self.include,),
            lib_roots=(lib,),
        )
        self.assertEqual(compile_flags, ["-I", str(self.include)])
        self.assertEqual(link_flags, ["-L", str(lib), "-lgflags"])

    def test_header_without_library_raises_missing(self):
        with self.assertRaises(deps.NativeDependencyMissing) as raised:
            deps.gflags_compile_flags(
                environ={},
                pkg_config=NO_PKG_CONFIG,
                include_roots=(self.include,),
                lib_roots=(),
            )
        self.assertIn("librar", str(raised.exception).lower())

    def test_unversioned_shared_and_static_library_names_count(self):
        for name in ("libgflags.so", "libgflags.dylib", "libgflags.a"):
            with self.subTest(library=name), tempfile.TemporaryDirectory() as tmp:
                include, lib = make_gflags_tree(Path(tmp), library_name=name)
                _flags, link_flags = deps.gflags_compile_flags(
                    environ={},
                    pkg_config=NO_PKG_CONFIG,
                    include_roots=(include,),
                    lib_roots=(lib,),
                )
                self.assertIn("-lgflags", link_flags)

    def test_versioned_file_with_unversioned_symlink_counts(self):
        # The normal distribution/Homebrew layout: libgflags.dylib is a
        # symlink to the versioned file, and -lgflags resolves through it.
        lib = make_gflags_lib(
            self.root,
            {
                "libgflags.2.3.dylib": "file",
                "libgflags.dylib": ("symlink", "libgflags.2.3.dylib"),
            },
        )
        _flags, link_flags = deps.gflags_compile_flags(
            environ={},
            pkg_config=NO_PKG_CONFIG,
            include_roots=(self.include,),
            lib_roots=(lib,),
        )
        self.assertEqual(link_flags, ["-L", str(lib), "-lgflags"])

    def assert_not_linkable(self, entries, label):
        """A non-linkable library dir must fail both discovery channels."""
        lib = make_gflags_lib(self.root, entries, name=label)
        with self.subTest(channel="standard roots", library=label):
            with self.assertRaises(deps.NativeDependencyMissing):
                deps.gflags_compile_flags(
                    environ={},
                    pkg_config=NO_PKG_CONFIG,
                    include_roots=(self.include,),
                    lib_roots=(lib,),
                )
        with self.subTest(channel="explicit override", library=label):
            with self.assertRaises(deps.NativeDependencyOverrideError):
                deps.gflags_compile_flags(
                    environ={
                        "GFLAGS_INCLUDE_DIR": str(self.include),
                        "GFLAGS_LIB_DIR": str(lib),
                    },
                    pkg_config=NO_PKG_CONFIG,
                )

    def test_versioned_only_library_is_not_linkable(self):
        # -lgflags cannot resolve to a versioned-only file; the real linker
        # fails with "ld: library 'gflags' not found" (review reproducer).
        self.assert_not_linkable(
            {"libgflags.2.3.dylib": "file"}, "versioned-only-dylib"
        )
        self.assert_not_linkable(
            {"libgflags.so.2.2": "file"}, "versioned-only-so"
        )

    def test_directory_named_like_a_library_is_not_linkable(self):
        self.assert_not_linkable({"libgflags.so": "dir"}, "directory-named-so")
        self.assert_not_linkable({"libgflags.dylib": "dir"}, "directory-named-dylib")

    def test_dangling_symlink_is_not_linkable(self):
        self.assert_not_linkable(
            {"libgflags.dylib": ("symlink", None)}, "dangling-dylib"
        )

    def test_nothreads_variant_alone_is_not_linkable_gflags(self):
        lib = make_gflags_lib(self.root, {"libgflags_nothreads.a": "file"})
        self.assertRaises(
            deps.NativeDependencyMissing,
            deps.gflags_compile_flags,
            environ={},
            pkg_config=NO_PKG_CONFIG,
            include_roots=(self.include,),
            lib_roots=(lib,),
        )

    def test_missing_everywhere_raises_missing(self):
        with self.assertRaises(deps.NativeDependencyMissing) as raised:
            deps.gflags_compile_flags(
                environ={}, pkg_config=NO_PKG_CONFIG, include_roots=(), lib_roots=()
            )
        self.assertIn("gflags", str(raised.exception).lower())
        for root in deps.STANDARD_LIB_ROOTS:
            self.assertNotIn(".coordination", root)

    def test_real_host_resolution_returns_usable_flags(self):
        try:
            compile_flags, link_flags = deps.gflags_compile_flags()
        except deps.NativeDependencyMissing as error:
            self.skipTest(
                f"host gflags not installed ({error}); standard discovery not exercised here"
            )
        self.assertIn("-lgflags", link_flags)
        # Every reported include directory must carry the real header
        # (pkg-config may legitimately emit no -I when headers sit in the
        # compiler's default search path).
        include_dirs = []
        for index, flag in enumerate(compile_flags):
            if flag == "-I" and index + 1 < len(compile_flags):
                include_dirs.append(compile_flags[index + 1])
            elif flag.startswith("-I") and len(flag) > 2:
                include_dirs.append(flag[2:])
        for directory in include_dirs:
            self.assertTrue(
                (Path(directory) / "gflags" / "gflags.h").is_file(),
                f"gflags.h missing from reported include dir {directory}",
            )


class ConsumerPortabilityGuardsTests(unittest.TestCase):
    """The Gemma/MiniCPM/HIMLoco host tests must stay portable."""

    def test_gemma_chat_app_test_uses_portable_dependencies(self):
        text = GEMMA_TEST.read_text()
        self.assertNotIn(".coordination", text)
        self.assertNotIn("/opt/homebrew", text)
        self.assertNotIn('"-liconv"', text)
        self.assertIn("iconv_link_flags", text)
        self.assertIn("gflags_compile_flags", text)

    def test_minicpm_native_core_test_uses_portable_dependencies(self):
        text = MINICPM_TEST.read_text()
        self.assertNotIn(".coordination", text)
        self.assertIn("json_include_dir", text)

    def test_himloco_cli_test_uses_portable_dependencies(self):
        text = HIMLOCO_TEST.read_text()
        self.assertNotIn(".coordination", text)
        self.assertNotIn('"/usr/include"', text)
        self.assertIn("json_include_dir", text)


if __name__ == "__main__":
    unittest.main()
