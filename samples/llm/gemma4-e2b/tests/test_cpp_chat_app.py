"""Interactive chat model + entry host checks (compile + run scenarios).

Compiles the production model source ``src/gemma4.cpp``, the CLI
argument/presentation source ``src/cli.cpp`` and the production entry
``src/main.cpp`` against the engine doubles in
``tests/native/chat_app_doubles.cpp``, the SDK header double in
``tests/native/sdk_fixtures`` and clearly-marked compile stubs for the
third-party tokenizers-cpp / OpenCV headers. The scenario driver calls the
Gemma4 model API directly (one predict per turn); the entry binary exercises
the real REPL — ParseOptions flags plus the per-turn command dispatch —
through redirected stdin, including the invalid-input-then-continue path.

Boundaries: the doubles replace the engines, tokenizer and image decoding —
no HBM, BPU, tokenization or OpenCV decoding runs here, and nothing in this
module is board evidence. The check verifies the session logic that is
host-verifiable: engine construction order and messages, streaming display,
context commands, image-turn wiring, history trimming and the thin entry's
flag parsing/validation. Linking the real SDK/OpenCV/tokenizers-cpp stack
remains board-side and is not-run on the host.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"
REPO = SAMPLE.parents[2]
CXX = os.environ.get("GEMMA_CXX", "c++")
HOST_VALIDATION = REPO / "utils" / "tools" / "host_validation"
if str(HOST_VALIDATION) not in sys.path:
    sys.path.insert(0, str(HOST_VALIDATION))

from native_dependencies import (  # noqa: E402
    NativeDependencyMissing,
    gflags_compile_flags,
    iconv_link_flags,
    json_include_dir,
)


def resolve_json_include():
    """nlohmann include dir: ``GEMMA_JSON_INCLUDE`` override or discovery.

    Standard discovery (pkg-config / system include roots) lives in
    ``utils/tools/host_validation/native_dependencies.py``; a machine without any
    nlohmann headers skips these host checks explicitly, while an invalid
    override fails the run.
    """
    try:
        return json_include_dir(os.environ.get("GEMMA_JSON_INCLUDE"))
    except NativeDependencyMissing as error:
        raise unittest.SkipTest(f"{error}; chat app not-run") from error


def compile_sources(sources, output, *, json_include, extra_includes=(), extra_flags=()):
    command = [
        CXX,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(NATIVE / "sdk_fixtures"),
        "-I",
        str(NATIVE / "app_stubs"),
        "-I",
        str(CPP / "inc"),
        "-I",
        str(json_include),
        *extra_includes,
        *[str(source) for source in sources],
        "-o",
        str(output),
        # iconv lives in libiconv on macOS and in libc on Linux.
        *iconv_link_flags(),
        *extra_flags,
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise AssertionError(f"compile failed:\n{done.stdout}{done.stderr}")
    return command


class ChatAppScenarioTests(unittest.TestCase):
    """Chat scenarios against the production Gemma4 model + engine doubles."""

    @classmethod
    def setUpClass(cls):
        if shutil.which(CXX) is None:
            raise unittest.SkipTest("C++17 compiler unavailable; chat model not-run")
        cls.json_include = resolve_json_include()
        try:
            # cli.cpp owns the gflags command line, so the scenario build
            # needs the host gflags headers/libs as well.
            cls.gflags_compile, cls.gflags_link = gflags_compile_flags()
        except NativeDependencyMissing as error:
            raise unittest.SkipTest(f"{error}; chat model not-run") from error
        cls.build = tempfile.TemporaryDirectory(prefix="gemma-chat-app-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "chat_app_test"
        compile_sources(
            [
                NATIVE / "chat_app_test.cpp",
                NATIVE / "chat_app_doubles.cpp",
                CPP / "src/gemma4.cpp",
                CPP / "src/cli.cpp",
            ],
            cls.binary,
            json_include=cls.json_include,
            extra_includes=cls.gflags_compile,
            extra_flags=cls.gflags_link,
        )

    def run_scenario(self, scenario, *, image_path=None):
        argv = [str(self.binary), scenario]
        if image_path is not None:
            argv.append(str(image_path))
        return subprocess.run(
            argv,
            input="",
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_text_turn_streams_and_reports_context(self):
        done = self.run_scenario("text_turn")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK text_turn", done.stdout)
        self.assertIn("WW", done.stdout)  # two streamed tokens via doubles
        self.assertIn("tok/s]", done.stdout)
        self.assertIn("[context] prompt=", done.stdout)
        self.assertIn("Context: ", done.stdout)  # usage after one turn
        self.assertIn(", turns=1", done.stdout)

    def test_image_turn_injects_vision_features(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "photo.bin"
            image.write_bytes(b"\x89PNG-ish-bytes")
            done = self.run_scenario("image_turn", image_path=image)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK image_turn", done.stdout)
        self.assertIn("Image loaded (430080 features).", done.stdout)

    def test_reset_and_context_commands(self):
        done = self.run_scenario("reset_context")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK reset_context", done.stdout)
        self.assertIn("Session reset.", done.stdout)
        self.assertIn("Context: 0/4096 tokens, remaining=4096, turns=0", done.stdout)

    def test_oversize_prompt_is_rejected_without_generation(self):
        done = self.run_scenario("oversize_prompt")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK oversize_prompt", done.stdout)
        self.assertIn("exceeding the 4096-token KV cache.", done.stdout)

    def test_history_trims_oldest_turns_within_budget(self):
        done = self.run_scenario("history_trim")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK history_trim", done.stdout)
        self.assertIn("Oldest chat turns were removed", done.stdout)

    def test_rebuild_each_turn_resets_session(self):
        done = self.run_scenario("rebuild_each_turn")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK rebuild_each_turn", done.stdout)

    def test_session_context_grows_across_turns(self):
        done = self.run_scenario("session_growth")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK session_growth", done.stdout)


class ChatAppEntryTests(unittest.TestCase):
    """The production thin main.cpp: ParseOptions, construct, per-turn REPL.

    main.cpp must stay visible-thin: the CLI module parses flags and classifies
    terminal lines, main constructs the named Gemma4 model and dispatches one
    model.predict call per chat turn. These checks exercise that moved
    boundary end-to-end through the real entry binary.
    """

    @classmethod
    def setUpClass(cls):
        if shutil.which(CXX) is None:
            raise unittest.SkipTest("C++17 compiler unavailable; entry not-run")
        cls.json_include = resolve_json_include()
        try:
            # Real host gflags: explicit GFLAGS_INCLUDE_DIR/GFLAGS_LIB_DIR
            # override, pkg-config, or standard system roots.
            cls.gflags_compile, cls.gflags_link = gflags_compile_flags()
        except NativeDependencyMissing as error:
            raise unittest.SkipTest(f"{error}; entry not-run") from error
        cls.build = tempfile.TemporaryDirectory(prefix="gemma-chat-main-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "chat_main_host"
        compile_sources(
            [
                CPP / "src/main.cpp",
                CPP / "src/cli.cpp",
                CPP / "src/gemma4.cpp",
                NATIVE / "chat_app_doubles.cpp",
            ],
            cls.binary,
            json_include=cls.json_include,
            extra_includes=cls.gflags_compile,
            extra_flags=cls.gflags_link,
        )

    def run_entry(self, arguments, stdin_text=""):
        return subprocess.run(
            [str(self.binary), *arguments],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def run_entry_bytes(self, stdin_bytes):
        # Raw terminal bytes (e.g. GB18030) must reach the process as-is.
        return subprocess.run(
            [str(self.binary)],
            input=stdin_bytes,
            capture_output=True,
            timeout=120,
        )

    def test_construct_and_quit_cleanly(self):
        done = self.run_entry([], "/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        # The title line is ANSI-colored per letter; assert the plain subtitle.
        self.assertIn("Vision-Language Model | D-Robotics", done.stdout)
        self.assertIn("Loading vision model...", done.stdout)
        self.assertIn("KV cache: 4096 tokens", done.stdout)
        self.assertIn("gemma4> ", done.stdout)

    def test_flag_validation_exit_codes(self):
        self.assertEqual(self.run_entry(["--max_tokens=-1"]).returncode, 2)
        self.assertEqual(
            self.run_entry(["--min_response_tokens=0"]).returncode, 2
        )

    def test_flags_reach_the_session(self):
        done = self.run_entry(["--max_tokens=512"], "/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("max output: 512", done.stdout)

    def test_gb18030_terminal_input_is_converted(self):
        done = self.run_entry_bytes(b"\xc4\xe3\xba\xc3\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(
            b"[input] Converted GB18030 terminal bytes to UTF-8.", done.stderr
        )

    def test_full_turn_through_the_entry_repl(self):
        done = self.run_entry([], "hello\n/context\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("WW", done.stdout)  # streamed tokens via doubles
        self.assertIn("tok/s]", done.stdout)
        self.assertIn("Context: ", done.stdout)
        self.assertIn(", turns=1", done.stdout)

    def test_invalid_terminal_line_reprompts_and_continues(self):
        # Invalid UTF-8/GB18030 input must print the terminal error, re-prompt
        # and keep the session alive: the next (valid) line still reaches
        # model.predict. The previous entry loop exited on kInvalid, which
        # silently dropped every later line of the session.
        done = self.run_entry_bytes(b"\xff\xfe\x80\nhello\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(
            b"Input error: terminal text is neither valid UTF-8 nor GB18030",
            done.stderr,
        )
        self.assertIn(b"WW", done.stdout)  # the later valid line ran a turn

    def test_reset_context_image_and_quit_through_the_repl(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "photo.bin"
            image.write_bytes(b"\x89PNG-ish-bytes")
            done = self.run_entry([], f"/reset\n/context\n/image {image}\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Session reset.", done.stdout)
        self.assertIn("Context: 0/4096 tokens, remaining=4096, turns=0", done.stdout)
        self.assertIn(f"Processing image: {image}", done.stdout)
        self.assertIn("Image loaded (430080 features).", done.stdout)
        self.assertIn("Now type your question about the image.", done.stdout)

    def test_image_argument_errors_stay_in_the_repl(self):
        done = self.run_entry([], "/image \n/image /nonexistent/photo.jpg\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Error: /image requires a file path", done.stdout)
        self.assertIn("Error: cannot open image file: /nonexistent/photo.jpg", done.stdout)

    def test_flag_defaults_report_auto_output_budget(self):
        # No flags: max_tokens 0 reports the all-remaining-tokens budget and
        # min_response_tokens keeps its default reserve.
        done = self.run_entry([], "/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(
            "KV cache: 4096 tokens; max output: auto (all remaining tokens)",
            done.stdout,
        )


if __name__ == "__main__":
    unittest.main()
