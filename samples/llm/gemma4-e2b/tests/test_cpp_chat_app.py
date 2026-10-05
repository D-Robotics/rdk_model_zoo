"""Interactive chat application host checks (compile + run scenarios).

Compiles the production application session source ``src/gemma4_chat_app.cpp``
and the production entry ``src/main.cpp`` against the engine doubles in
``tests/native/chat_app_doubles.cpp``, the SDK header double in
``tests/native/sdk_fixtures`` and clearly-marked compile stubs for the
third-party tokenizers-cpp / OpenCV headers, then executes REPL scenarios
through redirected stdin.

Boundaries: the doubles replace the engines, tokenizer and image decoding —
no HBM, BPU, tokenization or OpenCV decoding runs here, and nothing in this
module is board evidence. The check verifies the application session logic
that is host-verifiable: engine construction order and messages, streaming
display, context commands, image-turn wiring, history trimming and the thin
entry's flag validation. Linking the real SDK/OpenCV/tokenizers-cpp stack
remains board-side and is not-run on the host.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SAMPLE = Path(__file__).resolve().parents[1]
CPP = SAMPLE / "runtime/cpp"
NATIVE = SAMPLE / "tests/native"
REPO = SAMPLE.parents[2]
CXX = os.environ.get("GEMMA_CXX", "c++")
JSON_INCLUDE = Path(
    os.environ.get(
        "GEMMA_JSON_INCLUDE", REPO.parent / ".coordination/asr-json/include"
    )
)

COMMON_INCLUDES = [
    "-I",
    str(NATIVE / "sdk_fixtures"),
    "-I",
    str(NATIVE / "app_stubs"),
    "-I",
    str(CPP / "inc"),
    "-I",
    str(JSON_INCLUDE),
]


def compile_sources(sources, output, *, extra_includes=(), extra_flags=()):
    command = [
        CXX,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        *COMMON_INCLUDES,
        *extra_includes,
        *[str(source) for source in sources],
        "-o",
        str(output),
        # macOS keeps iconv in libiconv rather than libc.
        "-liconv",
        *extra_flags,
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise AssertionError(f"compile failed:\n{done.stdout}{done.stderr}")
    return command


class ChatAppScenarioTests(unittest.TestCase):
    """REPL scenarios against the production app session + engine doubles."""

    @classmethod
    def setUpClass(cls):
        if shutil.which(CXX) is None:
            raise unittest.SkipTest("C++17 compiler unavailable; chat app not-run")
        if not JSON_INCLUDE.is_dir():
            raise unittest.SkipTest(
                "nlohmann json headers unavailable; chat app not-run"
            )
        cls.build = tempfile.TemporaryDirectory(prefix="gemma-chat-app-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "chat_app_test"
        compile_sources(
            [
                NATIVE / "chat_app_test.cpp",
                NATIVE / "chat_app_doubles.cpp",
                CPP / "src/gemma4_chat_app.cpp",
            ],
            cls.binary,
        )

    def run_scenario(self, scenario, stdin_text, *, image_path=None):
        argv = [str(self.binary), scenario]
        if image_path is not None:
            argv.append(str(image_path))
        if isinstance(stdin_text, bytes):
            # Raw terminal bytes (e.g. GB18030) must reach the process as-is.
            return subprocess.run(
                argv, input=stdin_text, capture_output=True, timeout=120
            )
        return subprocess.run(
            argv,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_text_turn_streams_and_reports_context(self):
        done = self.run_scenario("text_turn", "hello\n/context\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK text_turn", done.stdout)
        self.assertIn("WW", done.stdout)  # two streamed tokens via doubles
        self.assertIn("tok/s]", done.stdout)
        self.assertIn("[context] prompt=", done.stdout)
        self.assertIn("Context: ", done.stdout)  # /context after one turn
        self.assertIn(", turns=1", done.stdout)

    def test_image_turn_injects_vision_features(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "photo.bin"
            image.write_bytes(b"\x89PNG-ish-bytes")
            done = self.run_scenario(
                "image_turn",
                f"/image {image}\nwhat is this\n/quit\n",
                image_path=image,
            )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK image_turn", done.stdout)
        self.assertIn("Image loaded (430080 features).", done.stdout)

    def test_reset_and_context_commands(self):
        done = self.run_scenario("reset_context", "/reset\n/context\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK reset_context", done.stdout)
        self.assertIn("Session reset.", done.stdout)
        self.assertIn("Context: 0/4096 tokens, remaining=4096, turns=0", done.stdout)

    def test_oversize_prompt_is_rejected_without_generation(self):
        done = self.run_scenario("oversize_prompt", "a" * 5000 + "\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK oversize_prompt", done.stdout)
        self.assertIn("exceeding the 4096-token KV cache.", done.stdout)

    def test_history_trims_oldest_turns_within_budget(self):
        done = self.run_scenario(
            "history_trim", "x" * 3000 + "\n" + "y" * 3000 + "\n/quit\n"
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK history_trim", done.stdout)
        self.assertIn("Oldest chat turns were removed", done.stdout)

    def test_rebuild_each_turn_resets_session(self):
        done = self.run_scenario(
            "rebuild_each_turn", "one\ntwo\n/quit\n"
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK rebuild_each_turn", done.stdout)

    def test_session_context_grows_across_turns(self):
        done = self.run_scenario("session_growth", "first\nsecond\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("OK session_growth", done.stdout)

    def test_gb18030_terminal_input_is_converted(self):
        done = self.run_scenario("text_turn", b"\xc4\xe3\xba\xc3\n/quit\n")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(
            b"[input] Converted GB18030 terminal bytes to UTF-8.", done.stderr
        )


class ChatAppEntryTests(unittest.TestCase):
    """The production thin main.cpp: flag validation and construct-and-run."""

    @classmethod
    def setUpClass(cls):
        if shutil.which(CXX) is None:
            raise unittest.SkipTest("C++17 compiler unavailable; entry not-run")
        if not JSON_INCLUDE.is_dir():
            raise unittest.SkipTest("nlohmann json headers unavailable; entry not-run")
        cls.gflags_include = Path("/opt/homebrew/include")
        cls.gflags_lib = Path("/opt/homebrew/lib")
        for candidate in (
            cls.gflags_include / "gflags" / "gflags.h",
            cls.gflags_lib / "libgflags.dylib",
        ):
            if not candidate.exists():
                raise unittest.SkipTest(
                    f"host gflags unavailable ({candidate}); entry not-run"
                )
        cls.build = tempfile.TemporaryDirectory(prefix="gemma-chat-main-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.binary = Path(cls.build.name) / "chat_main_host"
        compile_sources(
            [
                CPP / "src/main.cpp",
                CPP / "src/gemma4_chat_app.cpp",
                NATIVE / "chat_app_doubles.cpp",
            ],
            cls.binary,
            extra_includes=["-I", str(cls.gflags_include)],
            extra_flags=[
                "-L",
                str(cls.gflags_lib),
                "-lgflags",
            ],
        )

    def run_entry(self, arguments, stdin_text=""):
        return subprocess.run(
            [str(self.binary), *arguments],
            input=stdin_text,
            capture_output=True,
            text=True,
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


if __name__ == "__main__":
    unittest.main()
