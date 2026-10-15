// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Host tests for the shared linewise label reader (load_linewise_labels).
// Covers the three label-file shapes the samples ship today — single-quoted
// Python dict lines, double-quoted dict lines whose text contains apostrophes
// (the real ImageNet names file, e.g. entry 731) and plain one-label-per-line
// text — plus the original boundaries: CR stripping, empty lines, brace-only
// lines, trailing-comma plain lines and the missing-file return.
//
// Only an explicit per-line dict shape (optional braces/whitespace, a numeric
// key with ':', a value wrapped in ONE quote style) may be unwrapped. Quotes
// embedded in plain text stay part of the label, and the whole shipped
// ImageNet names file is checked entry by entry against the repo's existing
// plain one-label-per-line oracle
// (samples/vision/resnet/test_data/imagenet1000_labels.txt, kept equal to
// ast.literal_eval of the dict file).
#include "file_io.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef REPO_ROOT
#define REPO_ROOT "."
#endif

#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)

namespace {

// Scratch directory uniquely owned by this process; only this path is ever
// cleaned. The suffix probe relies on create_directory being atomic, so two
// concurrent runs never share a directory and one run's cleanup can never
// delete another run's files (the previous fixed /tmp/c_utils_test_file_io
// plus a bare remove_all could).
std::filesystem::path g_temp_dir;

std::filesystem::path make_owned_temp_dir() {
  const auto base = std::filesystem::temp_directory_path();
  std::error_code ec;
  for (int suffix = 0; suffix < 10000; ++suffix) {
    std::filesystem::path candidate =
        base / ("c_utils_test_file_io-" + std::to_string(suffix));
    ec.clear();
    if (std::filesystem::create_directory(candidate, ec)) {
      return candidate;
    }
  }
  throw std::runtime_error("no free c_utils_test_file_io-<n> scratch dir");
}

const std::filesystem::path &temp_dir() {
  if (g_temp_dir.empty()) {
    g_temp_dir = make_owned_temp_dir();
  }
  return g_temp_dir;
}

void cleanup_temp_dir() {
  if (!g_temp_dir.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(g_temp_dir, ec);
    g_temp_dir.clear();
  }
}

std::filesystem::path write_file(const std::string &name,
                                 const std::string &content) {
  auto path = temp_dir() / name;
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
  out << content;
  return path;
}

void expect_labels(const std::vector<std::string> &got,
                   const std::vector<std::string> &want) {
  if (got != want) {
    std::cerr << "  got:";
    for (const auto &l : got)
      std::cerr << " [" << l << "]";
    std::cerr << "\n  want:";
    for (const auto &l : want)
      std::cerr << " [" << l << "]";
    std::cerr << std::endl;
  }
  EXPECT(got == want);
}

void test_dict_single_quote() {
  // The head of the shipped imagenet_classes.names: one dict entry per line.
  auto path = write_file("single.dict",
                         "{0: 'tench, Tinca tinca',\n"
                         " 1: 'goldfish, Carassius auratus',\n"
                         " 2: 'great white shark, white shark'}");
  expect_labels(load_linewise_labels(path.string()),
                {"tench, Tinca tinca", "goldfish, Carassius auratus",
                 "great white shark, white shark"});
}

void test_dict_double_quote_apostrophe() {
  // Real entries from imagenet_classes.names: double-quoted shells whose
  // text contains single apostrophes. The apostrophe must never be treated
  // as the label delimiter, and key/index prefixes must be stripped.
  auto path = write_file(
      "double.dict",
      "731: \"plunger, plumber's helper\",\n"
      "607: \"jack-o'-lantern\",\n"
      "986: \"yellow lady's slipper, yellow lady-slipper, Cypripedium "
      "calceolus, Cypripedium parviflorum\",\r\n"
      "319: \"dragonfly, darning needle, devil's darning needle, sewing "
      "needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk\",\n");
  expect_labels(load_linewise_labels(path.string()),
                {"plunger, plumber's helper", "jack-o'-lantern",
                 "yellow lady's slipper, yellow lady-slipper, Cypripedium "
                 "calceolus, Cypripedium parviflorum",
                 "dragonfly, darning needle, devil's darning needle, sewing "
                 "needle, snake feeder, snake doctor, mosquito hawk, skeeter "
                 "hawk"});
}

void test_plain_text_lines() {
  // One label per line: apostrophes, commas and spaces stay untouched.
  auto path = write_file("plain.names",
                         "golden retriever\n"
                         "plumber's helper\n"
                         "wolf, Arctic wolf\n");
  expect_labels(load_linewise_labels(path.string()),
                {"golden retriever", "plumber's helper", "wolf, Arctic wolf"});
}

void test_plain_text_embedded_double_quotes() {
  // Regression: quotes embedded in a plain label are part of the text, not a
  // format wrapper — the whole line must survive unchanged.
  auto path = write_file("embedded.names",
                         "a \"quoted\" label\n"
                         "the \"word\" and \"another\" here\n");
  expect_labels(load_linewise_labels(path.string()),
                {"a \"quoted\" label", "the \"word\" and \"another\" here"});
}

void test_plain_text_multiple_apostrophes_and_stray_quotes() {
  // Regression: a plain label with several apostrophes (or a stray quote) is
  // not a dict entry — none of it may be cut away.
  auto path = write_file("apostrophes.names",
                         "don't can't shan't\n"
                         "l'history of l'art\n"
                         "'tis the season\n"
                         "12:30 lunch\n"
                         "it's fine\n");
  expect_labels(load_linewise_labels(path.string()),
                {"don't can't shan't", "l'history of l'art", "'tis the season",
                 "12:30 lunch", "it's fine"});
}

void test_plain_text_trailing_comma_and_empty_lines() {
  // Original boundaries: a trailing comma on a plain line is trimmed, empty
  // lines (including CRLF-only lines) are skipped, CR is stripped. A line
  // that is only spaces has always been kept as a label (only fully empty
  // lines are skipped) — pinned here as pre-existing behavior, unchanged.
  auto path = write_file("boundaries.names",
                         "bulbul,\r\n"
                         "\r\n"
                         "\n"
                         "jay,\n"
                         "   \n");
  expect_labels(load_linewise_labels(path.string()),
                {"bulbul", "jay", "   "});
}

void test_brace_only_lines_skipped() {
  // A wrapped dict could put the closing brace on its own line; such a line
  // carries no label and must be skipped, not crash or emit "{"/"}".
  auto path = write_file("braces.dict",
                         "0: 'goldfish, Carassius auratus',\n"
                         "}\n"
                         "1: 'tench, Tinca tinca'\n");
  expect_labels(load_linewise_labels(path.string()),
                {"goldfish, Carassius auratus", "tench, Tinca tinca"});
}

void test_quoted_text_keeps_other_quote_style() {
  // Single-quoted dict value keeps inner double quotes; double-quoted dict
  // value keeps inner single quotes — what ast.literal_eval returns for the
  // same line.
  auto path = write_file("mixed.dict",
                         "0: 'say \"hi\" aloud',\n"
                         "1: \"it's 'fine' today\",\n");
  expect_labels(load_linewise_labels(path.string()),
                {"say \"hi\" aloud", "it's 'fine' today"});
}

void test_missing_file_returns_empty() {
  expect_labels(load_linewise_labels(temp_dir() / "does-not-exist.names"), {});
}

void test_temp_dir_uniquely_owned_and_cleaned_only() {
  // Two acquisitions never share one directory, and cleaning one owned
  // directory never deletes another run's files — the shape the old fixed
  // /tmp/c_utils_test_file_io plus bare remove_all got wrong.
  auto mine = make_owned_temp_dir();
  auto theirs = make_owned_temp_dir();
  EXPECT(mine != theirs);
  auto marker = mine / "marker.txt";
  {
    std::ofstream out(marker, std::ios::binary);
    out << "keep";
  }
  std::filesystem::remove_all(theirs);  // the other run cleans only its own
  EXPECT(std::filesystem::is_regular_file(marker));
  std::filesystem::remove_all(mine);    // this test cleans only its own
}

void test_shipped_imagenet_names_matches_python_parse() {
  // The two shipped ImageNet label files carry the same 1000 entries — the
  // main repo check already established that every line of the plain file
  // equals ast.literal_eval of the dict file (the Python reader's parse).
  // Reading both through the C++ reader must reproduce that equality entry
  // by entry: the dict file exercises linewise unwrapping, the plain file
  // exercises verbatim one-label-per-line reading.
  std::filesystem::path repo = REPO_ROOT;
  auto names = repo / "datasets/imagenet/imagenet_classes.names";
  auto plain = repo / "samples/vision/resnet/test_data/imagenet1000_labels.txt";
  EXPECT(std::filesystem::is_regular_file(names));
  EXPECT(std::filesystem::is_regular_file(plain));
  auto from_dict = load_linewise_labels(names.string());
  auto from_plain = load_linewise_labels(plain.string());
  EXPECT(from_plain.size() == 1000);
  EXPECT(from_dict.size() == from_plain.size());
  // Spot checks for failure clarity, then the full sweep.
  EXPECT(from_plain.at(0) == "tench, Tinca tinca");
  EXPECT(from_plain.at(731) == "plunger, plumber's helper");
  EXPECT(from_plain.at(607) == "jack-o'-lantern");
  EXPECT(from_plain.at(999) == "toilet tissue, toilet paper, bathroom tissue");
  for (size_t i = 0; i < from_plain.size() && i < from_dict.size(); ++i) {
    if (from_dict[i] != from_plain[i]) {
      std::cerr << "  entry " << i << ": dict [" << from_dict[i]
                << "] plain [" << from_plain[i] << "]" << std::endl;
      EXPECT(false);
    }
  }
}

} // namespace

int main() {
  struct Case {
    const char *name;
    void (*fn)();
  };
  const Case cases[] = {
      {"dict_single_quote", test_dict_single_quote},
      {"dict_double_quote_apostrophe", test_dict_double_quote_apostrophe},
      {"plain_text_lines", test_plain_text_lines},
      {"plain_text_embedded_double_quotes",
       test_plain_text_embedded_double_quotes},
      {"plain_text_multiple_apostrophes_and_stray_quotes",
       test_plain_text_multiple_apostrophes_and_stray_quotes},
      {"plain_text_trailing_comma_and_empty_lines",
       test_plain_text_trailing_comma_and_empty_lines},
      {"brace_only_lines_skipped", test_brace_only_lines_skipped},
      {"quoted_text_keeps_other_quote_style",
       test_quoted_text_keeps_other_quote_style},
      {"missing_file_returns_empty", test_missing_file_returns_empty},
      {"temp_dir_uniquely_owned_and_cleaned_only",
       test_temp_dir_uniquely_owned_and_cleaned_only},
      {"shipped_imagenet_names_matches_python_parse",
       test_shipped_imagenet_names_matches_python_parse},
  };
  for (const auto &c : cases) {
    try {
      c.fn();
    } catch (const std::exception &e) {
      std::cerr << "FAIL " << c.name << ": " << e.what() << std::endl;
      cleanup_temp_dir();
      return 1;
    }
    std::cout << "PASS " << c.name << std::endl;
  }
  cleanup_temp_dir();
  std::cout << "ALL " << sizeof(cases) / sizeof(cases[0]) << " CASES PASS"
            << std::endl;
  return 0;
}
