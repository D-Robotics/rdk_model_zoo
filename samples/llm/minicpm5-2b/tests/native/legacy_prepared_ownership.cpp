// CORE-R1 regression: PreparedRequest owns its own SDK pointers through
// non-elided return, copy and move, for short (SSO) and long strings.
// Compiled with -fno-elide-constructors so no optimization can mask the bug.
#include <iostream>
#include <string>
#include <utility>

#include "minicpm5.hpp"

namespace {
bool self_owned(const PreparedRequest& prepared) {
  return prepared.input.requests == &prepared.request &&
         prepared.request.prompt == prepared.prompt.c_str() &&
         prepared.request.chat_template == prepared.chat_template.c_str() &&
         prepared.input.request_num == 1;
}

std::string long_text(const char* seed) {
  std::string text;
  for (int i = 0; i < 12; ++i) text += seed;  // Far beyond the SSO buffer.
  return text;
}

// Copies through a function boundary; -fno-elide-constructors forces real
// copy/move constructor calls on every return below.
PreparedRequest copied(const PreparedRequest& source) { return source; }
PreparedRequest moved(PreparedRequest source) { return source; }
}  // namespace

int failures = 0;
#define CHECK(condition, step)              \
  do {                                      \
    if (!(condition)) {                     \
      std::cout << "FAIL " << step << "\n"; \
      ++failures;                           \
    }                                       \
  } while (0)

template <typename Make>
void exercise(const char* label, Make make, const std::string& prompt,
              const std::string& templ) {
  // Non-elided return from prepare_request.
  PreparedRequest returned = make(prompt, templ);
  CHECK(self_owned(returned), (label + std::string(": returned")).c_str());
  CHECK(returned.request.request_id == 0 &&
            returned.request.type == XLM_INPUT_PROMPT &&
            returned.request.new_chat &&
            returned.request.infer_backend == XLM_INFER_BACKEND_BPU_ANY,
        (label + std::string(": returned fields")).c_str());
  CHECK(std::string(returned.request.prompt) == prompt &&
            std::string(returned.request.chat_template) == templ,
        (label + std::string(": returned content")).c_str());

  // Copy constructor and copy assignment.
  PreparedRequest copy_constructed(returned);
  CHECK(self_owned(copy_constructed) && self_owned(returned),
        (label + std::string(": copy ctor")).c_str());
  CHECK(copy_constructed.request.prompt != returned.request.prompt,
        (label + std::string(": copy is independent")).c_str());
  PreparedRequest copy_assigned;
  copy_assigned = returned;
  CHECK(self_owned(copy_assigned),
        (label + std::string(": copy assign")).c_str());

  // Move constructor and move assignment; the moved-from carrier stays
  // self-consistent (pointers into its own, now empty, strings).
  PreparedRequest move_constructed(std::move(returned));
  CHECK(self_owned(move_constructed),
        (label + std::string(": move ctor")).c_str());
  CHECK(self_owned(returned),
        (label + std::string(": moved-from rebound")).c_str());
  PreparedRequest move_assigned;
  move_assigned = moved(move_constructed);
  CHECK(self_owned(move_assigned) && self_owned(move_constructed),
        (label + std::string(": move assign")).c_str());
  CHECK(std::string(move_assigned.request.prompt) == prompt,
        (label + std::string(": moved content")).c_str());
}

int main() {
  // Short strings live in the SSO buffer; long strings come from the heap.
  exercise("short", prepare_request, "short", "template");
  exercise("long", prepare_request, long_text("提示词"),
           long_text("{template}"));

  // Copy independence: rebinding the copy must not disturb the original.
  {
    const std::string templ = long_text("template");
    PreparedRequest original("original prompt", templ);
    PreparedRequest duplicate(original);
    duplicate.bind();
    CHECK(original.request.prompt == original.prompt.c_str() &&
              duplicate.request.prompt == duplicate.prompt.c_str(),
          "independent bindings after copy");
    CHECK(
        original.request.chat_template == original.chat_template.c_str() &&
            duplicate.request.chat_template == duplicate.chat_template.c_str(),
        "independent template bindings after copy");
  }

  // Copy and move across function returns, non-elided.
  {
    const std::string templ = long_text("template");
    PreparedRequest original("through boundary", templ);
    PreparedRequest via_copy = copied(original);
    PreparedRequest via_move = moved(std::move(original));
    CHECK(self_owned(via_copy) && self_owned(original),
          "function-boundary copy owned by both sides");
    CHECK(self_owned(via_move) && self_owned(original),
          "function-boundary move owned and source rebound");
    CHECK(std::string(via_copy.request.prompt) == "through boundary" &&
              std::string(via_move.request.prompt) == "through boundary",
          "function-boundary content preserved");
  }
  if (failures) {
    std::cout << failures << " ownership checks failed\n";
    return 1;
  }
  std::cout << "legacy_prepared_ownership OK\n";
  return 0;
}
