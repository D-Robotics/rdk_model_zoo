# CORE-N1: syntax compilation of the four README example blocks, extracted verbatim
# sources: /Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration/docs/releases/unified-migration/evidence/2026-09-28-b11-minicpm-core-refactor/readme-examples/*.cpp (first cpp block of each README, unmodified)

$ c++ -std=c++17 -Wall -Wextra -Werror -fno-elide-constructors -fsyntax-only -I tests/native/fixtures -I runtime/legacy/inc readme-examples/legacy_readme_en.cpp
  rc=0
$ c++ -std=c++17 -Wall -Wextra -Werror -fno-elide-constructors -fsyntax-only -I tests/native/fixtures -I runtime/legacy/inc readme-examples/legacy_readme_cn.cpp
  rc=0
$ c++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only -I tests/native/fixtures -I runtime/cpp/inc -I /Users/Max/Workspace/company/development/RDK_MODEL_ZOO/.coordination/asr-json/include readme-examples/cpp_readme_en.cpp
  rc=0
$ c++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only -I tests/native/fixtures -I runtime/cpp/inc -I /Users/Max/Workspace/company/development/RDK_MODEL_ZOO/.coordination/asr-json/include readme-examples/cpp_readme_cn.cpp
  rc=0
