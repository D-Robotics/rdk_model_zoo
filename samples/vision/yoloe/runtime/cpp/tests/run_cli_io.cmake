# Unique test-owned temporary path; never remove user result directories.
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef suffix)
set(output "${TEST_OUTPUT}-${suffix}")
execute_process(COMMAND "${TEST_BINARY}" "${output}" "${TEST_LABELS}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "CLI I/O test failed (${result}); artifacts retained: ${output}")
endif()
file(REMOVE_RECURSE "${output}")
