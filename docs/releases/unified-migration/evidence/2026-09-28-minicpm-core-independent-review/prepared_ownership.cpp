#include "minicpm5.hpp"
#include <iostream>
#include <utility>
int main() {
 auto request = prepare_request("short", "template");
 const bool returned = request.input.requests == &request.request && request.request.prompt == request.prompt.c_str() && request.request.chat_template == request.chat_template.c_str();
 auto copied = request;
 const bool copy = copied.input.requests == &copied.request && copied.request.prompt == copied.prompt.c_str() && copied.request.chat_template == copied.chat_template.c_str();
 auto moved = std::move(request);
 const bool move = moved.input.requests == &moved.request && moved.request.prompt == moved.prompt.c_str() && moved.request.chat_template == moved.chat_template.c_str();
 std::cout << "return pointers owned=" << returned << " copy pointers owned=" << copy << " move pointers owned=" << move << "\n";
 return returned && copy && move ? 0 : 1;
}
