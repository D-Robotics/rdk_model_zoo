#include "text_fixture.hpp"
#include <iostream>
int main() {
  try { gemma4::TextEngine engine("fixture","fixture");
    std::vector<float> short_hidden(1,0.0f);
    auto out=engine.ContinueGenerate({11,22},1,&short_hidden);
    std::cout<<"unexpected success: "<<out.size()<<"\n";
  } catch(const std::exception& e) {std::cout<<"rejected: "<<e.what()<<"\n";}
}
