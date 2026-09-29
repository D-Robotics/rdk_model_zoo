#include "gemma4_text_inputs.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
namespace gemma4 {
TokenEmbeddings::TokenEmbeddings(const std::string&) {}
void TokenEmbeddings::Lookup(const std::vector<int64_t>& ids, float* out) const {std::fill(out,out+ids.size()*kHiddenSize,0.0f);}
}
int main() {
  try { gemma4::TokenEmbeddings embeddings("fixture");
    auto batch=gemma4::PrepareBatchInputs(embeddings,{11},4096,1,nullptr,gemma4::kChunkSize);
    std::cout << "unexpected success: " << batch.positions[0] << "\n";
  } catch(const std::exception& e) {std::cout<<"rejected: "<<e.what()<<"\n";}
}
