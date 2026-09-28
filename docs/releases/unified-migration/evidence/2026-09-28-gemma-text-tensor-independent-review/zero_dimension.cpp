#include "gemma4_text_tensor.hpp"
#include "gemma4_config.hpp"
#include <iostream>
#include <exception>
int main() {
 hbDNNTensorProperties p{};
 p.tensorType=HB_DNN_TENSOR_TYPE_F32; p.quantiType=NONE;
 p.validShape.numDimensions=2; p.validShape.dimensionSize[0]=0;
 p.validShape.dimensionSize[1]=gemma4::kHiddenSize;
 p.stride[1]=4; p.stride[0]=gemma4::kHiddenSize*4; p.alignedByteSize=p.stride[0];
 try { gemma4::ValidateTextTensor(p,gemma4::TextTensorRole::kInputsEmbeds,gemma4::kChunkSize); }
 catch(const std::exception& e) { std::cout << "rejected: " << e.what() << '\n';return 0; }
 std::cerr << "unexpected acceptance\n";return 1;
}
