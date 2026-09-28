#include "gemma4_model_io.hpp"
#include <cstdlib>
#include <new>
#include <cstdio>
static bool fail_next=false;
static int released=0;
void* operator new(std::size_t n) { if(fail_next){fail_next=false;throw std::bad_alloc();} if(void* p=std::malloc(n))return p;throw std::bad_alloc(); }
void operator delete(void* p) noexcept {std::free(p);}
int hbUCPFree(hbUCPSysMem* m){if(m->virAddr){++released;std::free(m->virAddr);m->virAddr=nullptr;}return 0;}
int main(){
 int defects=0;
 for(int output=0;output<2;++output){
  void* buffer=std::malloc(16); int before=released; bool caught=false;
  {
   gemma4::ModelIo io;
   if(output)io.outputs.reserve(1);else io.inputs.reserve(1);
   hbDNNTensor t{};t.properties.alignedByteSize=16;t.sysMem.virAddr=buffer;
   fail_next=true;
   try{if(output)io.AddOutput(t);else io.AddInput(t);}catch(const std::bad_alloc&){caught=true;}
   fail_next=false;
  }
  std::printf("%s caught=%d released=%d\n",output?"output":"input",caught,released-before);
  if(!caught||released-before!=1){++defects;if(released==before)std::free(buffer);}
 }
 return defects?1:0;
}
