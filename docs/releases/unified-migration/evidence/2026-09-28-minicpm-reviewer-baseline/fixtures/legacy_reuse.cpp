#include "minicpm5.hpp"
#include <fstream>
#include <iostream>
static Callback cb;static int calls=0;
param_t xlm_create_default_param(){return {};}
int xlm_init(param_t*,Callback callback,xlm_handle_t* h){cb=callback;*h=new int(1);return 0;}
int xlm_infer(xlm_handle_t,xlm_input_t*,void* u){if(++calls==1)cb(nullptr,XLM_STATE_END,u);return 0;}
int xlm_destroy(xlm_handle_t* h){delete static_cast<int*>(*h);*h=nullptr;return 0;}
int main(int argc,char**argv){if(argc!=2)return 2;std::ofstream(argv[1])<<"template";MiniCPM5Config c;c.template_path=argv[1];MiniCPM5 model(c);model.init();const int first=model.predict();model.init();const int second=model.predict();std::cout<<"first="<<first<<" second_without_END="<<second<<"\n";return second==0?1:0;}
