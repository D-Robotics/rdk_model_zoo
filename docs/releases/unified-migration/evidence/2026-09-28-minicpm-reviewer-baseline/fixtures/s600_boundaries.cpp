#include "minicpm5.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
int main(int argc,char**argv){if(argc!=3)return 2;std::filesystem::path dir=argv[1];std::filesystem::create_directories(dir/"tmp");setenv("TMPDIR",(dir/"tmp").c_str(),1);for(auto name:{"MiniCPM5-2B_language_chunk_256_cache_4096_w8_nash-p_corenum_4_4.hbm","MiniCPM5-2B_embed_tokens.bin","tokenizer.json","tokenizer_config.json"})std::ofstream(dir/name)<<"fixture";minicpm5::Config c;c.model_path=dir.string();if(std::string(argv[2])=="metric"){minicpm5::MiniCPM5 model(c);auto r=model.Generate("hello");std::cout<<"accepted_nonfinite_ttft="<<(!std::isfinite(r.ttft_ms))<<" accepted_negative_decode="<<(r.decode_tps<0)<<"\n";return !std::isfinite(r.ttft_ms)||r.decode_tps<0?1:0;}oellm::throw_init=true;try{minicpm5::MiniCPM5 model(c);}catch(const std::exception&){size_t count=0;for(auto const& item:std::filesystem::directory_iterator(dir/"tmp")){(void)item;++count;}std::cout<<"temp_files_after_Init_exception="<<count<<"\n";return count?1:0;}return 2;}
