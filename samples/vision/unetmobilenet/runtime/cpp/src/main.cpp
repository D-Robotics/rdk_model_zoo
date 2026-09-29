// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "model_runner.hpp"
#include "unetmobilenet.hpp"
#include "visualization.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <opencv2/imgcodecs.hpp>

namespace {
std::string quoted(const std::string& value) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:value) {
        if(c=='"' || c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<int>(c)<<std::dec;
        else out<<c;
    }
    out<<'"';return out.str();
}
void parent_directory(const std::string& path) {
    const auto parent=std::filesystem::path(path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
}
}
int main(int argc,char** argv) {
    try {
        std::map<std::string,std::string> args{{"--alpha-f","0.75"},{"--img-save-path","result.jpg"},
            {"--mask-save-path","unetmobilenet_mask.png"},{"--report-path","unetmobilenet_cpp_report.json"},
            {"--priority","0"},{"--bpu-core","-1"},{"--target",""},{"--model-path",""},{"--test-img",""}};
        for(int i=1;i<argc;++i) {
            std::string flag=argv[i];std::replace(flag.begin(),flag.end(),'_','-');
            if(flag=="--help" || flag=="-h") {
                std::cout<<"Required: --target s100|s600 --model-path file.hbm --test-img image\n"
                         <<"Options: --alpha-f 0.75 --img-save-path result.jpg --mask-save-path unetmobilenet_mask.png\n"
                         <<"         --report-path unetmobilenet_cpp_report.json --priority 0 --bpu-core -1\n";
                return 0;
            }
            if(!args.count(flag) || i+1>=argc)throw std::invalid_argument("Unknown option or missing value: "+flag);
            args[flag]=argv[++i];
        }
        for(const auto* name:{"--target","--model-path","--test-img"})
            if(args[name].empty())throw std::invalid_argument(std::string("Required: ")+name);
        std::size_t parsed=0;
        const double alpha=std::stod(args["--alpha-f"],&parsed);
        if(parsed!=args["--alpha-f"].size() || !std::isfinite(alpha) || alpha<0 || alpha>1)
            throw std::invalid_argument("alpha-f must be finite and in [0,1]");
        const int priority=std::stoi(args["--priority"],&parsed);
        if(parsed!=args["--priority"].size())throw std::invalid_argument("priority must be an integer");
        const int core=std::stoi(args["--bpu-core"],&parsed);
        if(parsed!=args["--bpu-core"].size())throw std::invalid_argument("bpu-core must be an integer");
        if(std::filesystem::path(args["--mask-save-path"]).extension()!=".png")
            throw std::invalid_argument("mask-save-path must end in .png");
        const auto image=cv::imread(args["--test-img"],cv::IMREAD_COLOR);
        if(image.empty())throw std::invalid_argument("Cannot read input image");
        unetmobilenet::ModelRunner runner(args["--model-path"],args["--target"],priority,core);
        unetmobilenet::UnetMobileNetTask task([&](const cv::Mat& y,const cv::Mat& uv){return runner.run(y,uv);});
        const auto prepared=task.pre_process(image);
        const auto raw=task.forward(prepared);
        const auto labels=task.post_process(raw,prepared.context);
        const auto overlay=unetmobilenet::render_overlay(image,labels,alpha);
        cv::Mat png;labels.convertTo(png,CV_8U);
        for(const auto* name:{"--img-save-path","--mask-save-path","--report-path"})parent_directory(args[name]);
        if(!cv::imwrite(args["--img-save-path"],overlay) || !cv::imwrite(args["--mask-save-path"],png))
            throw std::runtime_error("Cannot save image/mask");
        std::ostringstream report;
        report<<"{\n  \"target\": "<<quoted(args["--target"])
              <<",\n  \"asset_id\": "<<quoted("s:unetmobilenet:"+args["--target"]+"/unet_mobilenet_1024x2048_nv12.hbm")
              <<",\n  \"model_path\": "<<quoted(args["--model-path"])
              <<",\n  \"input_path\": "<<quoted(args["--test-img"])
              <<",\n  \"publisher_sha256\": null,\n  \"runtime_version\": \"unknown\","
              <<"\n  \"mask_shape\": ["<<labels.rows<<","<<labels.cols<<"],"
              <<"\n  \"score_shape\": [1,"<<raw.spec.shape[1]<<","<<raw.spec.shape[2]<<",19],"
              <<"\n  \"score_dtype\": "<<quoted(raw.spec.type==unetmobilenet::ScoreType::Int32?"int32":"float32")
              <<",\n  \"scaled\": "<<(raw.spec.scaled?"true":"false")
              <<",\n  \"alpha_f\": "<<alpha<<",\n  \"priority\": "<<priority<<",\n  \"bpu_core\": "<<core
              <<",\n  \"img_save_path\": "<<quoted(args["--img-save-path"])
              <<",\n  \"mask_save_path\": "<<quoted(args["--mask-save-path"])<<"\n}\n";
        std::ofstream stream(args["--report-path"]);stream<<report.str();stream.close();
        if(!stream)throw std::runtime_error("Cannot write report");
        std::cout<<report.str();return 0;
    } catch(const std::exception& error) {
        std::cerr<<"error: "<<error.what()<<'\n';return 2;
    }
}
