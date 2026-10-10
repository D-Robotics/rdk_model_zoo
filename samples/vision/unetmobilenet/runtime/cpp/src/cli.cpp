// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// UnetMobileNet CLI: option parsing, image loading, overlay rendering and the
// image/mask/report writer. No model or SDK work.
#include "cli.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace unetmobilenet {
namespace {
void parent_directory(const std::string& path) {
    const auto parent=std::filesystem::path(path).parent_path();
    if(!parent.empty())std::filesystem::create_directories(parent);
}
}

CliOptions parse_options(const std::vector<std::string>& args) {
    CliOptions out;
    std::map<std::string,std::string> values{{"--alpha-f","0.75"},{"--img-save-path","result.jpg"},
        {"--mask-save-path","unetmobilenet_mask.png"},{"--report-path","unetmobilenet_cpp_report.json"},
        {"--priority","0"},{"--bpu-core","-1"},{"--target",""},{"--model-path",""},{"--test-img",""}};
    for(std::size_t i=0;i<args.size();++i) {
        std::string flag=args[i];
        std::replace(flag.begin(),flag.end(),'_','-');
        if(flag=="--help" || flag=="-h") { out.help=true;return out; }
        if(!values.count(flag) || i+1>=args.size())
            throw std::invalid_argument("Unknown option or missing value: "+flag);
        values[flag]=args[++i];
    }
    for(const auto* name:{"--target","--model-path","--test-img"})
        if(values[name].empty())throw std::invalid_argument(std::string("Required: ")+name);
    std::size_t parsed=0;
    const double alpha=std::stod(values["--alpha-f"],&parsed);
    if(parsed!=values["--alpha-f"].size() || !std::isfinite(alpha) || alpha<0 || alpha>1)
        throw std::invalid_argument("alpha-f must be finite and in [0,1]");
    const int priority=std::stoi(values["--priority"],&parsed);
    if(parsed!=values["--priority"].size())throw std::invalid_argument("priority must be an integer");
    const int core=std::stoi(values["--bpu-core"],&parsed);
    if(parsed!=values["--bpu-core"].size())throw std::invalid_argument("bpu-core must be an integer");
    if(std::filesystem::path(values["--mask-save-path"]).extension()!=".png")
        throw std::invalid_argument("mask-save-path must end in .png");
    out.model_path=values["--model-path"];
    out.test_img=values["--test-img"];
    out.target=values["--target"];
    out.alpha_f=alpha;
    out.img_save_path=values["--img-save-path"];
    out.mask_save_path=values["--mask-save-path"];
    out.report_path=values["--report-path"];
    out.priority=priority;
    out.bpu_core=core;
    return out;
}

void print_help() {
    std::cout<<"Required: --target s100|s600 --model-path file.hbm --test-img image\n"
             <<"Options: --alpha-f 0.75 --img-save-path result.jpg --mask-save-path unetmobilenet_mask.png\n"
             <<"         --report-path unetmobilenet_cpp_report.json --priority 0 --bpu-core -1\n";
}

cv::Mat load_image(const std::string& path) {
    const auto image=cv::imread(path,cv::IMREAD_COLOR);
    if(image.empty())throw std::invalid_argument("Cannot read input image");
    return image;
}

std::string json_quote(const std::string& value) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:value) {
        if(c=='"' || c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<int>(c)<<std::dec;
        else out<<c;
    }
    out<<'"';return out.str();
}

cv::Mat render_overlay(const cv::Mat& image,const cv::Mat& labels,double alpha_f) {
    if(image.empty() || image.type()!=CV_8UC3 || labels.type()!=CV_32S || labels.size()!=image.size())
        throw std::invalid_argument("Overlay requires original-sized BGR image and int32 labels");
    if(!std::isfinite(alpha_f) || alpha_f<0 || alpha_f>1)throw std::invalid_argument("alpha-f must be 0..1");
    static const cv::Vec3b colors[19]={
        {56,56,255},{151,157,255},{31,112,255},{29,178,255},{49,210,207},
        {10,249,72},{23,204,146},{134,219,61},{52,147,26},{187,212,0},
        {168,153,44},{255,194,0},{147,69,52},{255,115,100},{236,24,0},
        {255,56,132},{133,0,82},{255,56,203},{200,149,255}
    };
    cv::Mat colored(image.size(),CV_8UC3),result;
    for(int y=0;y<labels.rows;++y)for(int x=0;x<labels.cols;++x) {
        const int id=labels.at<std::int32_t>(y,x);
        if(id<0 || id>=19)throw std::invalid_argument("Class ID is outside 0..18");
        colored.at<cv::Vec3b>(y,x)=colors[id];
    }
    cv::addWeighted(image,alpha_f,colored,1-alpha_f,0,result);
    return result;
}

void save_results(const CliOptions& options,const cv::Mat& image,const cv::Mat& labels,const ScoreSpec& scores) {
    const auto overlay=render_overlay(image,labels,options.alpha_f);
    cv::Mat png;labels.convertTo(png,CV_8U);
    parent_directory(options.img_save_path);
    parent_directory(options.mask_save_path);
    parent_directory(options.report_path);
    if(!cv::imwrite(options.img_save_path,overlay) || !cv::imwrite(options.mask_save_path,png))
        throw std::runtime_error("Cannot save image/mask");
    std::ostringstream report;
    report<<"{\n  \"target\": "<<json_quote(options.target)
          <<",\n  \"asset_id\": "<<json_quote("s:unetmobilenet:"+options.target+"/unet_mobilenet_1024x2048_nv12.hbm")
          <<",\n  \"model_path\": "<<json_quote(options.model_path)
          <<",\n  \"input_path\": "<<json_quote(options.test_img)
          <<",\n  \"publisher_sha256\": null,\n  \"runtime_version\": \"unknown\","
          <<"\n  \"mask_shape\": ["<<labels.rows<<","<<labels.cols<<"],"
          <<"\n  \"score_shape\": [1,"<<scores.shape[1]<<","<<scores.shape[2]<<",19],"
          <<"\n  \"score_dtype\": "<<json_quote(scores.type==ScoreType::Int32?"int32":"float32")
          <<",\n  \"scaled\": "<<(scores.scaled?"true":"false")
          <<",\n  \"alpha_f\": "<<options.alpha_f<<",\n  \"priority\": "<<options.priority<<",\n  \"bpu_core\": "<<options.bpu_core
          <<",\n  \"img_save_path\": "<<json_quote(options.img_save_path)
          <<",\n  \"mask_save_path\": "<<json_quote(options.mask_save_path)<<"\n}\n";
    std::ofstream stream(options.report_path);stream<<report.str();stream.close();
    if(!stream)throw std::runtime_error("Cannot write report");
    std::cout<<report.str();
}
}  // namespace unetmobilenet
