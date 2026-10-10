// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// Entry point: parse options, construct the model, predict, save results.
#include "cli.hpp"
#include "segment.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    try {
        const auto options=unetmobilenet::parse_options({argv+1,argv+argc});
        if(options.help) { unetmobilenet::print_help();return 0; }
        const auto image=unetmobilenet::load_image(options.test_img);
        unetmobilenet::UnetMobileNet model(options.model_path,options.target,options.priority,options.bpu_core);
        const auto labels=model.predict(image);
        unetmobilenet::save_results(options,image,labels,model.score_spec());
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"error: "<<error.what()<<'\n';return 2;
    }
}
