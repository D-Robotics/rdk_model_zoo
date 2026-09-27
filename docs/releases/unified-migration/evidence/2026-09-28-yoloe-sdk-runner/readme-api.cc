#include "yoloe.h"
yoloe::Result process_image(yoloe::Config config,
                            std::unique_ptr<yoloe::Runner> backend,
                            const cv::Mat& image) {
    yoloe::YOLOE task(config, std::move(backend));
    auto prepared = task.pre_process(image);
    auto raw = task.infer(prepared);
    return task.post_process(raw);
    // task.predict(image) composes the same three operations.
}

#include "sdk_runner.h"
#include "yoloe.h"
yoloe::Result process_sdk_image(const cv::Mat& image, yoloe::SdkModel model,
                                yoloe::SdkPreflight preflight) {
    auto backend = std::make_unique<yoloe::SdkRunner>(model, std::move(preflight));
    yoloe::Config config;
    config.protocol = backend->protocol();
    yoloe::YOLOE task(config, std::move(backend));
    return task.predict(image);
}
