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
