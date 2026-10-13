// Visible entry point: parse arguments, build the named tracker, replay the
// detection sequence, hand results to the CLI for rendering and saving.
#include <exception>
#include <iostream>
#include <vector>

#include "cli.hpp"
#include "track.hpp"

int main(int argc, char **argv) {
  using hobot_mot_sample::FrameTracks;
  using hobot_mot_sample::HobotMotTracker;
  using hobot_mot_sample::Options;
  using hobot_mot_sample::TrackOptions;
  try {
    const Options options =
        hobot_mot_sample::parse_options({argv + 1, argv + argc});
    const auto records = hobot_mot_sample::load_frame_records(options.input_path);

    TrackOptions track_options;
    track_options.config_path = options.config_path;
    track_options.frame_width = options.frame_width;
    track_options.frame_height = options.frame_height;
    HobotMotTracker tracker(track_options);

    std::vector<FrameTracks> results;
    results.reserve(records.size());
    for (const auto &record : records) {
      results.push_back(tracker.predict(record.detections, record.timestamp_ms,
                                        record.image_width,
                                        record.image_height));
    }
    hobot_mot_sample::render_frames(results, records);
    if (!options.output_path.empty()) {
      hobot_mot_sample::save_results(options.output_path, records, results);
      std::cout << "results written: " << options.output_path << "\n";
    }
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "Error: " << exc.what() << "\n";
    return 2;
  }
}
