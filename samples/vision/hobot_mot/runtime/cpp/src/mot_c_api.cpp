// C ABI implementation over HobotMotTracker. Every entry point translates
// C++ exceptions into a return code plus message; no exception or abort ever
// crosses this boundary. See inc/mot_c_api.h for the contract.
#include "mot_c_api.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "track.hpp"

namespace {

using hobot_mot_sample::Detection;
using hobot_mot_sample::FrameTracks;
using hobot_mot_sample::HobotMotTracker;
using hobot_mot_sample::MotConfigError;
using hobot_mot_sample::MotInputError;
using hobot_mot_sample::TrackOptions;
using hobot_mot_sample::TrackState;

struct CapiTracker {
  std::unique_ptr<HobotMotTracker> tracker;
  FrameTracks last;  // results of the most recent mot_capi_process
};

void set_error(char *err, size_t err_len, const std::string &message) {
  if (err != nullptr && err_len > 0) {
    std::snprintf(err, err_len, "%s", message.c_str());
  }
}

template <typename Func>
int guarded(char *err, size_t err_len, Func &&func) {
  try {
    return func();
  } catch (const MotConfigError &exc) {
    set_error(err, err_len, exc.what());
    return MOT_CAPI_ERR_CONFIG;
  } catch (const MotInputError &exc) {
    set_error(err, err_len, exc.what());
    return MOT_CAPI_ERR_INPUT;
  } catch (const std::exception &exc) {
    set_error(err, err_len, exc.what());
    return MOT_CAPI_ERR_INTERNAL;
  } catch (...) {
    set_error(err, err_len, "unknown native error");
    return MOT_CAPI_ERR_INTERNAL;
  }
}

CapiTracker *as_tracker(mot_capi_handle_t handle) {
  return static_cast<CapiTracker *>(handle);
}

}  // namespace

extern "C" int mot_capi_create(const char *config_path, int32_t frame_width,
                               int32_t frame_height,
                               mot_capi_handle_t *out_handle, char *err,
                               size_t err_len) {
  return guarded(err, err_len, [&] {
    if (config_path == nullptr || out_handle == nullptr) {
      set_error(err, err_len, "config_path and out_handle are required");
      return MOT_CAPI_ERR_PARAM;
    }
    TrackOptions options;
    options.config_path = config_path;
    options.frame_width = frame_width > 0 ? frame_width : 960;
    options.frame_height = frame_height > 0 ? frame_height : 544;
    auto wrapper = std::make_unique<CapiTracker>();
    wrapper->tracker = std::make_unique<HobotMotTracker>(options);
    *out_handle = wrapper.release();
    return MOT_CAPI_OK;
  });
}

extern "C" int mot_capi_destroy(mot_capi_handle_t handle) {
  delete as_tracker(handle);  // delete of nullptr is a no-op
  return MOT_CAPI_OK;
}

extern "C" int mot_capi_process(mot_capi_handle_t handle, const float *boxes,
                                const float *scores, const int32_t *class_ids,
                                int32_t n, int64_t timestamp_ms,
                                int32_t frame_width, int32_t frame_height,
                                char *err, size_t err_len) {
  return guarded(err, err_len, [&] {
    CapiTracker *wrapper = as_tracker(handle);
    if (wrapper == nullptr || wrapper->tracker == nullptr) {
      set_error(err, err_len, "handle is not initialized");
      return MOT_CAPI_ERR_PARAM;
    }
    if (n < 0 || (n > 0 && (boxes == nullptr || scores == nullptr ||
                            class_ids == nullptr))) {
      set_error(err, err_len, "n must be >= 0 with non-null arrays");
      return MOT_CAPI_ERR_PARAM;
    }
    std::vector<Detection> detections;
    detections.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      Detection det;
      det.x1 = boxes[4 * static_cast<size_t>(i)];
      det.y1 = boxes[4 * static_cast<size_t>(i) + 1];
      det.x2 = boxes[4 * static_cast<size_t>(i) + 2];
      det.y2 = boxes[4 * static_cast<size_t>(i) + 3];
      det.score = scores[i];
      det.class_id = class_ids[i];
      detections.push_back(det);
    }
    wrapper->last = wrapper->tracker->predict(detections, timestamp_ms,
                                              frame_width, frame_height);
    return MOT_CAPI_OK;
  });
}

extern "C" int32_t mot_capi_result_count(mot_capi_handle_t handle) {
  const CapiTracker *wrapper = as_tracker(handle);
  if (wrapper == nullptr) {
    return 0;
  }
  return static_cast<int32_t>(wrapper->last.tracks.size());
}

extern "C" int mot_capi_copy_results(mot_capi_handle_t handle,
                                     int32_t *out_boxes, int32_t *out_ids,
                                     int32_t *out_states, int32_t n, char *err,
                                     size_t err_len) {
  return guarded(err, err_len, [&] {
    const CapiTracker *wrapper = as_tracker(handle);
    if (wrapper == nullptr) {
      set_error(err, err_len, "handle is not initialized");
      return MOT_CAPI_ERR_PARAM;
    }
    const auto &tracks = wrapper->last.tracks;
    if (n < static_cast<int32_t>(tracks.size()) ||
        (n > 0 && (out_boxes == nullptr || out_ids == nullptr ||
                   out_states == nullptr))) {
      set_error(err, err_len, "result buffers must hold result_count rows");
      return MOT_CAPI_ERR_CAPACITY;
    }
    for (size_t i = 0; i < tracks.size(); ++i) {
      out_boxes[4 * i] = tracks[i].x1;
      out_boxes[4 * i + 1] = tracks[i].y1;
      out_boxes[4 * i + 2] = tracks[i].x2;
      out_boxes[4 * i + 3] = tracks[i].y2;
      out_ids[i] = tracks[i].track_id;
      out_states[i] = static_cast<int32_t>(tracks[i].state);
    }
    return MOT_CAPI_OK;
  });
}

extern "C" int32_t mot_capi_disappeared_count(mot_capi_handle_t handle) {
  const CapiTracker *wrapper = as_tracker(handle);
  if (wrapper == nullptr) {
    return 0;
  }
  return static_cast<int32_t>(wrapper->last.removed_ids.size());
}

extern "C" int mot_capi_copy_disappeared(mot_capi_handle_t handle,
                                         int32_t *out_class_ids,
                                         int32_t *out_track_ids,
                                         int32_t pair_capacity, char *err,
                                         size_t err_len) {
  return guarded(err, err_len, [&] {
    const CapiTracker *wrapper = as_tracker(handle);
    if (wrapper == nullptr) {
      set_error(err, err_len, "handle is not initialized");
      return MOT_CAPI_ERR_PARAM;
    }
    const auto &removed = wrapper->last.removed_ids;
    if (pair_capacity < static_cast<int32_t>(removed.size()) ||
        (!removed.empty() &&
         (out_class_ids == nullptr || out_track_ids == nullptr))) {
      set_error(err, err_len,
                "disappeared buffers must hold disappeared_count pairs");
      return MOT_CAPI_ERR_CAPACITY;
    }
    for (size_t i = 0; i < removed.size(); ++i) {
      out_class_ids[i] = removed[i].class_id;
      out_track_ids[i] = removed[i].track_id;
    }
    return MOT_CAPI_OK;
  });
}

extern "C" int mot_capi_reset(mot_capi_handle_t handle, char *err,
                              size_t err_len) {
  return guarded(err, err_len, [&] {
    CapiTracker *wrapper = as_tracker(handle);
    if (wrapper == nullptr || wrapper->tracker == nullptr) {
      set_error(err, err_len, "handle is not initialized");
      return MOT_CAPI_ERR_PARAM;
    }
    wrapper->tracker->reset();
    wrapper->last = FrameTracks{};
    return MOT_CAPI_OK;
  });
}
