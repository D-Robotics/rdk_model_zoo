/* C ABI over the HobotMot tracker sample (see inc/track.hpp).
 *
 * This boundary is shared with the Python runtime: runtime/python/native.py
 * loads libhobot_mot_capi.so through ctypes and forwards detections with the
 * same semantics as the C++ entry point. It owns the complete translation
 * between plain C buffers and the C++ tracker, so no other file does.
 *
 * Ownership: the handle belongs to the caller and is released exactly by
 * mot_capi_destroy. All array arguments are caller-allocated; results of the
 * last mot_capi_process call are kept inside the handle until the next call
 * and copied out by the mot_capi_copy_* functions, which never mutate tracker
 * state and can be retried with larger buffers. Every function catches C++
 * exceptions and reports through the return code plus a message buffer; no
 * exception crosses this boundary and no function calls abort().
 */
#ifndef RDK_MODEL_ZOO_HOBOT_MOT_C_API_H_
#define RDK_MODEL_ZOO_HOBOT_MOT_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes returned by every function. */
#define MOT_CAPI_OK 0
#define MOT_CAPI_ERR_PARAM 1      /* invalid argument value */
#define MOT_CAPI_ERR_CONFIG 2     /* tracker config rejected */
#define MOT_CAPI_ERR_INPUT 3      /* detections/frame rejected */
#define MOT_CAPI_ERR_STATE 4      /* wrong call order on the handle */
#define MOT_CAPI_ERR_CAPACITY 5   /* copy buffer too small; nothing written */
#define MOT_CAPI_ERR_INTERNAL 6   /* C++ exception or upstream failure */

/* Upstream per-detection states (hobot_mot::DataState). */
#define MOT_CAPI_STATE_VALID 0
#define MOT_CAPI_STATE_INVALID 4

/* Opaque tracker handle; one handle per video stream. */
typedef void *mot_capi_handle_t;

/* Create a tracker. config_path must point to a validated IOU_2.0 JSON
 * config (the same checks as the C++ TrackOptions run here); err receives a
 * NUL-terminated message on failure. */
int mot_capi_create(const char *config_path, int32_t frame_width,
                    int32_t frame_height, mot_capi_handle_t *out_handle,
                    char *err, size_t err_len);

/* Release a handle; safe on NULL. */
int mot_capi_destroy(mot_capi_handle_t handle);

/* Feed one frame. boxes is n*4 floats (x1, y1, x2, y2 pixel coordinates),
 * scores and class_ids are n elements each, timestamp_ms must strictly
 * increase between calls on the same handle. Results stay in the handle
 * until the next process call. */
int mot_capi_process(mot_capi_handle_t handle, const float *boxes,
                     const float *scores, const int32_t *class_ids,
                     int32_t n, int64_t timestamp_ms, int32_t frame_width,
                     int32_t frame_height, char *err, size_t err_len);

/* Number of result rows of the last process call (== n of that call). */
int32_t mot_capi_result_count(mot_capi_handle_t handle);

/* Copy per-detection results of the last process call: out_boxes n*4 ints in
 * input order, out_ids and out_states n ints. Returns MOT_CAPI_ERR_CAPACITY
 * when n is smaller than the stored row count (nothing is written). */
int mot_capi_copy_results(mot_capi_handle_t handle, int32_t *out_boxes,
                          int32_t *out_ids, int32_t *out_states, int32_t n,
                          char *err, size_t err_len);

/* Number of (class_id, track_id) pairs removed in the last process call. */
int32_t mot_capi_disappeared_count(mot_capi_handle_t handle);

/* Copy removed identities as class_id, track_id pairs (2 ints each). */
int mot_capi_copy_disappeared(mot_capi_handle_t handle,
                              int32_t *out_class_ids, int32_t *out_track_ids,
                              int32_t pair_capacity, char *err,
                              size_t err_len);

/* Drop all tracker state: class instances are rebuilt with the same config
 * and track ids restart from 1, exactly like a fresh handle. */
int mot_capi_reset(mot_capi_handle_t handle, char *err, size_t err_len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* RDK_MODEL_ZOO_HOBOT_MOT_C_API_H_ */
