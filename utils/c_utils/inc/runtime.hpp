/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hb_ucp_status.h"
#include "hb_dnn_status.h"

/**
 * @def HBDNN_CHECK_SUCCESS
 * @brief Check a HBDNN API call for success and throw std::runtime_error on failure.
 * @param func_call  HBDNN API call expression.                        // (in)
 * @param context_str Text to append for easier debugging.             // (in)
 */
#define HBDNN_CHECK_SUCCESS(func_call, context_str)                         \
    do {                                                                    \
        int32_t __err_code = (func_call);                                   \
        if (__err_code != 0) {                                              \
            const char* __err_desc =                                        \
                hbDNNGetErrorDesc(__err_code);                              \
            fprintf(stderr,                                                 \
                    "DNN Error (code=%d, desc=%s): %s\n",                  \
                    __err_code,                                             \
                    (__err_desc ? __err_desc : "null"),                     \
                    (context_str));                                         \
            return __err_code;                                              \
        }                                                                   \
    } while (0)

/**
 * @def HBUCP_CHECK_SUCCESS
 * @brief Check a UCP API call for success, print error log, and return error code on failure.
 *
 * @param func_call    UCP API call expression.            // (in)
 * @param context_str  Context string for easier debugging. // (in)
 *
 * @note The enclosing function must return int32_t.
 */
#define HBUCP_CHECK_SUCCESS(func_call, context_str)                              \
    do {                                                                         \
        int32_t __err_code = (func_call);                                        \
        if (__err_code != 0) {                                                   \
            const char* __err_desc = hbUCPGetErrorDesc(__err_code);              \
            fprintf(stderr,                                                     \
                    "UCP Error (code=%d, desc=%s): %s\n",                        \
                    __err_code,                                                  \
                    (__err_desc ? __err_desc : "null"),                          \
                    (context_str));                                              \
            /* fflush(stderr); */                                                \
            return __err_code;                                                   \
        }                                                                        \
    } while (0)
