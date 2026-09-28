// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "audio_io.h"
#include "contract.h"
namespace asr {
// Independent-window SRC_SINC_BEST_QUALITY, no state between chunks.
// Includes channel averaging, var+1e-5 normalization, then padding to 30000.
PreparedChunk prepare_chunk(const AudioChunk &chunk);
} // namespace asr
