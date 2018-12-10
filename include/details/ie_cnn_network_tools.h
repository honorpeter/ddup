// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header file for CNNNetwork tools
 * @file ie_cnn_network_tools.h
 */
#pragma once
#include <vector>
#include "ie_common.h"
#include "ie_icnn_network.hpp"

namespace InferenceEngine {
namespace details {

INFERENCE_ENGINE_API_CPP(std::vector<CNNLayerPtr>) CNNNetSortTopologically(const ICNNNetwork & network);

}  // namespace details
}  // namespace InferenceEngine
