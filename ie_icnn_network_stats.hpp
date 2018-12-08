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
 * @brief This is a header file for the ICNNNetworkStats class
 * @file ie_icnn_network_stats.hpp
 */
#pragma once

#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <map>
#include "details/ie_irelease.hpp"

namespace InferenceEngine {

class NetworkNodeStats;

using NetworkNodeStatsPtr = std::shared_ptr<NetworkNodeStats>;
using NetworkNodeStatsWeakPtr = std::weak_ptr<NetworkNodeStats>;
using NetworkStatsMap = std::map<std::string, NetworkNodeStatsPtr>;
/**
 * @class ICNNNetworkStats
 * @brief This is the interface to describe the NN topology scoring statistics
 */
class ICNNNetworkStats : public details::IRelease {
public:
    virtual void setNodesStats(const NetworkStatsMap& stats) = 0;
    virtual const NetworkStatsMap& getNodesStats() const = 0;

    virtual bool isEmpty() const = 0;
};


class NetworkNodeStats {
public:
    NetworkNodeStats() { }
    explicit NetworkNodeStats(int statCount) {
        float mn = (std::numeric_limits<float>::max)();
        float mx = (std::numeric_limits<float>::min)();

        for (int i = 0; i < statCount; i++) {
            _minOutputs.push_back(mn);
            _maxOutputs.push_back(mx);
        }
    }

public:
    std::vector<float> _minOutputs;
    std::vector<float> _maxOutputs;
};


}  // namespace InferenceEngine
