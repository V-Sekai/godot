//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#pragma once
#include <libTimekeeper/StopWatch.hh>
#include <nlohmann/json.hpp>


namespace nlohmann {

template<> struct adl_serializer<Timekeeper::Duration> {
    static void to_json(json& j, Timekeeper::Duration const &dur)
    {
        auto to_json_microseconds = [] (auto &dur) {
            return std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        };
        j = nlohmann::json{
        {"system_us", to_json_microseconds(dur.system)},
        {"user_us", to_json_microseconds(dur.user)},
        {"wall_us", to_json_microseconds(dur.wallclock)}};
    }
    static void from_json(const json&, Timekeeper::Duration&) {
        throw std::runtime_error("json deserialization for Timekeeper::Duration not implemented.");
    }

};

template<> struct adl_serializer<Timekeeper::HierarchicalStopWatchResult> {
    static void to_json(json& j, Timekeeper::HierarchicalStopWatchResult const &res)
    {
        j["count"] = res.count;
        j["duration"] = res.duration;
        if (!res.children.empty()) {
            auto &children = j["children"]; // = nlohmann::json();
            for (const auto &ch: res.children)
            {
                children[ch.name] = ch;
            }
        }
    }
    static void from_json(const json&, Timekeeper::HierarchicalStopWatchResult&) {
        throw std::runtime_error("json deserialization for Timekeeper::HierarchicalStopWatchResult not implemented.");
    }
};

} // namespace nlohmann
