// Simple time tracking for planning

#pragma once

#include "core/typedefs.h"

struct PlannerHLClock {
    int64_t start_time;
    int64_t end_time;
    int64_t duration;

    PlannerHLClock() : start_time(0), end_time(0), duration(0) {}
    
    void set_start_time(int64_t p_time) { start_time = p_time; }
    int64_t get_start_time() const { return start_time; }
    
    void set_end_time(int64_t p_time) { end_time = p_time; }
    int64_t get_end_time() const { return end_time; }
    
    void set_duration(int64_t p_duration) { duration = p_duration; }
    int64_t get_duration() const { return duration; }
};