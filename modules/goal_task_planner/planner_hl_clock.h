// Absolute time tracking for planning (microseconds since Unix epoch)

#pragma once

#include "core/typedefs.h"
#include "core/os/os.h"

struct PlannerHLClock {
    // All times in absolute microseconds since Unix epoch (1970-01-01 00:00:00 UTC)
    int64_t start_time = 0;  // Absolute microseconds
    int64_t end_time = 0;    // Absolute microseconds
    int64_t duration = 0;     // Duration in microseconds

    PlannerHLClock() {}
    
    void set_start_time(int64_t p_time) { start_time = p_time; }
    int64_t get_start_time() const { return start_time; }
    
    void set_end_time(int64_t p_time) { end_time = p_time; }
    int64_t get_end_time() const { return end_time; }
    
    void set_duration(int64_t p_duration) { duration = p_duration; }
    int64_t get_duration() const { return duration; }
    
    // Helper methods for absolute time conversion
    // Convert from Godot's unix time (seconds) to microseconds
    static int64_t unix_time_to_microseconds(double p_unix_time) {
        return static_cast<int64_t>(p_unix_time * 1000000.0);
    }
    
    // Get current absolute time in microseconds
    static int64_t now_microseconds() {
        return unix_time_to_microseconds(OS::get_singleton()->get_unix_time());
    }
    
    // Set start time to current absolute time
    void set_start_now() {
        start_time = now_microseconds();
    }
    
    // Set end time to current absolute time
    void set_end_now() {
        end_time = now_microseconds();
    }
    
    // Calculate duration from start_time to end_time
    void calculate_duration() {
        if (end_time > start_time) {
            duration = end_time - start_time;
        } else {
            duration = 0;
        }
    }
    
    // Calculate end_time from start_time and duration
    void calculate_end_from_duration() {
        end_time = start_time + duration;
    }
};