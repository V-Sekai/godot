// Implementation of PlannerHLClock

#include "planner_hl_clock.h"

void PlannerHLClock::update(int64_t p_physical_time) {
    if (p_physical_time > logical_time) {
        logical_time = p_physical_time;
        counter = 0;
    } else {
        counter++;
    }
    emit_signal("hlc_updated", logical_time, counter);
}