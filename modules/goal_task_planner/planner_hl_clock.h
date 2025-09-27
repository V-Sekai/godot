// Hybrid Logical Clock for temporal ordering

#pragma once

#include "core/io/resource.h"
#include "core/object/object.h"

class PlannerHLClock : public Resource {
    GDCLASS(PlannerHLClock, Resource);

private:
    int64_t logical_time;
    int64_t counter;

public:
    PlannerHLClock() : logical_time(0), counter(0) {}  // Exact Elixir %{l: 0, c: 0}
    void update(int64_t p_physical_time);
    int64_t get_logical_time() const { return logical_time; }
    int64_t get_counter() const { return counter; }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("update", "physical_time"), &PlannerHLClock::update);
        ClassDB::bind_method(D_METHOD("get_logical_time"), &PlannerHLClock::get_logical_time);
        ClassDB::bind_method(D_METHOD("get_counter"), &PlannerHLClock::get_counter);
        ADD_SIGNAL(MethodInfo("hlc_updated", PropertyInfo(Variant::INT, "logical_time"), PropertyInfo(Variant::INT, "counter")));
    }
};