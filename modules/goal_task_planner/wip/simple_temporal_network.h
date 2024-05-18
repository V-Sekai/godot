// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// simple_temporal_network.h
// SPDX-License-Identifier: MIT

#pragma once

#include "core/io/resource.h"
#include "core/math/vector2i.h"
#include "temporal_constraint.h"

class SimpleTemporalNetwork : public Resource {
    GDCLASS(SimpleTemporalNetwork, Resource);

private:
    Array constraints;
    int num_nodes = 0;
    Array node_intervals;
    Dictionary node_indices;

    Dictionary node_index_cache;

public:

    Array getConstraints() const { return constraints; }
    int getNumNodes() const { return num_nodes; }
    Array getNodeIntervals() const { return node_intervals; }
    Dictionary getNodeIndices() const { return node_indices; }
    Dictionary getNodeIndexCache() const { return node_index_cache; }

    void setConstraints(const Array& value) { constraints = value; }
    void setNumNodes(int value) { num_nodes = value; }
    void setNodeIntervals(const Array& value) { node_intervals = value; }
    void setNodeIndices(const Dictionary& value) { node_indices = value; }
    void setNodeIndexCache(const Dictionary& value) { node_index_cache = value; }

    String _to_string();
    int get_node_index(int time_point);
    bool check_overlap(Ref<TemporalConstraint> new_constraint);
    bool add_temporal_constraint(Ref<TemporalConstraint> from_constraint, Ref<TemporalConstraint> to_constraint = nullptr, float min_gap = 0, float max_gap = 0);
    void update_constraints_list(Ref<TemporalConstraint> constraint, Ref<TemporalConstraint> node);
    void add_constraints_to_list(Ref<TemporalConstraint> from_constraint, Ref<TemporalConstraint> to_constraint);
    Ref<TemporalConstraint> process_constraint(Ref<TemporalConstraint> constraint);
    Ref<TemporalConstraint> get_temporal_constraint_by_name(String constraint_name);
    bool is_consistent();
    Array enumerate_decompositions(Ref<TemporalConstraint> vertex);
    bool is_leaf(Ref<TemporalConstraint> vertex);
    bool is_or(Ref<TemporalConstraint> vertex);
    Array get_children(Ref<TemporalConstraint> vertex);
    Array cartesian_product(Array arrays);
    void update_state(Dictionary state);

protected:
    static void _bind_methods();
};
