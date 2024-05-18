// simple_temporal_network.cpp

#include "simple_temporal_network.h"

String SimpleTemporalNetwork::_to_string() {
    if (resource_name.is_empty()) {
        return "SimpleTemporalNetwork";
    }
    return resource_name;
}

int SimpleTemporalNetwork::get_node_index(int time_point) {
    if (node_index_cache.has(time_point)) {
        return node_index_cache[time_point];
    }

    for (int i = 0; i < node_intervals.size(); i++) {
        Vector2i interval = node_intervals[i];
        if (interval.x <= time_point && time_point <= interval.y) {
            int index = node_indices[interval];
            node_index_cache[time_point] = index;
            return index;
        }
    }

    print_line("Time point not found in any interval");
    return -1;
}

bool SimpleTemporalNetwork::check_overlap(Ref<TemporalConstraint> new_constraint) {
    for (int i = 0; i < constraints.size(); i++) {
        Ref<TemporalConstraint> constraint = constraints[i];
        if (constraint->resource_name == new_constraint->resource_name) {
            if (constraint->overlaps_with(new_constraint)) {
                return true;
            }
        }
    }
    return false;
}

bool SimpleTemporalNetwork::add_temporal_constraint(Ref<TemporalConstraint> from_constraint, Ref<TemporalConstraint> to_constraint, float min_gap, float max_gap) {
    if (check_overlap(from_constraint) || (to_constraint != nullptr && check_overlap(to_constraint))) {
        return false;
    }

    add_constraints_to_list(from_constraint, to_constraint);

    Ref<TemporalConstraint> from_node = process_constraint(from_constraint);
    if (!from_node) {
        print_line("Failed to process from_constraint");
        return false;
    }

    Ref<TemporalConstraint> to_node = nullptr;
    if (to_constraint != nullptr) {
        to_node = process_constraint(to_constraint);
        if (!to_node) {
            print_line("Failed to process to_constraint");
            return false;
        }
        outgoing_edges[from_node] = outgoing_edges.get(from_node, Array()) + to_node;
    }

    update_constraints_list(from_constraint, from_node);
    if (to_constraint != nullptr) {
        update_constraints_list(to_constraint, to_node);
    }

    return true;
}

void SimpleTemporalNetwork::update_constraints_list(Ref<TemporalConstraint> constraint, Ref<TemporalConstraint> node) {
    int index = constraints.find(constraint);
    if (index != -1) {
        constraints[index] = node;
    } else {
        constraints.append(node);
    }
}

void SimpleTemporalNetwork::add_constraints_to_list(Ref<TemporalConstraint> from_constraint, Ref<TemporalConstraint> to_constraint) {
    if (from_constraint) {
        constraints.append(from_constraint);
    }
    if (to_constraint) {
        constraints.append(to_constraint);
    }
}

Ref<TemporalConstraint> SimpleTemporalNetwork::process_constraint(Ref<TemporalConstraint> constraint) {
    Vector2i interval = constraint->get_time_interval();
    if (!node_indices.has(interval)) {
        node_indices[interval] = num_nodes;
        node_intervals.append(interval);
        num_nodes += 1;
    }

    int index = constraints.find(constraint);
    if (index != -1) {
        constraints[index] = constraint;
    }

    return constraint;
}

Ref<TemporalConstraint> SimpleTemporalNetwork::get_temporal_constraint_by_name(String constraint_name) {
    for (int i = 0; i < constraints.size(); i++) {
        Ref<TemporalConstraint> constraint = constraints[i];
        if (constraint->get_name() == get_name()) {
            return constraint;
        }
    }
    return nullptr;
}

bool SimpleTemporalNetwork::is_consistent() {
    if (constraints.size() == 0) {
        return true;
    }

    String constraints_str;
    for (int i = 0; i < constraints.size(); i++) {
        Ref<TemporalConstraint> c = constraints[i];
        constraints_str += c->to_string() + ", ";
    }

    constraints.sort_custom<TemporalConstraintComparator>();

    for (int i = 0; i < constraints.size(); i++) {
            Ref<TemporalConstraint> constraints_i =  constraints[i];
        for (int j = i + 1; j < constraints.size(); j++) {
            Ref<TemporalConstraint> constraints_j = constraints[j];
            constraints_i->set_name(constraints_j->get_name());
            if (constraints_i->get_time_interval().y > constraints_j->get_time_interval().x && constraints_i->get_time_interval().x < constraints_j->get_time_interval().y) {
                print_line("Overlapping constraints: " + constraints_i->get_name() + " and " + constraints_j->get_name());
                return false;
            }
        }

        Array decompositions = enumerate_decompositions(constraints_i);
        if (decompositions.is_empty()) {
            print_line("No valid decompositions for constraint: " + constraints_i->get_name());
            return false;
        }
    }

    return true;
}

Array SimpleTemporalNetwork::enumerate_decompositions(Ref<TemporalConstraint> vertex) {
    Array leafs;

    if (!Object::cast_to<TemporalConstraint>(vertex)) {
        print_line("Error: vertex must be an instance of TemporalConstraint.");
        return leafs;
    }

    if (is_leaf(vertex)) {
        Array array;
        array.append(vertex);
        leafs.append(array);
    } else {
        if (is_or(vertex)) {
            Array children = get_children(vertex);
            for (int i = 0; i < children.size(); i++) {
                Ref<TemporalConstraint> child = children[i];
                leafs += enumerate_decompositions(child);
            }
        } else {
            Array op;
            Array children = get_children(vertex);
            for (int i = 0; i < children.size(); i++) {
                Ref<TemporalConstraint> child = children[i];
                op += enumerate_decompositions(child);
            }
            leafs = cartesian_product(op);
        }
    }

    return leafs;
}

bool SimpleTemporalNetwork::is_leaf(Ref<TemporalConstraint> vertex) {
    return !outgoing_edges.has(vertex) || outgoing_edges[vertex].is_empty();
}

bool SimpleTemporalNetwork::is_or(Ref<TemporalConstraint> vertex) {
    return outgoing_edges.has(vertex) && outgoing_edges[vertex].size() > 1;
}

Array SimpleTemporalNetwork::get_children(Ref<TemporalConstraint> vertex) {
    Array children;

    if (outgoing_edges.has(vertex)) {
        Array child_vertices = outgoing_edges[vertex];
        for (int i = 0; i < child_vertices.size(); i++) {
            Ref<TemporalConstraint> child_vertex = child_vertices[i];
            children.append(child_vertex);
        }
    }

    return children;
}

Array SimpleTemporalNetwork::cartesian_product(Array arrays) {
    Array result;
    result.append(Array());

    for (int i = 0; i < arrays.size(); i++) {
        Array arr = arrays[i];
        Array temp;
        temp.append(Array());

        for (int j = 0; j < result.size(); j++) {
            Array res = result[j];
            for (int k = 0; k < arr.size(); k++) {
                Variant item = arr[k];
                temp.append(res + Array().append(item));
            }
        }

        result = temp;
    }

    return result;
}

void SimpleTemporalNetwork::update_state(Dictionary state) {
    List<Variant> keys;
    state.get_key_list(&keys);

    for (List<Variant>::Element* E = keys.front(); E; E = E->next()) {
        Variant key = E->get();
        Variant value = state[key];

        if (Object::cast_to<TemporalConstraint>(value)) {
            Ref<TemporalConstraint> constraint = TemporalConstraint::_new();
            constraint->time_interval.x = value["time_interval"]["x"];
            constraint->time_interval.y = value["time_interval"]["y"];
            constraint->duration = value["duration"];
            constraint->temporal_qualifier = value["temporal_qualifier"];
            constraint->resource_name = value["resource_name"];

            add_temporal_constraint(constraint);
        }
    }
}

void SimpleTemporalNetwork::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_node_index", "time_point"), &SimpleTemporalNetwork::get_node_index);
    ClassDB::bind_method(D_METHOD("check_overlap", "new_constraint"), &SimpleTemporalNetwork::check_overlap);
    ClassDB::bind_method(D_METHOD("add_temporal_constraint", "from_constraint", "to_constraint", "min_gap", "max_gap"), &SimpleTemporalNetwork::add_temporal_constraint, DEFVAL(nullptr), DEFVAL(0), DEFVAL(0));
    ClassDB::bind_method(D_METHOD("update_constraints_list", "constraint", "node"), &SimpleTemporalNetwork::update_constraints_list);
    ClassDB::bind_method(D_METHOD("add_constraints_to_list", "from_constraint", "to_constraint"), &SimpleTemporalNetwork::add_constraints_to_list);
    ClassDB::bind_method(D_METHOD("process_constraint", "constraint"), &SimpleTemporalNetwork::process_constraint);
    ClassDB::bind_method(D_METHOD("get_temporal_constraint_by_name", "constraint_name"), &SimpleTemporalNetwork::get_temporal_constraint_by_name);
    ClassDB::bind_method(D_METHOD("is_consistent"), &SimpleTemporalNetwork::is_consistent);
    ClassDB::bind_method(D_METHOD("enumerate_decompositions", "vertex"), &SimpleTemporalNetwork::enumerate_decompositions);
    ClassDB::bind_method(D_METHOD("is_leaf", "vertex"), &SimpleTemporalNetwork::is_leaf);
    ClassDB::bind_method(D_METHOD("is_or", "vertex"), &SimpleTemporalNetwork::is_or);
    ClassDB::bind_method(D_METHOD("get_children", "vertex"), &SimpleTemporalNetwork::get_children);
    ClassDB::bind_method(D_METHOD("cartesian_product", "arrays"), &SimpleTemporalNetwork::cartesian_product);
    ClassDB::bind_method(D_METHOD("update_state", "state"), &SimpleTemporalNetwork::update_state);
}



