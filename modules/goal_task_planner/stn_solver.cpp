/**************************************************************************/
/*  stn_solver.cpp                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "stn_solver.h"
#include "core/string/print_string.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"

constexpr int64_t PlannerSTNSolver::STN_INFINITY;
constexpr int64_t PlannerSTNSolver::STN_NEG_INFINITY;

PlannerSTNSolver::PlannerSTNSolver() {
	consistent = true;
	next_time_point_id = 0;
	// Internal structures are initialized automatically
}

PlannerSTNSolver::~PlannerSTNSolver() {
}

int64_t PlannerSTNSolver::get_time_point_index(const String &p_name) const {
	const int64_t *idx = time_points_map_internal.getptr(p_name);
	if (idx == nullptr) {
		return -1;
	}
	return *idx;
}

void PlannerSTNSolver::ensure_time_point(const String &p_name) {
	if (!time_points_map_internal.has(p_name)) {
		int64_t index = next_time_point_id++;
		time_points_map_internal[p_name] = index;
		time_points_list_internal.push_back(p_name);

		// Expand distance matrix to accommodate new point
		// Initialize with STN_INFINITY for all distances except self (0)
		uint32_t current_size = time_points_list_internal.size();
		distance_matrix_internal.resize(current_size);

		for (uint32_t i = 0; i < current_size; i++) {
			if (distance_matrix_internal[i].size() < current_size) {
				distance_matrix_internal[i].resize(current_size);
			}

			for (uint32_t j = 0; j < current_size; j++) {
				if (i == j) {
					distance_matrix_internal[i][j] = 0; // Distance to self is 0
				} else if (j >= distance_matrix_internal[i].size() - 1) {
					distance_matrix_internal[i][j] = STN_INFINITY; // New point, initialize to infinity
				}
			}
		}
	}
}

PlannerSTNSolver::Constraint PlannerSTNSolver::intersect_constraints(const Constraint &p_a, const Constraint &p_b) const {
	// Intersection: take the tighter constraint (max of mins, min of maxes)
	int64_t new_min = (p_a.min_distance > p_b.min_distance) ? p_a.min_distance : p_b.min_distance;
	int64_t new_max = (p_a.max_distance < p_b.max_distance) ? p_a.max_distance : p_b.max_distance;

	// If min > max, constraints are incompatible (empty intersection)
	if (new_min > new_max) {
		return Constraint(STN_INFINITY, STN_NEG_INFINITY); // Empty/invalid constraint
	}

	return Constraint(new_min, new_max);
}

void PlannerSTNSolver::rebuild_distance_matrix() {
	uint32_t n = time_points_list_internal.size();
	distance_matrix_internal.clear();
	distance_matrix_internal.resize(n);

	// Initialize distance matrix
	for (uint32_t i = 0; i < n; i++) {
		LocalVector<int64_t> row;
		row.resize(n);
		for (uint32_t j = 0; j < n; j++) {
			if (i == j) {
				row[j] = 0;
			} else {
				row[j] = STN_INFINITY;
			}
		}
		distance_matrix_internal[i] = row;
	}

	// Add constraints to distance matrix
	for (const KeyValue<String, Constraint> &E : constraints_map_internal) {
		String key = E.key;
		Constraint constraint = E.value;

		// Check for invalid constraints (empty intersection) - these indicate inconsistency
		if (constraint.min_distance > constraint.max_distance) {
			// Invalid constraint found - mark as inconsistent
			consistent = false;
			continue;
		}

		// Parse key: "from:to"
		int colon_pos = key.find(":");
		if (colon_pos < 0) {
			continue;
		}
		String from = key.substr(0, colon_pos);
		String to = key.substr(colon_pos + 1);

		int64_t from_idx = get_time_point_index(from);
		int64_t to_idx = get_time_point_index(to);

		if (from_idx < 0 || to_idx < 0 || from_idx >= (int64_t)n || to_idx >= (int64_t)n) {
			continue;
		}

		// Set distance to max (temporal constraint: to - from <= max)
		// In STN, constraint [min, max] means: min <= to - from <= max
		// This translates to: distance[from][to] <= max
		// Also need to handle min: to - from >= min means from - to <= -min
		// So: distance[to][from] <= -min
		int64_t current_dist = distance_matrix_internal[from_idx][to_idx];
		if (current_dist == STN_INFINITY || constraint.max_distance < current_dist) {
			distance_matrix_internal[from_idx][to_idx] = constraint.max_distance;
		}

		// Handle min_distance constraint: set reverse edge
		// min <= to - from means from - to <= -min
		int64_t reverse_dist = distance_matrix_internal[to_idx][from_idx];
		int64_t min_reverse = -constraint.min_distance;
		if (reverse_dist == STN_INFINITY || min_reverse < reverse_dist) {
			distance_matrix_internal[to_idx][from_idx] = min_reverse;
		}
	}
}

void PlannerSTNSolver::run_floyd_warshall() {
	uint32_t n = time_points_list_internal.size();
	if (n == 0) {
		consistent = true;
		return;
	}

	// Check for invalid constraints first (empty intersections indicate inconsistency)
	for (const KeyValue<String, Constraint> &E : constraints_map_internal) {
		if (E.value.min_distance > E.value.max_distance) {
			consistent = false;
			return;
		}
	}

	// Ensure distance matrix is built
	if (distance_matrix_internal.size() != n) {
		rebuild_distance_matrix();
		// Re-check for invalid constraints after rebuild
		for (const KeyValue<String, Constraint> &E : constraints_map_internal) {
			if (E.value.min_distance > E.value.max_distance) {
				consistent = false;
				return;
			}
		}
	}

	// Floyd-Warshall algorithm: all-pairs shortest paths
	for (uint32_t k = 0; k < n; k++) {
		for (uint32_t i = 0; i < n; i++) {
			int64_t dist_ik = distance_matrix_internal[i][k];

			if (dist_ik == STN_INFINITY) {
				continue; // Can't reach k from i
			}

			for (uint32_t j = 0; j < n; j++) {
				int64_t dist_kj = distance_matrix_internal[k][j];
				if (dist_kj == STN_INFINITY) {
					continue; // Can't reach j from k
				}

				int64_t dist_ij = distance_matrix_internal[i][j];
				int64_t new_dist = dist_ik + dist_kj;

				// Check for overflow
				if (dist_ik > 0 && dist_kj > 0 && new_dist < dist_ik) {
					new_dist = STN_INFINITY; // Overflow, treat as infinity
				} else if (dist_ik < 0 && dist_kj < 0 && new_dist > dist_ik) {
					new_dist = STN_NEG_INFINITY; // Underflow
				}

				if (new_dist < dist_ij) {
					distance_matrix_internal[i][j] = new_dist;
				}
			}
		}
	}

	// Check for negative cycles (inconsistency)
	consistent = !check_negative_cycles();
}

bool PlannerSTNSolver::check_negative_cycles() const {
	uint32_t n = time_points_list_internal.size();

	// Negative cycle exists if distance[i][i] < 0 for any i
	for (uint32_t i = 0; i < n; i++) {
		int64_t self_dist = distance_matrix_internal[i][i];
		if (self_dist < 0) {
			return true; // Negative cycle detected
		}
	}

	return false; // No negative cycles
}

int64_t PlannerSTNSolver::add_time_point(const String &p_name) {
	// Validate time point name is not empty
	if (p_name.is_empty()) {
		return -1; // Invalid time point name
	}
	ensure_time_point(p_name);
	return get_time_point_index(p_name);
}

bool PlannerSTNSolver::has_time_point(const String &p_name) const {
	return time_points_map_internal.has(p_name);
}

Array PlannerSTNSolver::get_time_points() const {
	// Convert LocalVector to Array for GDScript interface
	Array result;
	result.resize(time_points_list_internal.size());
	for (uint32_t i = 0; i < time_points_list_internal.size(); i++) {
		result[i] = time_points_list_internal[i];
	}
	return result;
}

bool PlannerSTNSolver::add_constraint(const String &p_from, const String &p_to, int64_t p_min, int64_t p_max) {
	return add_constraint(p_from, p_to, Constraint(p_min, p_max));
}

bool PlannerSTNSolver::add_constraint(const String &p_from, const String &p_to, const Constraint &p_constraint) {
	// Validate time point names are not empty
	if (p_from.is_empty() || p_to.is_empty()) {
		consistent = false;
		return false; // Invalid time point names
	}

	// Ensure time points exist
	ensure_time_point(p_from);
	ensure_time_point(p_to);

	// Check for invalid constraint
	if (p_constraint.min_distance > p_constraint.max_distance) {
		consistent = false;
		return false;
	}

	String forward_key = p_from + ":" + p_to;
	String reverse_key = p_to + ":" + p_from;

	// Get existing constraints if any
	Constraint forward_constraint = p_constraint;
	Constraint reverse_constraint = Constraint(-p_constraint.max_distance, -p_constraint.min_distance);

	const Constraint *existing_forward = constraints_map_internal.getptr(forward_key);
	if (existing_forward) {
		forward_constraint = intersect_constraints(*existing_forward, p_constraint);

		// Check if intersection is empty
		if (forward_constraint.min_distance > forward_constraint.max_distance) {
			// Store invalid constraint to mark inconsistency
			constraints_map_internal[forward_key] = forward_constraint;
			// Update reverse constraint to reflect the conflict
			reverse_constraint = Constraint(-forward_constraint.max_distance, -forward_constraint.min_distance);
			constraints_map_internal[reverse_key] = reverse_constraint;
			// Rebuild and check to ensure inconsistency is detected
			rebuild_distance_matrix();
			run_floyd_warshall();
			return false;
		}
	}

	const Constraint *existing_reverse = constraints_map_internal.getptr(reverse_key);
	if (existing_reverse) {
		Constraint new_reverse = Constraint(-p_constraint.max_distance, -p_constraint.min_distance);
		reverse_constraint = intersect_constraints(*existing_reverse, new_reverse);

		// Check if intersection is empty
		if (reverse_constraint.min_distance > reverse_constraint.max_distance) {
			// Store invalid constraint to mark inconsistency
			constraints_map_internal[reverse_key] = reverse_constraint;
			// Update forward constraint to reflect the conflict
			forward_constraint = Constraint(-reverse_constraint.max_distance, -reverse_constraint.min_distance);
			constraints_map_internal[forward_key] = forward_constraint;
			// Rebuild and check to ensure inconsistency is detected
			rebuild_distance_matrix();
			run_floyd_warshall();
			return false;
		}
	}

	// Store constraints in internal HashMap
	constraints_map_internal[forward_key] = forward_constraint;
	constraints_map_internal[reverse_key] = reverse_constraint;

	// Rebuild distance matrix and run Floyd-Warshall
	rebuild_distance_matrix();
	run_floyd_warshall();

	return consistent;
}

bool PlannerSTNSolver::remove_constraint(const String &p_from, const String &p_to) {
	String forward_key = p_from + ":" + p_to;
	String reverse_key = p_to + ":" + p_from;

	bool removed = false;
	if (constraints_map_internal.has(forward_key)) {
		constraints_map_internal.erase(forward_key);
		removed = true;
	}
	if (constraints_map_internal.has(reverse_key)) {
		constraints_map_internal.erase(reverse_key);
		removed = true;
	}

	if (removed) {
		rebuild_distance_matrix();
		run_floyd_warshall();
	}

	return removed;
}

PlannerSTNSolver::Constraint PlannerSTNSolver::get_constraint(const String &p_from, const String &p_to) const {
	String key = p_from + ":" + p_to;
	const Constraint *constraint = constraints_map_internal.getptr(key);
	if (constraint == nullptr) {
		return Constraint(STN_INFINITY, STN_INFINITY); // No constraint = unbounded
	}
	return *constraint;
}

bool PlannerSTNSolver::has_constraint(const String &p_from, const String &p_to) const {
	String key = p_from + ":" + p_to;
	return constraints_map_internal.has(key);
}

void PlannerSTNSolver::check_consistency() {
	rebuild_distance_matrix();
	run_floyd_warshall();
}

int64_t PlannerSTNSolver::get_distance(const String &p_from, const String &p_to) const {
	// Validate time point names are not empty
	if (p_from.is_empty() || p_to.is_empty()) {
		return STN_INFINITY; // Invalid time point names
	}

	int64_t from_idx = get_time_point_index(p_from);
	int64_t to_idx = get_time_point_index(p_to);

	if (from_idx < 0 || to_idx < 0 || from_idx >= (int64_t)distance_matrix_internal.size()) {
		return STN_INFINITY;
	}

	if (to_idx >= (int64_t)distance_matrix_internal[from_idx].size()) {
		return STN_INFINITY;
	}

	return distance_matrix_internal[from_idx][to_idx];
}

int64_t PlannerSTNSolver::get_earliest_time(const String &p_point) const {
	// Validate time point name is not empty
	if (p_point.is_empty()) {
		return STN_INFINITY; // Invalid time point name
	}

	// Distance from origin (time point 0) to this point
	// Assuming origin is at index 0, or we need to track it
	if (time_points_list_internal.size() == 0) {
		return 0;
	}

	// Validate time point exists
	if (!has_time_point(p_point)) {
		return STN_INFINITY; // Time point not found
	}

	// For now, return distance from first time point (could be origin)
	String origin = time_points_list_internal[0];
	return get_distance(origin, p_point);
}

int64_t PlannerSTNSolver::get_latest_time(const String &p_point) const {
	// Validate time point name is not empty
	if (p_point.is_empty()) {
		return STN_INFINITY; // Invalid time point name
	}

	// Latest time is negative of distance from point to origin
	if (time_points_list_internal.size() == 0) {
		return 0;
	}

	// Validate time point exists
	if (!has_time_point(p_point)) {
		return STN_INFINITY; // Time point not found
	}

	String origin = time_points_list_internal[0];
	int64_t dist_to_origin = get_distance(p_point, origin);
	if (dist_to_origin == STN_INFINITY) {
		return STN_INFINITY;
	}

	return -dist_to_origin;
}

PlannerSTNSolver::Snapshot PlannerSTNSolver::create_snapshot() const {
	Snapshot snapshot;

	// Convert internal HashMap to Dictionary for serialization
	Dictionary time_points_dict;
	for (const KeyValue<String, int64_t> &E : time_points_map_internal) {
		time_points_dict[E.key] = E.value;
	}
	snapshot.time_points_map = time_points_dict;

	// Convert internal LocalVector to Array for serialization
	Array time_points_array;
	time_points_array.resize(time_points_list_internal.size());
	for (uint32_t i = 0; i < time_points_list_internal.size(); i++) {
		time_points_array[i] = time_points_list_internal[i];
	}
	snapshot.time_points_list = time_points_array;

	// Convert internal constraints HashMap to Dictionary for serialization
	Dictionary constraints_dict;
	for (const KeyValue<String, Constraint> &E : constraints_map_internal) {
		Dictionary constraint_dict;
		constraint_dict["min_distance"] = E.value.min_distance;
		constraint_dict["max_distance"] = E.value.max_distance;
		constraints_dict[E.key] = constraint_dict;
	}
	snapshot.constraints_map = constraints_dict;

	// Convert internal distance matrix to Array for serialization
	Array matrix_copy;
	matrix_copy.resize(distance_matrix_internal.size());
	for (uint32_t i = 0; i < distance_matrix_internal.size(); i++) {
		Array row;
		row.resize(distance_matrix_internal[i].size());
		for (uint32_t j = 0; j < distance_matrix_internal[i].size(); j++) {
			row[j] = distance_matrix_internal[i][j];
		}
		matrix_copy[i] = row;
	}
	snapshot.distance_matrix = matrix_copy;

	snapshot.consistent = consistent;
	snapshot.next_time_point_id = next_time_point_id;

	return snapshot;
}

void PlannerSTNSolver::restore_snapshot(const Snapshot &p_snapshot) {
	// Convert Dictionary to internal HashMap
	time_points_map_internal.clear();
	Array time_points_keys = p_snapshot.time_points_map.keys();
	for (int i = 0; i < time_points_keys.size(); i++) {
		String key = time_points_keys[i];
		int64_t value = p_snapshot.time_points_map[key];
		time_points_map_internal[key] = value;
	}

	// Convert Array to internal LocalVector
	time_points_list_internal.clear();
	time_points_list_internal.resize(p_snapshot.time_points_list.size());
	for (int i = 0; i < p_snapshot.time_points_list.size(); i++) {
		time_points_list_internal[i] = p_snapshot.time_points_list[i];
	}

	// Convert Dictionary to internal constraints HashMap
	constraints_map_internal.clear();
	Array constraint_keys = p_snapshot.constraints_map.keys();
	for (int i = 0; i < constraint_keys.size(); i++) {
		String key = constraint_keys[i];
		Dictionary constraint_dict = p_snapshot.constraints_map[key];
		Constraint constraint(constraint_dict["min_distance"], constraint_dict["max_distance"]);
		constraints_map_internal[key] = constraint;
	}

	// Convert Array to internal distance matrix
	distance_matrix_internal.clear();
	distance_matrix_internal.resize(p_snapshot.distance_matrix.size());
	for (int i = 0; i < p_snapshot.distance_matrix.size(); i++) {
		Array row = p_snapshot.distance_matrix[i];
		LocalVector<int64_t> row_vec;
		row_vec.resize(row.size());
		for (int j = 0; j < row.size(); j++) {
			row_vec[j] = row[j];
		}
		distance_matrix_internal[i] = row_vec;
	}

	consistent = p_snapshot.consistent;
	next_time_point_id = p_snapshot.next_time_point_id;
}

void PlannerSTNSolver::clear() {
	time_points_map_internal.clear();
	time_points_list_internal.clear();
	constraints_map_internal.clear();
	distance_matrix_internal.clear();
	consistent = true;
	next_time_point_id = 0;
}

String PlannerSTNSolver::to_string() const {
	String result = "STN Solver:\n";
	result += "  Time Points: " + itos((int)time_points_list_internal.size()) + "\n";
	result += "  Constraints: " + itos((int)constraints_map_internal.size()) + "\n";
	result += "  Consistent: " + String(consistent ? "true" : "false") + "\n";
	return result;
}
