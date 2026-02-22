/**************************************************************************/
/*  stn_solver.h                                                          */
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

#pragma once

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/pair.h"
#include "core/typedefs.h"
#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

// STN (Simple Temporal Network) Solver using Floyd-Warshall algorithm
// Handles temporal constraint validation and consistency checking

class PlannerSTNSolver {
public:
	// Constraint: min/max distance between two time points (in microseconds)
	struct Constraint {
		int64_t min_distance;
		int64_t max_distance;

		Constraint() :
				min_distance(0), max_distance(0) {}
		Constraint(int64_t p_min, int64_t p_max) :
				min_distance(p_min), max_distance(p_max) {}
	};

	// Snapshot for backtracking (uses Variant types for serialization)
	struct Snapshot {
		Dictionary time_points_map; // Converted to Dictionary for serialization
		Array time_points_list; // Converted to Array for serialization
		Dictionary constraints_map; // Converted to Dictionary for serialization
		Array distance_matrix; // Converted to Array for serialization
		bool consistent;
		int64_t next_time_point_id;

		// Convert to Dictionary for Variant storage
		Dictionary to_dictionary() const {
			Dictionary dict;
			dict["time_points_map"] = time_points_map;
			dict["time_points_list"] = time_points_list;
			dict["constraints_map"] = constraints_map;
			dict["distance_matrix"] = distance_matrix;
			dict["consistent"] = consistent;
			dict["next_time_point_id"] = next_time_point_id;
			return dict;
		}

		// Convert from Dictionary
		static Snapshot from_dictionary(const Dictionary &p_dict) {
			Snapshot snapshot;
			snapshot.time_points_map = p_dict["time_points_map"];
			snapshot.time_points_list = p_dict["time_points_list"];
			snapshot.constraints_map = p_dict["constraints_map"];
			snapshot.distance_matrix = p_dict["distance_matrix"];
			snapshot.consistent = p_dict["consistent"];
			snapshot.next_time_point_id = p_dict["next_time_point_id"];
			return snapshot;
		}
	};

	// Constants (avoid INFINITY macro conflict by using different name)
	// Made public for testing purposes
	static constexpr int64_t STN_INFINITY = INT64_MAX;
	static constexpr int64_t STN_NEG_INFINITY = INT64_MIN + 1; // Avoid overflow

private:
	// Time points: name -> index mapping (internal HashMap)
	HashMap<String, int64_t> time_points_map_internal; // String -> int64_t
	LocalVector<String> time_points_list_internal; // index -> String name

	// Constraints: {from, to} -> Constraint (internal HashMap)
	HashMap<String, Constraint> constraints_map_internal; // String key "from:to" -> Constraint

	// Floyd-Warshall distance matrix: distance_matrix[i][j] = shortest distance from i to j
	// Uses infinity for unreachable, negative values indicate negative cycles
	LocalVector<LocalVector<int64_t>> distance_matrix_internal; // 2D LocalVector for efficiency

	// Consistency flag
	bool consistent;

	// Next time point ID (for unique indexing)
	int64_t next_time_point_id;

	// Helper methods
	int64_t get_time_point_index(const String &p_name) const;
	void ensure_time_point(const String &p_name);
	void rebuild_distance_matrix();
	void run_floyd_warshall();
	bool check_negative_cycles() const;

	// Constraint intersection (tighten constraints)
	Constraint intersect_constraints(const Constraint &p_a, const Constraint &p_b) const;

public:
	PlannerSTNSolver();
	~PlannerSTNSolver();

	// Time point management
	int64_t add_time_point(const String &p_name);
	bool has_time_point(const String &p_name) const;
	Array get_time_points() const;

	// Constraint management
	bool add_constraint(const String &p_from, const String &p_to, int64_t p_min, int64_t p_max);
	bool add_constraint(const String &p_from, const String &p_to, const Constraint &p_constraint);
	bool remove_constraint(const String &p_from, const String &p_to);
	Constraint get_constraint(const String &p_from, const String &p_to) const;
	bool has_constraint(const String &p_from, const String &p_to) const;

	// Consistency checking
	bool is_consistent() const { return consistent; }
	void check_consistency(); // Re-run Floyd-Warshall and update consistency

	// Distance queries
	int64_t get_distance(const String &p_from, const String &p_to) const;
	int64_t get_earliest_time(const String &p_point) const;
	int64_t get_latest_time(const String &p_point) const;

	// Snapshot for backtracking
	Snapshot create_snapshot() const;
	void restore_snapshot(const Snapshot &p_snapshot);

	// Clear/reset
	void clear();

	// Debug
	String to_string() const;
};
