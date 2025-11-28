/**************************************************************************/
/*  test_game_domains_helpers.h                                           */
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

// Shared helper functions for game domain tests

#pragma once

#include "core/variant/array.h"
#include "core/variant/dictionary.h"

#ifdef TOOLS_ENABLED
namespace TestGameDomainsBacktracking {

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Helper function to attach constraints with entity requirements
static Dictionary attach_entity_constraints(const Array &p_action_array, const String &p_entity_type, const Array &p_capabilities) {
	Dictionary constraints_dict;
	Array entities_array;
	Dictionary entity_req;
	entity_req["type"] = p_entity_type;
	entity_req["capabilities"] = p_capabilities;
	entities_array.push_back(entity_req);
	constraints_dict["requires_entities"] = entities_array;

	Dictionary result;
	result["item"] = p_action_array;
	result["constraints"] = constraints_dict;
	return result;
}

// Helper function to attach temporal constraints
static Dictionary attach_temporal_constraints(const Array &p_action_array, int64_t p_start_time_micros, int64_t p_end_time_micros, int64_t p_duration_micros) {
	Dictionary constraints_dict;
	constraints_dict["duration"] = p_duration_micros;
	constraints_dict["start_time"] = p_start_time_micros;
	constraints_dict["end_time"] = p_end_time_micros;

	Dictionary result;
	result["item"] = p_action_array;
	result["temporal_constraints"] = constraints_dict;
	return result;
}

} // namespace TestGameDomainsBacktracking
#endif // TOOLS_ENABLED
