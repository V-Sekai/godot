/**************************************************************************/
/*  planner_metadata.h                                                    */
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

#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "entity_requirement.h"

// Planner metadata for temporal and entity requirements
// Used with commands, multigoals, unigoals, actions, and tasks
class PlannerMetadata {
public:
	String duration; // ISO 8601 duration string (e.g., "PT2H", "PT30M")
	LocalVector<PlannerEntityRequirement> requires_entities; // Entity requirements
	String start_time; // Optional ISO 8601 datetime
	String end_time; // Optional ISO 8601 datetime
	
	PlannerMetadata() {}
	
	PlannerMetadata(const String &p_duration, const LocalVector<PlannerEntityRequirement> &p_requires_entities) :
		duration(p_duration), requires_entities(p_requires_entities) {}
	
	// Validation
	bool is_valid() const {
		return !duration.is_empty() && requires_entities.size() > 0;
	}
	
	// Convert to Dictionary for GDScript interface
	Dictionary to_dictionary() const {
		Dictionary dict;
		dict["duration"] = duration;
		
		Array entities_array;
		for (uint32_t i = 0; i < requires_entities.size(); i++) {
			entities_array.push_back(requires_entities[i].to_dictionary());
		}
		dict["requires_entities"] = entities_array;
		
		if (!start_time.is_empty()) {
			dict["start_time"] = start_time;
		}
		if (!end_time.is_empty()) {
			dict["end_time"] = end_time;
		}
		
		return dict;
	}
	
	// Convert from Dictionary (GDScript interface)
	static PlannerMetadata from_dictionary(const Dictionary &p_dict) {
		PlannerMetadata metadata;
		metadata.duration = p_dict.get("duration", "");
		
		Variant entities_var = p_dict.get("requires_entities", Array());
		if (entities_var.get_type() == Variant::ARRAY) {
			Array entities_array = entities_var;
			metadata.requires_entities.resize(entities_array.size());
			for (int i = 0; i < entities_array.size(); i++) {
				Dictionary entity_dict = entities_array[i];
				metadata.requires_entities[i] = PlannerEntityRequirement::from_dictionary(entity_dict);
			}
		}
		
		metadata.start_time = p_dict.get("start_time", "");
		metadata.end_time = p_dict.get("end_time", "");
		
		return metadata;
	}
};

// Unigoal metadata extends PlannerMetadata with predicate field
class PlannerUnigoalMetadata : public PlannerMetadata {
public:
	String predicate; // Which predicate this unigoal method handles
	
	PlannerUnigoalMetadata() {}
	
	PlannerUnigoalMetadata(const String &p_predicate, const String &p_duration, const LocalVector<PlannerEntityRequirement> &p_requires_entities) :
		PlannerMetadata(p_duration, p_requires_entities), predicate(p_predicate) {}
	
	// Convert to Dictionary
	Dictionary to_dictionary() const {
		Dictionary dict = PlannerMetadata::to_dictionary();
		dict["predicate"] = predicate;
		return dict;
	}
	
	// Convert from Dictionary
	static PlannerUnigoalMetadata from_dictionary(const Dictionary &p_dict) {
		PlannerUnigoalMetadata metadata;
		PlannerMetadata base = PlannerMetadata::from_dictionary(p_dict);
		metadata.duration = base.duration;
		metadata.requires_entities = base.requires_entities;
		metadata.start_time = base.start_time;
		metadata.end_time = base.end_time;
		metadata.predicate = p_dict.get("predicate", "");
		return metadata;
	}
};

