#pragma once

#include "core/io/resource.h"
#include "core/object/object_id.h" // Include for ObjectID
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

class PlannerState : public Resource {
	GDCLASS(PlannerState, Resource);

	Dictionary data;

protected:
	static void _bind_methods();

public:
	Variant get_object(const String &p_predicate, const String &p_subject) const;
	void set_object(const String &p_predicate, const String &p_subject, Variant p_object);
	Array get_subject_property_list() const;
	bool has_subject_variable(const String &p_variable) const;
	bool has_subject(const String &p_predicate, const String &p_subject) const;

	PlannerState() {}
	~PlannerState() {}
};
