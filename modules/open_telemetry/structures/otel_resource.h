/**************************************************************************/
/*  otel_resource.h                                                      */
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

#ifndef OTEL_RESOURCE_H
#define OTEL_RESOURCE_H

#include "core/io/resource.h"
#include "core/variant/dictionary.h"

// OTelResource represents resource attributes (service.name, etc.)
// as defined in OTLP Resource specification
class OTelResource : public Resource {
	GDCLASS(OTelResource, Resource);
	friend class OTelDocument;
	friend class OTelState;

private:
	// Resource attributes following OTLP semantic conventions
	Dictionary attributes;

protected:
	static void _bind_methods();

public:
	OTelResource();

	// Attributes management
	Dictionary get_attributes() const;
	void set_attributes(const Dictionary &p_attributes);
	void add_attribute(const String &p_key, const Variant &p_value);
	Variant get_attribute(const String &p_key) const;
	bool has_attribute(const String &p_key) const;

	// Common semantic conventions helpers
	void set_service_name(const String &p_name);
	String get_service_name() const;

	void set_service_version(const String &p_version);
	String get_service_version() const;

	void set_service_instance_id(const String &p_id);
	String get_service_instance_id() const;

	// Serialization to OTLP format
	Dictionary to_otlp_dict() const;
	static Ref<OTelResource> from_otlp_dict(const Dictionary &p_dict);
};

#endif // OTEL_RESOURCE_H
