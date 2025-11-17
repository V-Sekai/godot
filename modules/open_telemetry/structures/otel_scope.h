/**************************************************************************/
/*  otel_scope.h                                                          */
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

#include "core/io/resource.h"
#include "core/variant/dictionary.h"

// OpenTelemetryScope represents instrumentation scope (formerly instrumentation library)
// as defined in OTLP InstrumentationScope specification
class OpenTelemetryScope : public Resource {
	GDCLASS(OpenTelemetryScope, Resource);
	friend class OpenTelemetryDocument;
	friend class OpenTelemetryState;

private:
	String version;
	Dictionary attributes;

protected:
	static void _bind_methods();

public:
	OpenTelemetryScope();

	// Scope identification (uses Resource::get_name() and Resource::set_name())

	String get_version() const;
	void set_version(const String &p_version);

	// Scope attributes
	Dictionary get_attributes() const;
	void set_attributes(const Dictionary &p_attributes);
	void add_attribute(const String &p_key, const Variant &p_value);

	// Serialization to OTLP format
	Dictionary to_otlp_dict() const;
	static Ref<OpenTelemetryScope> from_otlp_dict(const Dictionary &p_dict);
};
