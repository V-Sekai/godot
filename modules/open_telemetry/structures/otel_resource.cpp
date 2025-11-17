/**************************************************************************/
/*  otel_resource.cpp                                                     */
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

#include "otel_resource.h"

void OpenTelemetryResource::_bind_methods() {
	// Attributes management
	ClassDB::bind_method(D_METHOD("get_attributes"), &OpenTelemetryResource::get_attributes);
	ClassDB::bind_method(D_METHOD("set_attributes", "attributes"), &OpenTelemetryResource::set_attributes);
	ClassDB::bind_method(D_METHOD("add_attribute", "key", "value"), &OpenTelemetryResource::add_attribute);
	ClassDB::bind_method(D_METHOD("get_attribute", "key"), &OpenTelemetryResource::get_attribute);
	ClassDB::bind_method(D_METHOD("has_attribute", "key"), &OpenTelemetryResource::has_attribute);

	// Semantic conventions helpers
	ClassDB::bind_method(D_METHOD("set_service_name", "name"), &OpenTelemetryResource::set_service_name);
	ClassDB::bind_method(D_METHOD("get_service_name"), &OpenTelemetryResource::get_service_name);
	ClassDB::bind_method(D_METHOD("set_service_version", "version"), &OpenTelemetryResource::set_service_version);
	ClassDB::bind_method(D_METHOD("get_service_version"), &OpenTelemetryResource::get_service_version);
	ClassDB::bind_method(D_METHOD("set_service_instance_id", "id"), &OpenTelemetryResource::set_service_instance_id);
	ClassDB::bind_method(D_METHOD("get_service_instance_id"), &OpenTelemetryResource::get_service_instance_id);

	// Serialization
	ClassDB::bind_method(D_METHOD("to_otlp_dict"), &OpenTelemetryResource::to_otlp_dict);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "attributes"), "set_attributes", "get_attributes");
}

OpenTelemetryResource::OpenTelemetryResource() {
}

// Attributes management
Dictionary OpenTelemetryResource::get_attributes() const {
	return attributes;
}

void OpenTelemetryResource::set_attributes(const Dictionary &p_attributes) {
	attributes = p_attributes;
}

void OpenTelemetryResource::add_attribute(const String &p_key, const Variant &p_value) {
	attributes[p_key] = p_value;
}

Variant OpenTelemetryResource::get_attribute(const String &p_key) const {
	if (attributes.has(p_key)) {
		return attributes[p_key];
	}
	return Variant();
}

bool OpenTelemetryResource::has_attribute(const String &p_key) const {
	return attributes.has(p_key);
}

// Common semantic conventions helpers
void OpenTelemetryResource::set_service_name(const String &p_name) {
	attributes["service.name"] = p_name;
}

String OpenTelemetryResource::get_service_name() const {
	if (attributes.has("service.name")) {
		return attributes["service.name"];
	}
	return String();
}

void OpenTelemetryResource::set_service_version(const String &p_version) {
	attributes["service.version"] = p_version;
}

String OpenTelemetryResource::get_service_version() const {
	if (attributes.has("service.version")) {
		return attributes["service.version"];
	}
	return String();
}

void OpenTelemetryResource::set_service_instance_id(const String &p_id) {
	attributes["service.instance.id"] = p_id;
}

String OpenTelemetryResource::get_service_instance_id() const {
	if (attributes.has("service.instance.id")) {
		return attributes["service.instance.id"];
	}
	return String();
}

// Serialization to OTLP format
Dictionary OpenTelemetryResource::to_otlp_dict() const {
	Dictionary resource_dict;

	if (attributes.size() > 0) {
		// Convert attributes to OTLP format
		Array otlp_attributes;
		Array keys = attributes.keys();

		for (int i = 0; i < keys.size(); i++) {
			String key = keys[i];
			Variant value = attributes[key];

			Dictionary attr;
			attr["key"] = key;

			Dictionary value_dict;
			// Handle different value types according to OTLP spec
			switch (value.get_type()) {
				case Variant::STRING:
					value_dict["stringValue"] = String(value);
					break;
				case Variant::INT:
					value_dict["intValue"] = (int64_t)value;
					break;
				case Variant::FLOAT:
					value_dict["doubleValue"] = (double)value;
					break;
				case Variant::BOOL:
					value_dict["boolValue"] = (bool)value;
					break;
				default:
					// Convert unknown types to string
					value_dict["stringValue"] = String(value);
					break;
			}

			attr["value"] = value_dict;
			otlp_attributes.push_back(attr);
		}

		resource_dict["attributes"] = otlp_attributes;
	}

	return resource_dict;
}

Ref<OpenTelemetryResource> OpenTelemetryResource::from_otlp_dict(const Dictionary &p_dict) {
	Ref<OpenTelemetryResource> resource;
	resource.instantiate();

	if (p_dict.has("attributes")) {
		Array otlp_attributes = p_dict["attributes"];
		Dictionary attrs;

		for (int i = 0; i < otlp_attributes.size(); i++) {
			Dictionary attr = otlp_attributes[i];
			if (attr.has("key") && attr.has("value")) {
				String key = attr["key"];
				Dictionary value_dict = attr["value"];

				// Extract value based on type
				Variant value;
				if (value_dict.has("stringValue")) {
					value = value_dict["stringValue"];
				} else if (value_dict.has("intValue")) {
					value = value_dict["intValue"];
				} else if (value_dict.has("doubleValue")) {
					value = value_dict["doubleValue"];
				} else if (value_dict.has("boolValue")) {
					value = value_dict["boolValue"];
				}

				attrs[key] = value;
			}
		}

		resource->set_attributes(attrs);
	}

	return resource;
}
