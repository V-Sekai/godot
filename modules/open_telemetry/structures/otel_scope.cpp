/**************************************************************************/
/*  otel_scope.cpp                                                       */
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

#include "otel_scope.h"

void OTelScope::_bind_methods() {
	// Scope identification (name uses Resource::get_name/set_name)
	ClassDB::bind_method(D_METHOD("get_version"), &OTelScope::get_version);
	ClassDB::bind_method(D_METHOD("set_version", "version"), &OTelScope::set_version);

	// Scope attributes
	ClassDB::bind_method(D_METHOD("get_attributes"), &OTelScope::get_attributes);
	ClassDB::bind_method(D_METHOD("set_attributes", "attributes"), &OTelScope::set_attributes);
	ClassDB::bind_method(D_METHOD("add_attribute", "key", "value"), &OTelScope::add_attribute);

	// Serialization
	ClassDB::bind_method(D_METHOD("to_otlp_dict"), &OTelScope::to_otlp_dict);

	// Note: "name" property uses Resource's built-in get_name/set_name
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "version"), "set_version", "get_version");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "attributes"), "set_attributes", "get_attributes");
}

OTelScope::OTelScope() {
}

// Scope identification (name uses Resource's built-in methods)

String OTelScope::get_version() const {
	return version;
}

void OTelScope::set_version(const String &p_version) {
	version = p_version;
}

// Scope attributes
Dictionary OTelScope::get_attributes() const {
	return attributes;
}

void OTelScope::set_attributes(const Dictionary &p_attributes) {
	attributes = p_attributes;
}

void OTelScope::add_attribute(const String &p_key, const Variant &p_value) {
	attributes[p_key] = p_value;
}

// Serialization to OTLP format
Dictionary OTelScope::to_otlp_dict() const {
	Dictionary scope_dict;

	scope_dict["name"] = get_name();

	if (!version.is_empty()) {
		scope_dict["version"] = version;
	}

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
					value_dict["stringValue"] = String(value);
					break;
			}

			attr["value"] = value_dict;
			otlp_attributes.push_back(attr);
		}

		scope_dict["attributes"] = otlp_attributes;
	}

	return scope_dict;
}

Ref<OTelScope> OTelScope::from_otlp_dict(const Dictionary &p_dict) {
	Ref<OTelScope> scope;
	scope.instantiate();

	if (p_dict.has("name")) {
		scope->set_name(p_dict["name"]);
	}

	if (p_dict.has("version")) {
		scope->set_version(p_dict["version"]);
	}

	if (p_dict.has("attributes")) {
		Array otlp_attributes = p_dict["attributes"];
		Dictionary attrs;

		for (int i = 0; i < otlp_attributes.size(); i++) {
			Dictionary attr = otlp_attributes[i];
			if (attr.has("key") && attr.has("value")) {
				String key = attr["key"];
				Dictionary value_dict = attr["value"];

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

		scope->set_attributes(attrs);
	}

	return scope;
}
