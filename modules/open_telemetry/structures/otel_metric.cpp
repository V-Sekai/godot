/**************************************************************************/
/*  otel_metric.cpp                                                      */
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

#include "otel_metric.h"

void OTelMetric::_bind_methods() {
	// Metadata
	ClassDB::bind_method(D_METHOD("get_name"), &OTelMetric::get_name);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &OTelMetric::set_name);
	ClassDB::bind_method(D_METHOD("get_description"), &OTelMetric::get_description);
	ClassDB::bind_method(D_METHOD("set_description", "description"), &OTelMetric::set_description);
	ClassDB::bind_method(D_METHOD("get_unit"), &OTelMetric::get_unit);
	ClassDB::bind_method(D_METHOD("set_unit", "unit"), &OTelMetric::set_unit);
	ClassDB::bind_method(D_METHOD("get_type"), &OTelMetric::get_type);
	ClassDB::bind_method(D_METHOD("set_type", "type"), &OTelMetric::set_type);
	ClassDB::bind_method(D_METHOD("get_temporality"), &OTelMetric::get_temporality);
	ClassDB::bind_method(D_METHOD("set_temporality", "temporality"), &OTelMetric::set_temporality);

	// Data points
	ClassDB::bind_method(D_METHOD("get_data_points"), &OTelMetric::get_data_points);
	ClassDB::bind_method(D_METHOD("set_data_points", "data_points"), &OTelMetric::set_data_points);
	ClassDB::bind_method(D_METHOD("add_data_point", "data_point"), &OTelMetric::add_data_point);

	// Serialization
	ClassDB::bind_method(D_METHOD("to_otlp_dict"), &OTelMetric::to_otlp_dict);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_name", "get_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "description"), "set_description", "get_description");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "unit"), "set_unit", "get_unit");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "temporality"), "set_temporality", "get_temporality");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "data_points"), "set_data_points", "get_data_points");

	BIND_ENUM_CONSTANT(METRIC_TYPE_GAUGE);
	BIND_ENUM_CONSTANT(METRIC_TYPE_SUM);
	BIND_ENUM_CONSTANT(METRIC_TYPE_HISTOGRAM);
	BIND_ENUM_CONSTANT(METRIC_TYPE_EXPONENTIAL_HISTOGRAM);
	BIND_ENUM_CONSTANT(METRIC_TYPE_SUMMARY);

	BIND_ENUM_CONSTANT(AGGREGATION_TEMPORALITY_UNSPECIFIED);
	BIND_ENUM_CONSTANT(AGGREGATION_TEMPORALITY_DELTA);
	BIND_ENUM_CONSTANT(AGGREGATION_TEMPORALITY_CUMULATIVE);
}

OTelMetric::OTelMetric() {
}

// Metadata
String OTelMetric::get_name() const {
	return name;
}

void OTelMetric::set_name(const String &p_name) {
	name = p_name;
}

String OTelMetric::get_description() const {
	return description;
}

void OTelMetric::set_description(const String &p_description) {
	description = p_description;
}

String OTelMetric::get_unit() const {
	return unit;
}

void OTelMetric::set_unit(const String &p_unit) {
	unit = p_unit;
}

OTelMetric::MetricType OTelMetric::get_type() const {
	return type;
}

void OTelMetric::set_type(MetricType p_type) {
	type = p_type;
}

OTelMetric::AggregationTemporality OTelMetric::get_temporality() const {
	return temporality;
}

void OTelMetric::set_temporality(AggregationTemporality p_temporality) {
	temporality = p_temporality;
}

// Data points
Array OTelMetric::get_data_points() const {
	return data_points;
}

void OTelMetric::set_data_points(const Array &p_data_points) {
	data_points = p_data_points;
}

void OTelMetric::add_data_point(const Dictionary &p_data_point) {
	data_points.push_back(p_data_point);
}

// Serialization
Dictionary OTelMetric::to_otlp_dict() const {
	Dictionary metric_dict;
	
	metric_dict["name"] = name;
	
	if (!description.is_empty()) {
		metric_dict["description"] = description;
	}
	
	if (!unit.is_empty()) {
		metric_dict["unit"] = unit;
	}

	// Add type-specific data
	switch (type) {
		case METRIC_TYPE_GAUGE: {
			Dictionary gauge;
			gauge["dataPoints"] = data_points;
			metric_dict["gauge"] = gauge;
			break;
		}
		case METRIC_TYPE_SUM: {
			Dictionary sum;
			sum["dataPoints"] = data_points;
			sum["aggregationTemporality"] = (int)temporality;
			sum["isMonotonic"] = true;
			metric_dict["sum"] = sum;
			break;
		}
		case METRIC_TYPE_HISTOGRAM: {
			Dictionary histogram;
			histogram["dataPoints"] = data_points;
			histogram["aggregationTemporality"] = (int)temporality;
			metric_dict["histogram"] = histogram;
			break;
		}
		default:
			break;
	}

	return metric_dict;
}

Ref<OTelMetric> OTelMetric::from_otlp_dict(const Dictionary &p_dict) {
	Ref<OTelMetric> metric;
	metric.instantiate();

	if (p_dict.has("name")) {
		metric->set_name(p_dict["name"]);
	}
	
	if (p_dict.has("description")) {
		metric->set_description(p_dict["description"]);
	}
	
	if (p_dict.has("unit")) {
		metric->set_unit(p_dict["unit"]);
	}

	// Determine type and extract data points
	if (p_dict.has("gauge")) {
		metric->set_type(METRIC_TYPE_GAUGE);
		Dictionary gauge = p_dict["gauge"];
		if (gauge.has("dataPoints")) {
			metric->set_data_points(gauge["dataPoints"]);
		}
	} else if (p_dict.has("sum")) {
		metric->set_type(METRIC_TYPE_SUM);
		Dictionary sum = p_dict["sum"];
		if (sum.has("dataPoints")) {
			metric->set_data_points(sum["dataPoints"]);
		}
		if (sum.has("aggregationTemporality")) {
			metric->set_temporality((AggregationTemporality)(int)sum["aggregationTemporality"]);
		}
	} else if (p_dict.has("histogram")) {
		metric->set_type(METRIC_TYPE_HISTOGRAM);
		Dictionary histogram = p_dict["histogram"];
		if (histogram.has("dataPoints")) {
			metric->set_data_points(histogram["dataPoints"]);
		}
		if (histogram.has("aggregationTemporality")) {
			metric->set_temporality((AggregationTemporality)(int)histogram["aggregationTemporality"]);
		}
	}

	return metric;
}
