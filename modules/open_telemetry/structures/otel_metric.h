/**************************************************************************/
/*  otel_metric.h                                                         */
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

class OTelMetric : public Resource {
	GDCLASS(OTelMetric, Resource);
	friend class OTelDocument;
	friend class OTelState;

public:
	enum MetricType {
		METRIC_TYPE_GAUGE = 0,
		METRIC_TYPE_SUM = 1,
		METRIC_TYPE_HISTOGRAM = 2,
		METRIC_TYPE_EXPONENTIAL_HISTOGRAM = 3,
		METRIC_TYPE_SUMMARY = 4
	};

	enum AggregationTemporality {
		AGGREGATION_TEMPORALITY_UNSPECIFIED = 0,
		AGGREGATION_TEMPORALITY_DELTA = 1,
		AGGREGATION_TEMPORALITY_CUMULATIVE = 2
	};

private:
	String name;
	String description;
	String unit;
	MetricType type = METRIC_TYPE_GAUGE;
	AggregationTemporality temporality = AGGREGATION_TEMPORALITY_CUMULATIVE;

	// Data points
	Array data_points;

protected:
	static void _bind_methods();

public:
	OTelMetric();

	// Metric metadata
	String get_name() const;
	void set_name(const String &p_name);

	String get_description() const;
	void set_description(const String &p_description);

	String get_unit() const;
	void set_unit(const String &p_unit);

	MetricType get_type() const;
	void set_type(MetricType p_type);

	AggregationTemporality get_temporality() const;
	void set_temporality(AggregationTemporality p_temporality);

	// Data points
	Array get_data_points() const;
	void set_data_points(const Array &p_data_points);
	void add_data_point(const Dictionary &p_data_point);

	// Serialization
	Dictionary to_otlp_dict() const;
	static Ref<OTelMetric> from_otlp_dict(const Dictionary &p_dict);
};

VARIANT_ENUM_CAST(OTelMetric::MetricType);
VARIANT_ENUM_CAST(OTelMetric::AggregationTemporality);
