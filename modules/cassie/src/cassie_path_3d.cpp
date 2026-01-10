/**************************************************************************/
/*  cassie_path_3d.cpp                                                    */
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

#include "cassie_path_3d.h"

void CassiePath3D::_bind_methods() {
	// Point management
	ClassDB::bind_method(D_METHOD("add_point", "position", "normal"), &CassiePath3D::add_point, DEFVAL(Vector3(0, 1, 0)));
	ClassDB::bind_method(D_METHOD("insert_point", "index", "position", "normal"), &CassiePath3D::insert_point, DEFVAL(Vector3(0, 1, 0)));
	ClassDB::bind_method(D_METHOD("remove_point", "index"), &CassiePath3D::remove_point);
	ClassDB::bind_method(D_METHOD("set_point_position", "index", "position"), &CassiePath3D::set_point_position);
	ClassDB::bind_method(D_METHOD("set_point_normal", "index", "normal"), &CassiePath3D::set_point_normal);
	ClassDB::bind_method(D_METHOD("get_point_position", "index"), &CassiePath3D::get_point_position);
	ClassDB::bind_method(D_METHOD("get_point_normal", "index"), &CassiePath3D::get_point_normal);
	ClassDB::bind_method(D_METHOD("get_point_count"), &CassiePath3D::get_point_count);
	ClassDB::bind_method(D_METHOD("clear_points"), &CassiePath3D::clear_points);

	// Path properties
	ClassDB::bind_method(D_METHOD("set_closed", "closed"), &CassiePath3D::set_closed);
	ClassDB::bind_method(D_METHOD("is_path_closed"), &CassiePath3D::is_path_closed);

	// Curve processing
	ClassDB::bind_method(D_METHOD("beautify_laplacian", "lambda", "iterations"), &CassiePath3D::beautify_laplacian, DEFVAL(0.5f), DEFVAL(5));
	ClassDB::bind_method(D_METHOD("beautify_taubin", "lambda", "mu", "iterations"), &CassiePath3D::beautify_taubin, DEFVAL(0.5f), DEFVAL(-0.53f), DEFVAL(5));
	ClassDB::bind_method(D_METHOD("resample_uniform", "target_count"), &CassiePath3D::resample_uniform);
	ClassDB::bind_method(D_METHOD("smooth_normals"), &CassiePath3D::smooth_normals);

	// Sampling
	ClassDB::bind_method(D_METHOD("get_sample_points", "count"), &CassiePath3D::get_sample_points);
	ClassDB::bind_method(D_METHOD("get_sample_normals", "count"), &CassiePath3D::get_sample_normals);
	ClassDB::bind_method(D_METHOD("get_points"), &CassiePath3D::get_points);
	ClassDB::bind_method(D_METHOD("get_normals"), &CassiePath3D::get_normals);

	// Analysis
	ClassDB::bind_method(D_METHOD("get_total_length"), &CassiePath3D::get_total_length);
	ClassDB::bind_method(D_METHOD("get_average_segment_length"), &CassiePath3D::get_average_segment_length);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "closed"), "set_closed", "is_path_closed");
}

void CassiePath3D::add_point(const Vector3 &p_position, const Vector3 &p_normal) {
	points.push_back(p_position);
	normals.push_back(p_normal.normalized());
}

void CassiePath3D::insert_point(int p_index, const Vector3 &p_position, const Vector3 &p_normal) {
	ERR_FAIL_INDEX(p_index, points.size() + 1);
	points.insert(p_index, p_position);
	normals.insert(p_index, p_normal.normalized());
}

void CassiePath3D::remove_point(int p_index) {
	ERR_FAIL_INDEX(p_index, points.size());
	points.remove_at(p_index);
	normals.remove_at(p_index);
}

void CassiePath3D::set_point_position(int p_index, const Vector3 &p_position) {
	ERR_FAIL_INDEX(p_index, points.size());
	points.write[p_index] = p_position;
}

void CassiePath3D::set_point_normal(int p_index, const Vector3 &p_normal) {
	ERR_FAIL_INDEX(p_index, normals.size());
	normals.write[p_index] = p_normal.normalized();
}

Vector3 CassiePath3D::get_point_position(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, points.size(), Vector3());
	return points[p_index];
}

Vector3 CassiePath3D::get_point_normal(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, normals.size(), Vector3());
	return normals[p_index];
}

int CassiePath3D::get_point_count() const {
	return points.size();
}

void CassiePath3D::clear_points() {
	points.clear();
	normals.clear();
}

void CassiePath3D::set_closed(bool p_closed) {
	is_closed = p_closed;
}

bool CassiePath3D::is_path_closed() const {
	return is_closed;
}

void CassiePath3D::beautify_laplacian(float p_lambda, int p_iterations) {
	ERR_FAIL_COND_MSG(points.size() < 3, "At least 3 points required for beautification.");

	for (int iter = 0; iter < p_iterations; iter++) {
		PackedVector3Array new_points = points;

		for (int i = 0; i < points.size(); i++) {
			int prev_i = (i == 0) ? (is_closed ? points.size() - 1 : 0) : (i - 1);
			int next_i = (i == points.size() - 1) ? (is_closed ? 0 : points.size() - 1) : (i + 1);

			// Skip boundary points in open curves
			if (!is_closed && (i == 0 || i == points.size() - 1)) {
				continue;
			}

			Vector3 laplacian = (points[prev_i] + points[next_i]) / 2.0f - points[i];
			new_points.write[i] = points[i] + laplacian * p_lambda;
		}

		points = new_points;
	}
}

void CassiePath3D::beautify_taubin(float p_lambda, float p_mu, int p_iterations) {
	ERR_FAIL_COND_MSG(points.size() < 3, "At least 3 points required for beautification.");

	for (int iter = 0; iter < p_iterations; iter++) {
		// Laplacian smoothing step
		PackedVector3Array new_points = points;

		for (int i = 0; i < points.size(); i++) {
			int prev_i = (i == 0) ? (is_closed ? points.size() - 1 : 0) : (i - 1);
			int next_i = (i == points.size() - 1) ? (is_closed ? 0 : points.size() - 1) : (i + 1);

			if (!is_closed && (i == 0 || i == points.size() - 1)) {
				continue;
			}

			Vector3 laplacian = (points[prev_i] + points[next_i]) / 2.0f - points[i];
			new_points.write[i] = points[i] + laplacian * p_lambda;
		}

		points = new_points;

		// Anti-shrinking step (mu is typically negative)
		new_points = points;

		for (int i = 0; i < points.size(); i++) {
			int prev_i = (i == 0) ? (is_closed ? points.size() - 1 : 0) : (i - 1);
			int next_i = (i == points.size() - 1) ? (is_closed ? 0 : points.size() - 1) : (i + 1);

			if (!is_closed && (i == 0 || i == points.size() - 1)) {
				continue;
			}

			Vector3 laplacian = (points[prev_i] + points[next_i]) / 2.0f - points[i];
			new_points.write[i] = points[i] + laplacian * p_mu;
		}

		points = new_points;
	}
}

void CassiePath3D::resample_uniform(int p_target_count) {
	ERR_FAIL_COND_MSG(points.size() < 2, "At least 2 points required for resampling.");
	ERR_FAIL_COND_MSG(p_target_count < 2, "Target count must be at least 2.");

	float total_length = get_total_length();
	float segment_length = total_length / (p_target_count - (is_closed ? 0 : 1));

	PackedVector3Array new_points;
	PackedVector3Array new_normals;

	new_points.push_back(points[0]);
	new_normals.push_back(normals[0]);

	float accumulated_length = 0.0f;
	float target_length = segment_length;
	int current_segment = 0;

	while (new_points.size() < p_target_count && current_segment < points.size() - 1) {
		int next_segment = (current_segment + 1) % points.size();
		Vector3 segment_start = points[current_segment];
		Vector3 segment_end = points[next_segment];
		Vector3 normal_start = normals[current_segment];
		Vector3 normal_end = normals[next_segment];
		float length = segment_start.distance_to(segment_end);

		if (accumulated_length + length >= target_length) {
			float t = (target_length - accumulated_length) / length;
			new_points.push_back(segment_start.lerp(segment_end, t));
			new_normals.push_back(normal_start.slerp(normal_end, t).normalized());
			target_length += segment_length;
		} else {
			accumulated_length += length;
			current_segment++;
		}
	}

	if (!is_closed && new_points.size() < p_target_count) {
		new_points.push_back(points[points.size() - 1]);
		new_normals.push_back(normals[normals.size() - 1]);
	}

	points = new_points;
	normals = new_normals;
}

void CassiePath3D::smooth_normals() {
	ERR_FAIL_COND_MSG(normals.size() < 3, "At least 3 normals required for smoothing.");

	PackedVector3Array new_normals = normals;

	for (int i = 0; i < normals.size(); i++) {
		int prev_i = (i == 0) ? (is_closed ? normals.size() - 1 : 0) : (i - 1);
		int next_i = (i == normals.size() - 1) ? (is_closed ? 0 : normals.size() - 1) : (i + 1);

		if (!is_closed && (i == 0 || i == normals.size() - 1)) {
			continue;
		}

		Vector3 avg_normal = (normals[prev_i] + normals[i] + normals[next_i]) / 3.0f;
		new_normals.write[i] = avg_normal.normalized();
	}

	normals = new_normals;
}

PackedVector3Array CassiePath3D::get_sample_points(int p_count) const {
	if (p_count == points.size() || p_count <= 0) {
		return points;
	}

	PackedVector3Array samples;
	samples.resize(p_count);

	for (int i = 0; i < p_count; i++) {
		float t = (float)i / (p_count - 1);
		int segment_count = is_closed ? points.size() : (points.size() - 1);
		float index_f = t * segment_count;
		int index = (int)index_f;
		float frac = index_f - index;

		int next_index = (index + 1) % points.size();
		samples.write[i] = points[index].lerp(points[next_index], frac);
	}

	return samples;
}

PackedVector3Array CassiePath3D::get_sample_normals(int p_count) const {
	if (p_count == normals.size() || p_count <= 0) {
		return normals;
	}

	PackedVector3Array samples;
	samples.resize(p_count);

	for (int i = 0; i < p_count; i++) {
		float t = (float)i / (p_count - 1);
		int segment_count = is_closed ? normals.size() : (normals.size() - 1);
		float index_f = t * segment_count;
		int index = (int)index_f;
		float frac = index_f - index;

		int next_index = (index + 1) % normals.size();
		samples.write[i] = normals[index].slerp(normals[next_index], frac).normalized();
	}

	return samples;
}

float CassiePath3D::get_total_length() const {
	if (points.size() < 2) {
		return 0.0f;
	}

	float length = 0.0f;
	int end = is_closed ? points.size() : (points.size() - 1);

	for (int i = 0; i < end; i++) {
		int next_i = (i + 1) % points.size();
		length += points[i].distance_to(points[next_i]);
	}

	return length;
}

float CassiePath3D::get_average_segment_length() const {
	if (points.size() < 2) {
		return 0.0f;
	}

	int segment_count = is_closed ? points.size() : (points.size() - 1);
	return get_total_length() / segment_count;
}

CassiePath3D::CassiePath3D() {
}

CassiePath3D::~CassiePath3D() {
}
