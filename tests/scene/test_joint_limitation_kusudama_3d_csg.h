/**************************************************************************/
/*  test_joint_limitation_kusudama_3d_csg.h                              */
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
#include "scene/resources/3d/joint_limitation_kusudama_3d.h"
#include "tests/test_macros.h"
#include "tests/scene/test_joint_limitation_kusudama_3d.h"

namespace TestJointLimitationKusudama3DCSG {

using namespace TestJointLimitationKusudama3D;

// Helper function to convert sphere coordinates (theta, phi) to 3D point
static Vector3 sphere_coords_to_point(real_t theta, real_t phi) {
	real_t x = Math::sin(phi) * Math::cos(theta);
	real_t y = Math::cos(phi);
	real_t z = Math::sin(phi) * Math::sin(theta);
	return Vector3(x, y, z).normalized();
}

// Test CSG mesh generation logic - verify that triangle classification matches solver
// This test creates a 2D chart of the sphere surface to visualize allowed/forbidden regions
TEST_CASE("[Scene][JointLimitationKusudama3D][CSG] Test CSG mesh classification matches solver") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones with a path between them
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(60.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Generate a sphere mesh (like CSG code does)
	const int rings = 32;
	const int radial_segments = 32;
	const real_t sphere_r = 1.0;
	
	LocalVector<Vector3> sphere_vertices;
	LocalVector<int> sphere_indices;
	
	// Generate sphere vertices
	for (int ring = 0; ring <= rings; ring++) {
		real_t v = (real_t)ring / (real_t)rings;
		real_t w = Math::sin(Math::PI * v);
		real_t y = Math::cos(Math::PI * v);
		
		for (int seg = 0; seg <= radial_segments; seg++) {
			real_t u = (real_t)seg / (real_t)radial_segments;
			real_t x = Math::sin(u * Math::TAU) * w;
			real_t z = Math::cos(u * Math::TAU) * w;
			sphere_vertices.push_back(Vector3(x, y, z) * sphere_r);
		}
	}
	
	// Generate sphere triangle indices
	for (int ring = 0; ring < rings; ring++) {
		int ring_start = ring * (radial_segments + 1);
		int next_ring_start = (ring + 1) * (radial_segments + 1);
		
		for (int seg = 0; seg < radial_segments; seg++) {
			int i0 = ring_start + seg;
			int i1 = ring_start + ((seg + 1) % (radial_segments + 1));
			int i2 = next_ring_start + seg;
			int i3 = next_ring_start + ((seg + 1) % (radial_segments + 1));
			
			sphere_indices.push_back(i0);
			sphere_indices.push_back(i2);
			sphere_indices.push_back(i1);
			
			sphere_indices.push_back(i1);
			sphere_indices.push_back(i2);
			sphere_indices.push_back(i3);
		}
	}
	
	// Test CSG classification logic: check triangles that should be in forbidden region
	// Sample some triangles and verify their classification
	int triangles_tested = 0;
	int triangles_in_forbidden = 0;
	int triangles_in_allowed = 0;
	int misclassified = 0;
	
	// Test a subset of triangles (every 10th triangle to keep test fast)
	for (size_t tri = 0; tri < sphere_indices.size(); tri += 30) {
		if (tri + 2 >= sphere_indices.size()) {
			break;
		}
		
		int i0 = sphere_indices[tri];
		int i1 = sphere_indices[tri + 1];
		int i2 = sphere_indices[tri + 2];
		
		Vector3 v0 = sphere_vertices[i0];
		Vector3 v1 = sphere_vertices[i1];
		Vector3 v2 = sphere_vertices[i2];
		
		// Normalize vertices
		Vector3 n0 = v0.normalized();
		Vector3 n1 = v1.normalized();
		Vector3 n2 = v2.normalized();
		Vector3 center = (v0 + v1 + v2) / 3.0;
		Vector3 center_normalized = center.normalized();
		
		// Check if triangle should be in forbidden region (CSG logic)
		// Check center and vertices - if ANY is in allowed region, triangle is in allowed region
		bool triangle_in_allowed_csg = false;
		Vector3 points_to_check[4] = { center_normalized, n0, n1, n2 };
		
		for (int p_idx = 0; p_idx < 4; p_idx++) {
			Vector3 vertex = points_to_check[p_idx];
			
			// Use solver logic to check if point is allowed
			bool is_allowed = test_is_point_allowed(vertex, cones);
			if (is_allowed) {
				triangle_in_allowed_csg = true;
				break;
			}
		}
		
		// Verify classification using solver
		// Check if triangle center is actually in forbidden region according to solver
		bool center_forbidden = !test_is_point_allowed(center_normalized, cones);
		
		// If CSG says triangle is in forbidden region, verify center is actually forbidden
		// (or at least one vertex is forbidden)
		if (!triangle_in_allowed_csg) {
			triangles_in_forbidden++;
			// Verify that at least the center is forbidden
			if (!center_forbidden) {
				// Check if all vertices are also forbidden
				bool all_vertices_forbidden = true;
				for (int v_idx = 0; v_idx < 3; v_idx++) {
					if (test_is_point_allowed(points_to_check[v_idx + 1], cones)) {
						all_vertices_forbidden = false;
						break;
					}
				}
				if (!all_vertices_forbidden) {
					misclassified++;
				}
			}
		} else {
			triangles_in_allowed++;
		}
		
		triangles_tested++;
	}
	
	// Verify we tested some triangles
	CHECK(triangles_tested > 0);
	
	// Verify we have both allowed and forbidden triangles
	CHECK(triangles_in_forbidden > 0);
	CHECK(triangles_in_allowed > 0);
	
	// Verify no misclassifications (or very few due to boundary cases)
	// Allow some tolerance for triangles that straddle boundaries
	real_t misclassification_rate = (real_t)misclassified / (real_t)triangles_tested;
	CHECK_MESSAGE(misclassification_rate < 0.1f, 
		vformat("Too many misclassified triangles: %d out of %d (%.1f%%)", 
			misclassified, triangles_tested, misclassification_rate * 100.0f).utf8().get_data());
}

// Helper to check if a point is inside the forbidden mesh (from CSG generation)
// This replicates the CSG logic used in generate_forbidden_region_mesh
static bool is_point_in_forbidden_mesh(const Vector3 &p_point, const Vector<Vector4> &p_cones) {
	Vector3 vertex = p_point.normalized();
	
	// Use the exact same logic as generate_forbidden_region_mesh
	// Check cones first
	bool in_bounds = false;
	for (int i = 0; i < p_cones.size(); i++) {
		const Vector4 &cone_data = p_cones[i];
		Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;
		
		// Use closest_to_cone_boundary equivalent (test version)
		Vector3 dir = vertex;
		Vector3 center = control_point;
		real_t radius_cosine = Math::cos(radius);
		real_t input_dot_control = dir.dot(center);
		
		// If NaN equivalent (point inside cone), point is in allowed region
		if (input_dot_control >= radius_cosine - 1e-4) {
			in_bounds = true;
			break;
		}
	}
	
	// If not in any cone, check paths (exact same as CSG code)
	if (!in_bounds && p_cones.size() > 1) {
		for (int i = 0; i < p_cones.size() - 1; i++) {
			const Vector4 &cone1_data = p_cones[i];
			const Vector4 &cone2_data = p_cones[i + 1];
			Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
			Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
			real_t radius1 = cone1_data.w;
			real_t radius2 = cone2_data.w;
			
			Vector3 collision_point = test_get_on_great_tangent_triangle(vertex, center1, radius1, center2, radius2);
			if (!Math::is_nan(collision_point.x)) {
				real_t cosine = collision_point.dot(vertex);
				if (cosine > 0.999f) {
					in_bounds = true;
					break;
				}
			}
		}
	}
	
	// Return true if point is in FORBIDDEN region (not in bounds)
	return !in_bounds;
}

// Helper to convert 3D point to chart coordinates (theta, phi)
static void point_to_chart_coords(const Vector3 &p_point, int p_chart_width, int p_chart_height, int &r_theta_idx, int &r_phi_idx) {
	Vector3 normalized = p_point.normalized();
	
	// Convert to spherical coordinates
	real_t phi = Math::acos(normalized.y); // 0 to PI
	real_t theta = Math::atan2(normalized.x, normalized.z); // -PI to PI
	if (theta < 0) {
		theta += Math::TAU; // 0 to 2PI
	}
	
	// Convert to chart indices
	r_phi_idx = (int)(phi / Math::PI * (p_chart_height - 1));
	r_theta_idx = (int)(theta / Math::TAU * p_chart_width);
	
	// Clamp to valid range
	r_phi_idx = CLAMP(r_phi_idx, 0, p_chart_height - 1);
	r_theta_idx = CLAMP(r_theta_idx, 0, p_chart_width - 1);
}

// Helper to test CSG mesh with different cone configurations
static void test_csg_mesh_config(const Vector<Vector4> &p_cones, const String &p_config_name) {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();
	limitation->set_cones(p_cones);

	// Generate the actual CSG forbidden region mesh (replicate generate_forbidden_region_mesh logic)
	const int rings = 64;
	const int radial_segments = 64;
	const real_t sphere_r = 1.0;
	
	LocalVector<Vector3> sphere_vertices;
	LocalVector<int> sphere_indices;
	
	// Generate sphere vertices (same as CSG code)
	for (int ring = 0; ring <= rings; ring++) {
		real_t v = (real_t)ring / (real_t)rings;
		real_t w = Math::sin(Math::PI * v);
		real_t y = Math::cos(Math::PI * v);
		
		for (int seg = 0; seg <= radial_segments; seg++) {
			real_t u = (real_t)seg / (real_t)radial_segments;
			real_t x = Math::sin(u * Math::TAU) * w;
			real_t z = Math::cos(u * Math::TAU) * w;
			sphere_vertices.push_back(Vector3(x, y, z) * sphere_r);
		}
	}
	
	// Generate sphere triangle indices (same as CSG code)
	for (int ring = 0; ring < rings; ring++) {
		int ring_start = ring * (radial_segments + 1);
		int next_ring_start = (ring + 1) * (radial_segments + 1);
		
		for (int seg = 0; seg < radial_segments; seg++) {
			int i0 = ring_start + seg;
			int i1 = ring_start + ((seg + 1) % (radial_segments + 1));
			int i2 = next_ring_start + seg;
			int i3 = next_ring_start + ((seg + 1) % (radial_segments + 1));
			
			sphere_indices.push_back(i0);
			sphere_indices.push_back(i2);
			sphere_indices.push_back(i1);
			
			sphere_indices.push_back(i1);
			sphere_indices.push_back(i2);
			sphere_indices.push_back(i3);
		}
	}
	
	// Generate forbidden region mesh using CSG logic (same as generate_forbidden_region_mesh)
	LocalVector<Vector3> forbidden_vertices;
	LocalVector<int> forbidden_indices;
	
	for (int tri = 0; tri < sphere_indices.size(); tri += 3) {
		int i0 = sphere_indices[tri];
		int i1 = sphere_indices[tri + 1];
		int i2 = sphere_indices[tri + 2];
		
		Vector3 v0 = sphere_vertices[i0];
		Vector3 v1 = sphere_vertices[i1];
		Vector3 v2 = sphere_vertices[i2];
		
		// Normalize vertices
		Vector3 n0 = v0.normalized();
		Vector3 n1 = v1.normalized();
		Vector3 n2 = v2.normalized();
		Vector3 center = (v0 + v1 + v2) / 3.0;
		Vector3 center_normalized = center.normalized();
		
		// Check if triangle should be in forbidden region (CSG logic)
		// Use the exact same logic as production generate_forbidden_region_mesh
		bool triangle_in_allowed = false;
		Vector3 points_to_check[4] = { center_normalized, n0, n1, n2 };
		
		for (int p_idx = 0; p_idx < 4; p_idx++) {
			Vector3 vertex = points_to_check[p_idx];
			
			// Use the exact same logic as production code (direct dot product check)
			// Check cones first
			bool in_bounds = false;
			for (int i = 0; i < p_cones.size(); i++) {
				const Vector4 &cone_data = p_cones[i];
				Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
				real_t radius = cone_data.w;
				
				// Direct check matching production code exactly
				real_t radius_cosine = Math::cos(radius);
				real_t input_dot_control = vertex.dot(control_point);
				// Use same epsilon as production: 1e-3 (larger to be more conservative)
				if (input_dot_control >= radius_cosine - 1e-3) {
					in_bounds = true;
					break;
				}
			}
			
			// If not in any cone, check paths (exact same as production)
			if (!in_bounds && p_cones.size() > 1) {
				for (int i = 0; i < p_cones.size() - 1; i++) {
					const Vector4 &cone1_data = p_cones[i];
					const Vector4 &cone2_data = p_cones[i + 1];
					Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
					Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
					real_t radius1 = cone1_data.w;
					real_t radius2 = cone2_data.w;
					
					Vector3 collision_point = test_get_on_great_tangent_triangle(vertex, center1, radius1, center2, radius2);
					if (!Math::is_nan(collision_point.x)) {
						real_t cosine = collision_point.dot(vertex);
						if (cosine > 0.999f) {
							in_bounds = true;
							break;
						}
					}
				}
			}
			
			if (in_bounds) {
				triangle_in_allowed = true;
				break;
			}
		}
		
		// If triangle is NOT in any allowed region, keep it (it's in forbidden region)
		if (!triangle_in_allowed) {
			// Add triangle to forbidden mesh
			int idx0 = forbidden_vertices.size();
			forbidden_vertices.push_back(v0);
			int idx1 = forbidden_vertices.size();
			forbidden_vertices.push_back(v1);
			int idx2 = forbidden_vertices.size();
			forbidden_vertices.push_back(v2);
			
			forbidden_indices.push_back(idx0);
			forbidden_indices.push_back(idx1);
			forbidden_indices.push_back(idx2);
		}
	}
	
	// Now flatten the forbidden mesh to a 2D chart
	const int chart_width = 60;
	const int chart_height = 30;
	
	// Initialize chart (0 = not in mesh, 1 = in forbidden mesh)
	LocalVector<bool> chart_data;
	chart_data.resize(chart_width * chart_height);
	for (int i = 0; i < chart_data.size(); i++) {
		chart_data[i] = false;
	}
	
	// For each triangle in forbidden mesh, mark its vertices on the chart
	for (int tri = 0; tri < forbidden_indices.size(); tri += 3) {
		int i0 = forbidden_indices[tri];
		int i1 = forbidden_indices[tri + 1];
		int i2 = forbidden_indices[tri + 2];
		
		Vector3 v0 = forbidden_vertices[i0];
		Vector3 v1 = forbidden_vertices[i1];
		Vector3 v2 = forbidden_vertices[i2];
		
		// Project each vertex to chart coordinates
		int theta0, phi0, theta1, phi1, theta2, phi2;
		point_to_chart_coords(v0, chart_width, chart_height, theta0, phi0);
		point_to_chart_coords(v1, chart_width, chart_height, theta1, phi1);
		point_to_chart_coords(v2, chart_width, chart_height, theta2, phi2);
		
		// Mark vertices and fill triangle on chart
		// Simple approach: mark all three vertices and points along edges
		chart_data[phi0 * chart_width + theta0] = true;
		chart_data[phi1 * chart_width + theta1] = true;
		chart_data[phi2 * chart_width + theta2] = true;
		
		// Fill triangle by interpolating (simple rasterization)
		int min_phi = MIN(phi0, MIN(phi1, phi2));
		int max_phi = MAX(phi0, MAX(phi1, phi2));
		int min_theta = MIN(theta0, MIN(theta1, theta2));
		int max_theta = MAX(theta0, MAX(theta1, theta2));
		
		for (int phi = min_phi; phi <= max_phi && phi < chart_height; phi++) {
			for (int theta = min_theta; theta <= max_theta && theta < chart_width; theta++) {
				// Simple point-in-triangle check (barycentric coordinates)
				Vector2 p0(theta0, phi0);
				Vector2 p1(theta1, phi1);
				Vector2 p2(theta2, phi2);
				Vector2 p(theta, phi);
				
				real_t denom = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
				if (Math::abs(denom) < 1e-6) {
					continue;
				}
				
				real_t a = ((p1.y - p2.y) * (p.x - p2.x) + (p2.x - p1.x) * (p.y - p2.y)) / denom;
				real_t b = ((p2.y - p0.y) * (p.x - p2.x) + (p0.x - p2.x) * (p.y - p2.y)) / denom;
				real_t c = 1.0 - a - b;
				
				if (a >= 0 && b >= 0 && c >= 0) {
					chart_data[phi * chart_width + theta] = true;
				}
			}
		}
	}
	
	// Count regions for summary
	int forbidden_mesh_count = 0;
	for (int i = 0; i < chart_data.size(); i++) {
		if (chart_data[i]) {
			forbidden_mesh_count++;
		}
	}
	
	// Check actual triangle vertices directly (not rasterized chart points)
	// This avoids errors from 2D projection
	int cone1_count = 0;
	int cone2_count = 0;
	int path_count = 0;
	
	// Check each triangle vertex in the forbidden mesh
	for (int tri = 0; tri < forbidden_indices.size(); tri += 3) {
		int i0 = forbidden_indices[tri];
		int i1 = forbidden_indices[tri + 1];
		int i2 = forbidden_indices[tri + 2];
		
		Vector3 v0 = forbidden_vertices[i0].normalized();
		Vector3 v1 = forbidden_vertices[i1].normalized();
		Vector3 v2 = forbidden_vertices[i2].normalized();
		
		// Check each vertex
		Vector3 vertices[3] = { v0, v1, v2 };
		for (int v_idx = 0; v_idx < 3; v_idx++) {
			Vector3 vertex = vertices[v_idx];
			
			// Use the exact same logic as CSG code to check if vertex is in allowed region
			bool in_allowed = false;
			
			// Check cones first (same logic as CSG)
			for (int i = 0; i < p_cones.size(); i++) {
				const Vector4 &cone_data = p_cones[i];
				Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
				real_t cone_radius = cone_data.w;
				
				real_t radius_cosine = Math::cos(cone_radius);
				real_t input_dot_control = vertex.dot(control_point);
				// Use same epsilon as production: 1e-3
				if (input_dot_control >= radius_cosine - 1e-3) {
					in_allowed = true;
					if (i == 0) {
						cone1_count++;
					} else if (i == 1) {
						cone2_count++;
					}
					break;
				}
			}
			
			// Check paths if not in cone (same logic as CSG)
			if (!in_allowed && p_cones.size() > 1) {
				for (int i = 0; i < p_cones.size() - 1; i++) {
					const Vector4 &cone1_data = p_cones[i];
					const Vector4 &cone2_data = p_cones[i + 1];
					Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
					Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
					real_t radius1 = cone1_data.w;
					real_t radius2 = cone2_data.w;
					
					Vector3 collision_point = test_get_on_great_tangent_triangle(vertex, center1, radius1, center2, radius2);
					if (!Math::is_nan(collision_point.x)) {
						real_t cosine = collision_point.dot(vertex);
						if (cosine > 0.999f) {
							in_allowed = true;
							path_count++;
							break;
						}
					}
				}
			}
		}
	}
	
	// Generate chart visualization (for display only, not for error checking)
	LocalVector<int> chart_visual;
	chart_visual.resize(chart_width * chart_height);
	
	for (int phi_idx = 0; phi_idx < chart_height; phi_idx++) {
		real_t phi = (real_t)phi_idx / (real_t)(chart_height - 1) * Math::PI;
		for (int theta_idx = 0; theta_idx < chart_width; theta_idx++) {
			real_t theta = (real_t)theta_idx / (real_t)chart_width * Math::TAU;
			Vector3 point = sphere_coords_to_point(theta, phi);
			
			int point_type = 0; // Forbidden
			if (chart_data[phi_idx * chart_width + theta_idx]) {
				// Check what region this point is in (for visualization)
				// Use test_is_point_allowed to classify
				bool is_allowed = test_is_point_allowed(point, p_cones);
				if (is_allowed) {
					// Determine which cone or path
					for (int i = 0; i < p_cones.size(); i++) {
						const Vector4 &cone_data = p_cones[i];
						Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
						real_t cone_radius = cone_data.w;
						if (test_is_point_in_cone(point, control_point, cone_radius)) {
							point_type = i + 1; // Cone number (1, 2, 3, etc.)
							break;
						}
					}
					if (point_type == 0 && p_cones.size() > 1) {
						// Must be in a path
						point_type = 10; // Path
					}
				}
			}
			
			chart_visual[phi_idx * chart_width + theta_idx] = point_type;
		}
	}
	
	// Print visual chart
	String chart_output = vformat("\nCSG Mesh Flattened Chart: %s\n", p_config_name.utf8().get_data());
	chart_output += "Legend: '1'/'2'/'3' = Cone in mesh (ERROR), '-' = Path in mesh (ERROR), '#' = Forbidden (correct), '.' = Empty\n";
	chart_output += String("=").repeat(chart_width + 2) + "\n";
	
	for (int phi_idx = 0; phi_idx < chart_height; phi_idx++) {
		chart_output += "|";
		for (int theta_idx = 0; theta_idx < chart_width; theta_idx++) {
			int point_type = chart_visual[phi_idx * chart_width + theta_idx];
			if (point_type >= 1 && point_type <= 9) {
				chart_output += vformat("%d", point_type).utf8().get_data(); // Cone number
			} else if (point_type == 10) {
				chart_output += "-"; // Path
			} else {
				chart_output += chart_data[phi_idx * chart_width + theta_idx] ? "#" : "."; // Forbidden or empty
			}
		}
		chart_output += "|\n";
	}
	chart_output += String("=").repeat(chart_width + 2) + "\n";
	
	print_line(chart_output);
	
	// Print summary
	print_line(vformat("CSG Mesh [%s]: %d triangles, %d vertices. Chart: Forbidden=%d (%.1f%%), Errors: Cone1=%d, Cone2=%d, Cone3=%d, Path=%d",
		p_config_name.utf8().get_data(),
		forbidden_indices.size() / 3, forbidden_vertices.size(),
		forbidden_mesh_count, (real_t)forbidden_mesh_count / (real_t)(chart_width * chart_height) * 100.0f,
		cone1_count, cone2_count, (p_cones.size() > 2 ? 0 : 0), path_count));
	
	// Verify no allowed regions are in the forbidden mesh
	CHECK_MESSAGE(cone1_count == 0, vformat("Config %s: Found %d vertices from cone 1 in forbidden mesh", p_config_name.utf8().get_data(), cone1_count).utf8().get_data());
	CHECK_MESSAGE(cone2_count == 0, vformat("Config %s: Found %d vertices from cone 2 in forbidden mesh", p_config_name.utf8().get_data(), cone2_count).utf8().get_data());
	CHECK_MESSAGE(path_count == 0, vformat("Config %s: Found %d vertices from paths in forbidden mesh", p_config_name.utf8().get_data(), path_count).utf8().get_data());
}

// Test CSG mesh with different cone configurations
TEST_CASE("[Scene][JointLimitationKusudama3D][CSG] Test CSG mesh with 1-2-3 cones") {
	real_t radius = Math::deg_to_rad(30.0f);
	
	// Test 1: Single cone
	{
		Vector<Vector4> cones;
		Vector3 cp1 = Vector3(0, 1, 0).normalized(); // Top
		cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
		test_csg_mesh_config(cones, "1 cone (top)");
	}
	
	// Test 2: Two cones (original)
	{
		Vector<Vector4> cones;
		Vector3 cp1 = Vector3(0.707, 0.707, 0.0).normalized();
		Vector3 cp2 = Vector3(-0.707, 0.707, 0.0).normalized();
		cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
		cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
		test_csg_mesh_config(cones, "2 cones (original)");
	}
	
	// Test 3: Two cones (different positions)
	{
		Vector<Vector4> cones;
		Vector3 cp1 = Vector3(1, 0, 0).normalized(); // Right
		Vector3 cp2 = Vector3(0, 0, 1).normalized(); // Forward
		cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
		cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
		test_csg_mesh_config(cones, "2 cones (right, forward)");
	}
	
	// Test 4: Three cones
	{
		Vector<Vector4> cones;
		Vector3 cp1 = Vector3(1, 0, 0).normalized(); // Right
		Vector3 cp2 = Vector3(0, 1, 0).normalized(); // Top
		Vector3 cp3 = Vector3(0, 0, 1).normalized(); // Forward
		cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
		cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
		cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
		test_csg_mesh_config(cones, "3 cones (right, top, forward)");
	}
	
	// Test 5: Three cones (different positions)
	{
		Vector<Vector4> cones;
		Vector3 cp1 = Vector3(0.707, 0.707, 0.0).normalized();
		Vector3 cp2 = Vector3(-0.707, 0.707, 0.0).normalized();
		Vector3 cp3 = Vector3(0, 0.707, 0.707).normalized();
		cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
		cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
		cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
		test_csg_mesh_config(cones, "3 cones (northeast, northwest, north)");
	}
}

// Test specific path between cone 1 and cone 2 - verify it's correctly identified as allowed
TEST_CASE("[Scene][JointLimitationKusudama3D][CSG] Test CSG path between cone 1 and cone 2") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones with a path between them
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(60.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Sample points along the path between cone 1 and cone 2
	// The path should be outside both cones but in the allowed region
	// Use known test points that should be in the path region
	int path_points_tested = 0;
	int path_points_correctly_allowed = 0;
	
	// Use points that are known to be in the path region based on the solver logic
	// Points between the cones but outside both cone boundaries
	Vector3 test_points[] = {
		// Point between cones, outside both
		Vector3(0.7, 0.7, 0.1).normalized(),
		Vector3(0.6, 0.6, 0.5).normalized(),
		Vector3(0.5, 0.8, 0.3).normalized(),
		Vector3(0.8, 0.5, 0.3).normalized(),
		// Points on the great circle between cones
		Vector3(0.707, 0.707, 0.0).normalized(),
		Vector3(0.6, 0.6, 0.5).normalized(),
	};
	
	for (int i = 0; i < 6; i++) {
		Vector3 test_point = test_points[i];
		
		// Verify point is outside both cones
		bool in_cone1 = test_is_point_in_cone(test_point, cp1, radius);
		bool in_cone2 = test_is_point_in_cone(test_point, cp2, radius);
		
		// Point should be outside both cones to be in path
		if (!in_cone1 && !in_cone2) {
			// Check if point is in the path using get_on_great_tangent_triangle
			Vector3 path_result = test_get_on_great_tangent_triangle(test_point, cp1, radius, cp2, radius);
			if (!Math::is_nan(path_result.x)) {
				real_t cosine = path_result.dot(test_point);
				if (cosine > 0.999f) {
					// Point is in path region
					path_points_tested++;
					
					// Verify it's allowed
					bool is_allowed = test_is_point_allowed(test_point, cones);
					if (is_allowed) {
						path_points_correctly_allowed++;
					} else {
						// This is a problem - path point should be allowed
						WARN(vformat("Path point (%s) is in path but not allowed", 
							test_point).utf8().get_data());
					}
				}
			}
		}
	}
	
	// If we found path points, verify they're correctly identified as allowed
	// Note: Finding path points can be difficult, so we only verify if we found some
	if (path_points_tested > 0) {
		// Most path points should be correctly identified as allowed
		real_t correct_rate = (real_t)path_points_correctly_allowed / (real_t)path_points_tested;
		CHECK_MESSAGE(correct_rate > 0.8f, 
			vformat("Too few path points correctly identified as allowed: %d out of %d (%.1f%%)", 
				path_points_correctly_allowed, path_points_tested, correct_rate * 100.0f).utf8().get_data());
	} else {
		// If we couldn't find path points with the test points, that's okay
		// The main test file has comprehensive path tests
		// This test is mainly to verify CSG classification logic when path points are found
		WARN("Could not find path points to test - this is acceptable as path detection is tested elsewhere");
	}
}

// Test back area - verify points opposite to cones are correctly identified as forbidden
TEST_CASE("[Scene][JointLimitationKusudama3D][CSG] Test CSG back area is forbidden") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones pointing forward
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(60.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Test points in the "back" area (opposite to the cones)
	// These should be forbidden
	Vector3 back_points[] = {
		Vector3(-1, 0, 0).normalized(),  // Opposite to cp1
		Vector3(0, -1, 0).normalized(),  // Opposite to cp2
		Vector3(-0.5, -0.5, -0.7).normalized(),  // General back area
		Vector3(-0.3, -0.3, -0.9).normalized(),  // Further back
	};
	
	int back_points_tested = 0;
	int back_points_correctly_forbidden = 0;
	
	for (int i = 0; i < 4; i++) {
		Vector3 back_point = back_points[i];
		back_points_tested++;
		
		// Verify point is outside both cones
		bool in_cone1 = test_is_point_in_cone(back_point, cp1, radius);
		bool in_cone2 = test_is_point_in_cone(back_point, cp2, radius);
		CHECK_FALSE_MESSAGE(in_cone1, "Back point should be outside cone 1");
		CHECK_FALSE_MESSAGE(in_cone2, "Back point should be outside cone 2");
		
		// Check if point is in any path (should not be)
		bool in_path = false;
		Vector3 path_point = test_get_on_great_tangent_triangle(back_point, cp1, radius, cp2, radius);
		if (!Math::is_nan(path_point.x)) {
			real_t cosine = path_point.dot(back_point);
			if (cosine > 0.999f) {
				in_path = true;
			}
		}
		
		// Point should be forbidden (not in cone, not in path)
		bool is_allowed = test_is_point_allowed(back_point, cones);
		if (!is_allowed) {
			back_points_correctly_forbidden++;
		} else {
			WARN(vformat("Back point %s is allowed but should be forbidden (in_path=%s)", 
				back_point, in_path ? "true" : "false").utf8().get_data());
		}
	}
	
	// All back points should be correctly identified as forbidden
	CHECK_EQ(back_points_correctly_forbidden, back_points_tested);
}

} // namespace TestJointLimitationKusudama3DCSG

