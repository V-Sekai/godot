/**************************************************************************/
/*  csg_sculpted_primitive.h                                              */
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

#include "csg_shape.h"

#include "core/variant/type_info.h"
#include "scene/resources/texture.h"

/**
 * Base class for sculpted primitives that extend CSG functionality.
 * Implements common sculpting parameters: path cuts, profile cuts,
 * hollow, taper, twist, shear, and revolutions.
 */
class CSGSculptedPrimitive3D : public CSGPrimitive3D {
	GDCLASS(CSGSculptedPrimitive3D, CSGPrimitive3D);

public:
	enum ProfileCurve {
		PROFILE_CURVE_CIRCLE,
		PROFILE_CURVE_SQUARE,
		PROFILE_CURVE_ISOTRI,
		PROFILE_CURVE_EQUALTRI,
		PROFILE_CURVE_RIGHTTRI,
		PROFILE_CURVE_CIRCLE_HALF,
	};

	enum PathCurve {
		PATH_CURVE_LINE,
		PATH_CURVE_CIRCLE,
		PATH_CURVE_CIRCLE_33,
		PATH_CURVE_CIRCLE2,
	};

	enum HollowShape {
		HOLLOW_SAME,
		HOLLOW_CIRCLE,
		HOLLOW_SQUARE,
		HOLLOW_TRIANGLE,
	};

protected:
	// Profile parameters (cross-section)
	ProfileCurve profile_curve = PROFILE_CURVE_CIRCLE;
	real_t profile_begin = 0.0; // 0.0 to 1.0
	real_t profile_end = 1.0; // 0.0 to 1.0
	real_t hollow = 0.0; // 0.0 to 1.0
	HollowShape hollow_shape = HOLLOW_SAME;

	// Path parameters (sweep/extrusion)
	PathCurve path_curve = PATH_CURVE_LINE;
	real_t path_begin = 0.0; // 0.0 to 1.0
	real_t path_end = 1.0; // 0.0 to 1.0
	Vector2 scale = Vector2(1.0, 1.0);
	Vector2 shear = Vector2(0.0, 0.0);
	real_t twist_begin = 0.0; // -1.0 to 1.0
	real_t twist_end = 0.0; // -1.0 to 1.0
	real_t radius_offset = 0.0;
	Vector2 taper = Vector2(0.0, 0.0); // -1.0 to 1.0
	real_t revolutions = 1.0;
	real_t skew = 0.0;

	Ref<Material> material;

	static void _bind_methods();

public:
	// Profile parameters
	void set_profile_curve(int p_curve);
	int get_profile_curve() const;

	void set_profile_begin(real_t p_begin);
	real_t get_profile_begin() const;

	void set_profile_end(real_t p_end);
	real_t get_profile_end() const;

	void set_hollow(real_t p_hollow);
	real_t get_hollow() const;

	void set_hollow_shape(int p_shape);
	int get_hollow_shape() const;

	// Path parameters
	void set_path_curve(int p_curve);
	int get_path_curve() const;

	void set_path_begin(real_t p_begin);
	real_t get_path_begin() const;

	void set_path_end(real_t p_end);
	real_t get_path_end() const;

	void set_profile_scale(const Vector2 &p_scale);
	Vector2 get_profile_scale() const;

	void set_shear(const Vector2 &p_shear);
	Vector2 get_shear() const;

	void set_twist_begin(real_t p_twist);
	real_t get_twist_begin() const;

	void set_twist_end(real_t p_twist);
	real_t get_twist_end() const;

	void set_radius_offset(real_t p_offset);
	real_t get_radius_offset() const;

	void set_taper(const Vector2 &p_taper);
	Vector2 get_taper() const;

	void set_revolutions(real_t p_revolutions);
	real_t get_revolutions() const;

	void set_skew(real_t p_skew);
	real_t get_skew() const;

	// Material
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	CSGSculptedPrimitive3D();
};

// Register enum types with Godot's type system (must be at global scope)
MAKE_ENUM_TYPE_INFO(CSGSculptedPrimitive3D::ProfileCurve);
MAKE_ENUM_TYPE_INFO(CSGSculptedPrimitive3D::PathCurve);
MAKE_ENUM_TYPE_INFO(CSGSculptedPrimitive3D::HollowShape);

/**
 * Sculpted box primitive with advanced sculpting parameters.
 */
class CSGSculptedBox3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedBox3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	Vector3 size = Vector3(1, 1, 1);

protected:
	static void _bind_methods();

public:
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	CSGSculptedBox3D();
};

/**
 * Sculpted cylinder primitive with advanced sculpting parameters.
 */
class CSGSculptedCylinder3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedCylinder3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	real_t radius = 0.5;
	real_t height = 2.0;

protected:
	static void _bind_methods();

public:
	void set_radius(const real_t p_radius);
	real_t get_radius() const;

	void set_height(const real_t p_height);
	real_t get_height() const;

	CSGSculptedCylinder3D();
};

/**
 * Sculpted sphere primitive with advanced sculpting parameters.
 */
class CSGSculptedSphere3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedSphere3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	real_t radius = 0.5;

protected:
	static void _bind_methods();

public:
	void set_radius(const real_t p_radius);
	real_t get_radius() const;

	CSGSculptedSphere3D();
};

/**
 * Sculpted torus primitive with advanced sculpting parameters.
 */
class CSGSculptedTorus3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedTorus3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	real_t inner_radius = 0.25;
	real_t outer_radius = 0.5;

protected:
	static void _bind_methods();

public:
	void set_inner_radius(const real_t p_inner_radius);
	real_t get_inner_radius() const;

	void set_outer_radius(const real_t p_outer_radius);
	real_t get_outer_radius() const;

	CSGSculptedTorus3D();
};

/**
 * Sculpted prism primitive with advanced sculpting parameters.
 * A 5-sided solid (box with one very small face).
 */
class CSGSculptedPrism3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedPrism3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	Vector3 size = Vector3(1, 1, 1);

protected:
	static void _bind_methods();

public:
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	CSGSculptedPrism3D();
};

/**
 * Sculpted tube primitive with advanced sculpting parameters.
 * A hollow cylinder variant.
 */
class CSGSculptedTube3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedTube3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	real_t inner_radius = 0.25;
	real_t outer_radius = 0.5;
	real_t height = 2.0;

protected:
	static void _bind_methods();

public:
	void set_inner_radius(const real_t p_inner_radius);
	real_t get_inner_radius() const;

	void set_outer_radius(const real_t p_outer_radius);
	real_t get_outer_radius() const;

	void set_height(const real_t p_height);
	real_t get_height() const;

	CSGSculptedTube3D();
};

/**
 * Sculpted ring primitive with advanced sculpting parameters.
 * A torus variant with different proportions.
 */
class CSGSculptedRing3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedRing3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	real_t inner_radius = 0.4;
	real_t outer_radius = 0.5;
	real_t height = 0.1; // Ring has a small height/thickness

protected:
	static void _bind_methods();

public:
	void set_inner_radius(const real_t p_inner_radius);
	real_t get_inner_radius() const;

	void set_outer_radius(const real_t p_outer_radius);
	real_t get_outer_radius() const;

	void set_height(const real_t p_height);
	real_t get_height() const;

	CSGSculptedRing3D();
};

/**
 * Texture-based sculpted primitive.
 * Uses a texture where RGB values encode 3D coordinates (R=X, G=Y, B=Z).
 * This allows for highly variable organic shapes defined by texture data.
 */
class CSGSculptedTexture3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedTexture3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	Ref<Texture2D> sculpt_texture;
	bool mirror = false;
	bool invert = false;

	void _texture_changed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_sculpt_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_sculpt_texture() const;

	void set_mirror(bool p_mirror);
	bool get_mirror() const;

	void set_invert(bool p_invert);
	bool get_invert() const;

	CSGSculptedTexture3D();
};
