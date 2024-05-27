/**************************************************************************/
/*  intersector.h                                                         */
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

#ifndef INTERSECTOR_H
#define INTERSECTOR_H

#include "slicer_face.h"

/**
 * Contains functions related to finding intersection points
 * on SlicerFaces
 */
namespace Intersector {
// Note that this is slightly different than Face3::Side,
// as it refers to the position of a single Vector3 rather
// than a face
enum SideOfPlane {
	OVER,
	UNDER,
	ON,
};

struct SplitResult {
	Ref<Material> material;
	Vector<SlicerFace> upper_faces;
	Vector<SlicerFace> lower_faces;
	Vector<Vector3> intersection_points;

	void reset() {
		upper_faces.resize(0);
		lower_faces.resize(0);
		intersection_points.resize(0);
	}

	SplitResult() {}
};

/**
 * Calculates which side of the passed in plane the given point falls on
 */
SideOfPlane get_side_of(const Plane &plane, Vector3 point);

/**
 * Performs an intersection on the given face using the passed in plane and stores
 * the result in the result param.
 */
void split_face_by_plane(const Plane &plane, const SlicerFace &face, SplitResult &result);
} //namespace Intersector

#endif // INTERSECTOR_H
