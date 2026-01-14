/**************************************************************************/
/*  csg_sculpted_texture.h                                                */
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

#include "csg_sculpted_primitive_base.h"

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
