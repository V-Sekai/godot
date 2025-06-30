/**************************************************************************/
/*  image_texture_3d.cpp                                                  */
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

#include "image_texture_3d.h"
#include "../../profiling.h"

namespace zylann::godot {

Ref<ImageTexture3D> create_image_texture_3d(
		const Image::Format p_format,
		const Vector3i resolution,
		const bool p_mipmaps,
		const TypedArray<Image> &p_data
) {
	ZN_PROFILE_SCOPE();

	Ref<ImageTexture3D> texture;
	texture.instantiate();

#if defined(ZN_GODOT)
	Vector<Ref<Image>> images = to_ref_vector(p_data);
	texture->create(p_format, resolution.x, resolution.y, resolution.z, p_mipmaps, images);

#elif defined(ZN_GODOT_EXTENSION)
	texture->create(p_format, resolution.x, resolution.y, resolution.z, p_mipmaps, p_data);
#endif

	return texture;
}

void update_image_texture_3d(ImageTexture3D &p_texture, const TypedArray<Image> p_data) {
	ZN_PROFILE_SCOPE();

#if defined(ZN_GODOT)
	Vector<Ref<Image>> images = to_ref_vector(p_data);
	p_texture.update(images);

#elif defined(ZN_GODOT_EXTENSION)
	p_texture.update(p_data);
#endif
}

} // namespace zylann::godot
