/*************************************************************************/
/*  color_transform.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "color_transform.h"
#include "core/error/error_macros.h"
#include "core/io/image.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"
#include "pngconf.h"
#include "scene/resources/texture.h"
#include "servers/rendering_server.h"
#include <lcms2.h>
#include <stddef.h>
#include <stdint.h>

bool ColorTransform::is_valid() {
	return src_profile.is_valid() && src_profile->is_valid() && dst_profile.is_valid() && dst_profile->is_valid();
}

void ColorTransform::set_src_profile(Ref<ColorProfile> p_profile) {
	if (p_profile.is_null()) {
		ERR_PRINT("Source profile is null.");
		return;
	}
	src_profile = p_profile;
}

void ColorTransform::set_dst_profile(Ref<ColorProfile> p_profile) {
	if (p_profile.is_null()) {
		ERR_PRINT("Destination profile is null.");
		return;
	}
	dst_profile = p_profile;
}

void ColorTransform::set_intent(Intent p_intent) {
	intent = (cmsUInt32Number)p_intent;
}

void ColorTransform::set_black_point_compensation(bool p_black_point_compensation) {
	use_bpc = p_black_point_compensation;
}

RID ColorTransform::get_color_correction() {
	if (!is_valid()) {
		ERR_PRINT("Transform is not valid.");
		return RID();
	}

	// Use LCMS to transform data
	cmsHPROFILE src = src_profile->get_profile_handle().profile; // handles owned by ColorProfile, don't free
	cmsHPROFILE dst = dst_profile->get_profile_handle().profile;
	if (src == nullptr || dst == nullptr) {
		ERR_PRINT("Transform has invalid profiles. This should have been checked earlier.");
		return RID();
	}
	cmsUInt32Number flags = use_bpc ? cmsFLAGS_BLACKPOINTCOMPENSATION : 0;
	// Half allows hdr icc profiles to be used
	// Identity check
	cmsHTRANSFORM transform = cmsCreateTransform(src, TYPE_RGB_HALF_FLT, dst, TYPE_RGB_HALF_FLT, intent, flags);

	ERR_FAIL_COND_V_MSG(!transform, RID(), "Failed to create lcms transform.");
	List<Ref<Image>> slice_list;
	int32_t dim = 256;
	for (int32_t z = 0; z < dim; z++) {
		Ref<Image> lut = Image::create_empty(dim, dim, false, Image::FORMAT_RGBH);
		for (int32_t y = 0; y < dim; y++) {
			for (int32_t x = 0; x < dim; x++) {
				Color c;
				c.r = float(x) / dim;
				c.g = float(y) / dim;
				c.b = float(z) / dim;
				lut->set_pixel(x, y, c);
			}
		}
		LocalVector<uint8_t> data;
		data.resize(Image::get_image_data_size(dim, dim, Image::FORMAT_RGBH));
		cmsDoTransform(transform, lut->get_data().ptr(), data.ptr(), dim * dim); // cmsDoTransform wants number of pixels
		Ref<Image> out_lut = Image::create_from_data(dim, dim, false, Image::FORMAT_RGBH, data);
		slice_list.push_front(out_lut);
	}
	Vector<Ref<Image>> slices;
	for (Ref<Image> slice : slice_list) {
		slices.push_back(slice);
	}
	cmsDeleteTransform(transform); // we don't need it after use
	RID tex_3d = RS::get_singleton()->texture_3d_create(Image::FORMAT_RGBH, dim, dim, dim, false, slices);
	return tex_3d;
}

ColorTransform::ColorTransform() {
	intent = CM_INTENT_PERCEPTUAL;
	use_bpc = true;
}

ColorTransform::ColorTransform(Ref<ColorProfile> p_src, Ref<ColorProfile> p_dst, Intent p_intent, bool p_black_point_compensation) {
	if (p_src.is_null() || !p_src->is_valid()) {
		ERR_PRINT("p_src is null or invalid.");
		return;
	}
	if (p_dst.is_null() || !p_dst->is_valid()) {
		ERR_PRINT("p_dst is null or invalid.");
		return;
	}

	src_profile = p_src;
	dst_profile = p_dst;
	intent = p_intent;
	use_bpc = p_black_point_compensation;
}

ColorTransform::~ColorTransform() {
}
