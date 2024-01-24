/*************************************************************************/
/*  image_loader_ies.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "image_loader_ies.h"

#include "core/io/file_access.h"
#include "core/io/image.h"

#include "thirdparty/tinyies/tiny_ies.hpp"

Error ImageLoaderIES::load_image(Ref<Image> p_image, Ref<FileAccess> p_fileaccess, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	ERR_FAIL_COND_V(p_fileaccess.is_null(), ERR_INVALID_PARAMETER);
	tiny_ies<float>::light ies;
	std::string ies_file = p_fileaccess->get_as_utf8_string(true).utf8().get_data();
	std::string err_out, warn_out;
	if (!tiny_ies<float>::load_ies_from_buffer(ies_file, err_out, warn_out, ies)) {
		ERR_PRINT(vformat("IES image loader error: %s", err_out.c_str()));
		return FAILED;
	}
	ERR_FAIL_COND_V(ies.number_vertical_angles <= 1, ERR_INVALID_DATA);
	ERR_FAIL_COND_V(ies.number_horizontal_angles <= 1, ERR_INVALID_DATA);
	Vector<Vector<float>> texture_data;
	texture_data.resize(ies.number_vertical_angles);
	for (int i = 0; i < ies.number_vertical_angles; ++i) {
		Vector<float> inner_vector;
		inner_vector.resize(ies.number_horizontal_angles);
		for (int j = 0; j < ies.number_horizontal_angles; ++j) {
			inner_vector.write[j] = ies.candela[i * ies.number_horizontal_angles + j];
		}
		texture_data.write[i] = inner_vector;
	}

	int number_of_vertical_angles = ies.number_vertical_angles;
	int number_of_horizontal_angles = ies.number_horizontal_angles;

	float target_horizontal = 360.0f;
	float target_vertical = 180.0f;
	float min_horizontal_angle = ies.min_horizontal_angle;
	float max_horizontal_angle = ies.max_horizontal_angle;
	float min_vertical_angle = ies.min_vertical_angle;
	float max_vertical_angle = ies.max_vertical_angle;
	Ref<Image> image = Image::create_empty(target_horizontal, target_vertical - min_vertical_angle, false, Image::FORMAT_RGBF);
	image->fill(Color());
	for (int i = 0; i < image->get_height(); ++i) {
		for (int j = 0; j < image->get_width(); ++j) {
			float original_i = static_cast<float>(i) * (max_vertical_angle - min_vertical_angle) / image->get_height() + min_vertical_angle;
			float original_j = static_cast<float>(j) * (max_horizontal_angle - min_horizontal_angle) / image->get_width() + min_horizontal_angle;
			float new_i = (original_i - min_vertical_angle) / (max_vertical_angle - min_vertical_angle) * (target_vertical - min_vertical_angle);
			float new_j = (original_j - min_horizontal_angle) / (max_horizontal_angle - min_horizontal_angle) * (target_horizontal - min_horizontal_angle);

			if (new_i >= min_vertical_angle && new_i <= max_vertical_angle && new_j >= min_horizontal_angle && new_j <= max_horizontal_angle) {
				int tex_i = static_cast<int>(original_i * number_of_vertical_angles / (max_vertical_angle - min_vertical_angle));
				float normalized_angle = (original_j - min_horizontal_angle) / (max_horizontal_angle - min_horizontal_angle);
				float index_f = normalized_angle * number_of_horizontal_angles;

				int index_pre = CLAMP(static_cast<int>(index_f) - 1, 0, number_of_horizontal_angles - 2);
				int index_from = CLAMP(static_cast<int>(index_f), 0, number_of_horizontal_angles - 2);
				int index_to = CLAMP(index_from + 1, 0, number_of_horizontal_angles - 2);
				int index_post = CLAMP(index_to + 1, 0, number_of_horizontal_angles - 2);

				float intensity_pre = texture_data[tex_i][index_pre];
				float intensity_from = texture_data[tex_i][index_from];
				float intensity_to = texture_data[tex_i][index_to];
				float intensity_post = texture_data[tex_i][index_post];
				float intensity_j = Math::cubic_interpolate(intensity_pre, intensity_from, intensity_to, intensity_post, index_f - index_from) / ies.max_candela;

				float normalized_i = (original_i - min_vertical_angle) / (max_vertical_angle - min_vertical_angle);
				float index_f_i = normalized_i * number_of_vertical_angles;

				int index_pre_i = CLAMP(static_cast<int>(index_f_i) - 1, 0, number_of_vertical_angles - 2);
				int index_from_i = CLAMP(static_cast<int>(index_f_i), 0, number_of_vertical_angles - 2);
				int index_to_i = CLAMP(index_from_i + 1, 0, number_of_vertical_angles - 2);
				int index_post_i = CLAMP(index_to_i + 1, 0, number_of_vertical_angles - 2);

				float intensity_pre_i = texture_data[index_pre_i][tex_i];
				float intensity_from_i = texture_data[index_from_i][tex_i];
				float intensity_to_i = texture_data[index_to_i][tex_i];
				float intensity_post_i = texture_data[index_post_i][tex_i];

				float intensity_i = Math::cubic_interpolate(intensity_pre_i, intensity_from_i, intensity_to_i, intensity_post_i, index_f_i - index_from_i) / ies.max_candela;

				float final_intensity = (intensity_j + intensity_i) / 2;
				image->set_pixel(j, i, Color(final_intensity, final_intensity, final_intensity));
			}
		}
	}
	p_image->set_data(image->get_width(), image->get_height(), false, Image::FORMAT_RGBF, image->get_data());
	p_image->generate_mipmaps();
	return OK;
}

void ImageLoaderIES::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ies");
}

ImageLoaderIES::ImageLoaderIES() {
}
