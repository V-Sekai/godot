/**************************************************************************/
/*  litrt_model.cpp                                                       */
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

#include "litrt_model.h"

#include "core/error/error_macros.h"
#include "core/io/file_access.h"

void LiteRtModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_from_file", "path"), &LiteRtModel::load_from_file);
	ClassDB::bind_method(D_METHOD("is_valid"), &LiteRtModel::is_valid);
	ClassDB::bind_method(D_METHOD("get_num_signatures"), &LiteRtModel::get_num_signatures);
}

LiteRtModel::LiteRtModel() {
}

LiteRtModel::~LiteRtModel() {
	if (model != nullptr) {
		LiteRtDestroyModel(model);
		model = nullptr;
	}
}

Error LiteRtModel::load_from_file(const String &p_path) {
	if (model != nullptr) {
		return ERR_ALREADY_EXISTS;
	}

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (file.is_null()) {
		return ERR_FILE_NOT_FOUND;
	}

	PackedByteArray data = file->get_buffer(file->get_length());
	file.unref();

	if (data.size() == 0) {
		return ERR_INVALID_DATA;
	}

	LiteRtStatus status = LiteRtCreateModelFromBuffer(data.ptr(), data.size(), &model);
	if (status != kLiteRtStatusOk) {
		model = nullptr;
		return FAILED;
	}

	return OK;
}

int LiteRtModel::get_num_signatures() const {
	if (model == nullptr) {
		return 0;
	}

	LiteRtParamIndex num_signatures = 0;
	LiteRtStatus status = LiteRtGetNumSignatures(model, &num_signatures);
	if (status != kLiteRtStatusOk) {
		return 0;
	}

	return static_cast<int>(num_signatures);
}

