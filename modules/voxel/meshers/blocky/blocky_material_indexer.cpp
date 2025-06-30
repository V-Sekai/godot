/**************************************************************************/
/*  blocky_material_indexer.cpp                                           */
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

#include "blocky_material_indexer.h"
#include "../../util/godot/classes/material.h"
#include "../../util/string/format.h"
#include "blocky_baked_library.h"

namespace zylann::voxel::blocky {

unsigned int MaterialIndexer::get_or_create_index(const Ref<Material> &p_material) {
	for (size_t i = 0; i < materials.size(); ++i) {
		const Ref<Material> &material = materials[i];
		if (material == p_material) {
			return i;
		}
	}
#ifdef TOOLS_ENABLED
	if (materials.size() == MAX_MATERIALS) {
		ZN_PRINT_ERROR(
				format("Maximum material count reached ({}), try reduce your number of materials by reusing "
					   "them or using atlases.",
					   MAX_MATERIALS)
		);
	}
#endif
	const unsigned int ret = materials.size();
	materials.push_back(p_material);
	return ret;
}

} // namespace zylann::voxel::blocky
