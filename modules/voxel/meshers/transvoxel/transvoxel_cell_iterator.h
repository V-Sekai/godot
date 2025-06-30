/**************************************************************************/
/*  transvoxel_cell_iterator.h                                            */
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

#include "../../engine/detail_rendering/detail_rendering.h"
#include "../../util/containers/std_vector.h"
#include "transvoxel.h"

namespace zylann::voxel {

// Implement the generic interface to iterate voxel mesh cells, which can be used to compute virtual textures.
// This one is optimized to gather results of the Transvoxel mesher, which comes with cell information out of the box.
class TransvoxelCellIterator : public ICellIterator {
public:
	TransvoxelCellIterator(Span<const transvoxel::CellInfo> p_cell_infos) :
			_current_index(0), _triangle_begin_index(0) {
		// Make a copy
		_cell_infos.resize(p_cell_infos.size());
		for (unsigned int i = 0; i < p_cell_infos.size(); ++i) {
			_cell_infos[i] = p_cell_infos[i];
		}
	}

	unsigned int get_count() const override {
		return _cell_infos.size();
	}

	bool next(CurrentCellInfo &current) override {
		if (_current_index < _cell_infos.size()) {
			const transvoxel::CellInfo &cell = _cell_infos[_current_index];
			current.position = cell.position;
			current.triangle_count = cell.triangle_count;

			for (unsigned int i = 0; i < cell.triangle_count; ++i) {
				current.triangle_begin_indices[i] = _triangle_begin_index;
				_triangle_begin_index += 3;
			}

			++_current_index;
			return true;

		} else {
			return false;
		}
	}

	void rewind() override {
		_current_index = 0;
		_triangle_begin_index = 0;
	}

private:
	StdVector<transvoxel::CellInfo> _cell_infos;
	unsigned int _current_index;
	unsigned int _triangle_begin_index;
};

} // namespace zylann::voxel
