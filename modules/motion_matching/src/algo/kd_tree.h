/**************************************************************************/
/*  kd_tree.h                                                             */
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

#include "core/variant/variant.h"

#include <functional>
#include <vector>

class KDTree {
public:
	KDTree(const float *data, int point_dimension, int point_count);
	KDTree(int point_count);
	~KDTree();

	int search_nn(const float *pose_data, const float *query, const std::vector<float> &dimension_weigths) const;
	std::vector<int> search_knn(const float *pose_data,
			const float *query,
			const std::vector<float> &dimension_weigths,
			int k) const;

	PackedInt32Array get_node_indices() const;

	void rebuild_tree(int point_count, const PackedInt32Array &indices);

private:
	struct TreeNode {
		int axis;
		TreeNode *children[2];
		int index;
	};

	void _build_tree(const float *data, int point_dimension, int point_count);
	TreeNode *_build_tree_recursive(const float *data, int *indices, int npoints, int current_depth);

	void _clear_tree();
	void _clear_tree_recursive(TreeNode *node);

	void _search_nn_recursive(const float *pose_data, const float *query, const std::vector<float> &dimension_weigths, const TreeNode *node, float &best_distance, int &best_index, int depth) const;

	void _get_node_indices_recursive(const TreeNode *node, PackedInt32Array &indices) const;

	void _rebuild_tree_recursive(TreeNode *&node, const PackedInt32Array &indices, int &value_index, int current_depth);
	const int _point_dim = 0;
	TreeNode *_root = nullptr;
};
