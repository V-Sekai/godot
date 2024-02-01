/**************************************************************************/
/*  aabb_tree.cpp                                                         */
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

#include "aabb_tree.h"

void AABBTree::setup_tree(int dim, float fattening, int64_t nb_particles, bool touching_is_overlap) {
	bvh.removeAll();
	delete &bvh;
	bvh = aabb::Tree(dim, fattening, nb_particles, touching_is_overlap);
}

void AABBTree::insert_particle_at_position(int64_t index, PackedFloat32Array position, float radius) {
	auto begin = position.ptrw(), end = position.ptrw(); // We use the ptr as iterator.
	end = std::next(end, position.size());
	std::vector<float> pos(begin, end);
	bvh.insertParticle(index, pos, radius);
}

void AABBTree::insert_particle(int64_t index, PackedFloat32Array lowerbound, PackedFloat32Array upperbound) {
	auto begin = lowerbound.ptrw(), end = lowerbound.ptrw(); // We use the ptr as iterator.
	end = std::next(end, lowerbound.size());
	std::vector<float> lb(begin, end);

	begin = upperbound.ptrw();
	end = upperbound.ptrw(); // We use the ptr as iterator.
	end = std::next(end, upperbound.size());
	std::vector<float> ub(begin, end);

	bvh.insertParticle(index, lb, ub);
}

void AABBTree::remove_particle(int64_t index) {
	bvh.removeParticle(index);
}

void AABBTree::remove_all() {
	bvh.removeAll();
}

void AABBTree::update_particle_at_position(int64_t index, PackedFloat32Array position, float radius, bool always_reinsert) {
	auto begin = position.ptrw(), end = position.ptrw(); // We use the ptr as iterator.
	end = std::next(end, position.size());
	std::vector<float> pos(begin, end);

	bvh.updateParticle(index, pos, radius, always_reinsert);
}

void AABBTree::update_particle(int64_t index, PackedFloat32Array lowerbound, PackedFloat32Array upperbound, bool always_reinsert) {
	auto begin = lowerbound.ptrw(), end = lowerbound.ptrw(); // We use the ptr as iterator.
	end = std::next(end, lowerbound.size());
	std::vector<float> lb(begin, end);

	begin = upperbound.ptrw();
	end = upperbound.ptrw(); // We use the ptr as iterator.
	end = std::next(end, upperbound.size());
	std::vector<float> ub(begin, end);

	bvh.updateParticle(index, lb, ub, always_reinsert);
}

PackedInt64Array AABBTree::query_index(unsigned int i) {
	auto result = bvh.query(i);
	PackedInt64Array r;
	for (auto f : result) {
		r.push_back(f);
	}
	return r;
}

PackedInt64Array AABBTree::query_bounds(PackedFloat32Array lowerbound, PackedFloat32Array upperbound) {
	auto begin = lowerbound.ptrw(), end = lowerbound.ptrw(); // We use the ptr as iterator.
	end = std::next(end, lowerbound.size());
	std::vector<float> lb(begin, end);

	begin = upperbound.ptrw();
	end = upperbound.ptrw(); // We use the ptr as iterator.
	end = std::next(end, upperbound.size());
	std::vector<float> ub(begin, end);

	aabb::AABB bounds(lb, ub);
	auto result = bvh.query(bounds);
	PackedInt64Array r;
	for (auto f : result)
		r.push_back(f);
	return r;
}
