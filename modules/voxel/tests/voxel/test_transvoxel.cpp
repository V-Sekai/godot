/**************************************************************************/
/*  test_transvoxel.cpp                                                   */
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

#include "test_transvoxel.h"
#include "../../meshers/transvoxel/voxel_mesher_transvoxel.h"
#include "../../util/testing/test_macros.h"

namespace zylann::voxel::tests {

void test_transvoxel_issue772() {
	// There was a wrong assertion check on the values of component indices when texturing mode is SINGLE_S4

	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_INDICES, VoxelBuffer::DEPTH_8_BIT);
	voxels.create(Vector3iUtil::create(8));
	{
		Vector3i pos;

		const float h = voxels.get_size().y / 2.f + 0.1f;

		for (pos.z = 0; pos.z < voxels.get_size().z; ++pos.z) {
			for (pos.x = 0; pos.x < voxels.get_size().x; ++pos.x) {
				for (pos.y = 0; pos.y < voxels.get_size().y; ++pos.y) {
					const float gy = pos.y;
					const float sd = gy - h;
					voxels.set_voxel_f(sd, pos, VoxelBuffer::CHANNEL_SDF);
					if (sd < 1.f) {
						const uint8_t material_index = (pos.x + pos.y + pos.z) & 0xff;
						voxels.set_voxel(material_index, pos, VoxelBuffer::CHANNEL_INDICES);
					}
				}
			}
		}
	}

	Ref<VoxelMesherTransvoxel> mesher;
	mesher.instantiate();
	mesher->set_texturing_mode(VoxelMesherTransvoxel::TEXTURES_SINGLE_S4);
	VoxelMesher::Output output;
	// Used to crash
	mesher->build(output, VoxelMesher::Input{ voxels, nullptr, Vector3i(), 0, false, false, false });

	ZN_TEST_ASSERT(!VoxelMesher::is_mesh_empty(output.surfaces));
}

} // namespace zylann::voxel::tests
