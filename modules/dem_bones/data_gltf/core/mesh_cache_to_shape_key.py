# Copyright 2021 iFire#6518 and alexfreyre#1663
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This code ONLY applies to a mesh and simulations with the same vertex number.

import bpy

# Converts a MeshCache or Cloth modifiers to ShapeKeys
frame = bpy.context.scene.frame_start
for frame in range(bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_current = frame

    # for alembic files converted to PC2 and loaded as MeshCache
    bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier="MeshCache")

    # for cloth simulations inside blender using a Cloth modifier
    # bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier="Cloth")

# loop through shapekeys and add as keyframe per frame
# https://blender.stackexchange.com/q/149045/87258

frame = bpy.context.scene.frame_start
for frame in range(bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_current = frame

    for shapekey in bpy.data.shape_keys:
        for i, keyblock in enumerate(shapekey.key_blocks):
            if keyblock.name != "Basis":
                curr = i - 1
                if curr != frame:
                    keyblock.value = 0
                    keyblock.keyframe_insert("value", frame=frame)
                else:
                    keyblock.value = 1
                    keyblock.keyframe_insert("value", frame=frame)

# bpy.ops.object.modifier_remove(modifier="MeshCache")
# bpy.ops.object.modifier_remove(modifier="Cloth")
