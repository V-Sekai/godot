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

# Convert ABC vertex cache frames directly to blend shapes
import bpy

print("Converting ABC vertex cache frames to blend shapes...")

# Get all mesh objects
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

if not mesh_objects:
    print("No mesh objects found!")
    exit(1)

for obj in mesh_objects:
    print(f"Processing: {obj.name}")

    # Find mesh sequence cache modifiers (from ABC import)
    cache_modifiers = [mod for mod in obj.modifiers if mod.type == 'MESH_SEQUENCE_CACHE']

    if not cache_modifiers:
        print("  No mesh sequence cache modifiers found")
        continue

    print(f"  Found {len(cache_modifiers)} cache modifiers")

    # Initialize shape keys if not exists
    if not obj.data.shape_keys:
        bpy.ops.object.shape_key_add(from_mix=False)
        print("  Created Basis shape key")

    # Determine frame range - try to get from cache modifier info
    frame_start = 1
    frame_end = 30  # Default range

    print(f"  Sampling frames {frame_start} to {frame_end}...")

    shape_count = 0
    successful_frames = []

    for frame in range(frame_start, frame_end + 1):
        try:
            # Set current frame to activate cache deformation
            bpy.context.scene.frame_set(frame)

            # Force dependency graph update to apply cache deformation
            bpy.context.view_layer.update()
            obj.data.update()

            # Create new blend shape from current deformed positions
            bpy.ops.object.shape_key_add(from_mix=False)
            shape_key = obj.data.shape_keys.key_blocks[-1]
            shape_key.name = str(frame)  # Use frame number as name (e.g. "1", "2", "3")

            # Copy current vertex positions to the shape key
            # Since vertex ordering is preserved in ABC cache, this creates valid morph targets
            for i, vert in enumerate(obj.data.vertices):
                shape_key.data[i].co = vert.co

            print(f"  ✓ Created blend shape for frame {frame}")
            shape_count += 1
            successful_frames.append(frame)

        except Exception as e:
            print(f"  ✗ Failed to create shape for frame {frame}: {e}")
            break  # Stop if we hit errors

    print(f"  Created {shape_count} blend shapes from ABC cache frames")
    print(f"  Blender shapes: {successful_frames}")

    # Blend shape animations optional - the static shapes are what's important for DemBones
    # Skip animation keying for now to ensure GLTF export works
    if shape_count > 0:
        print("  Blend shapes ready for DemBones (animations optional for export)")
        print(f"  {shape_count} blend shapes contain ABC vertex deformation data")

print(f"ABC cache to blend shapes conversion complete!")
print("Each blend shape corresponds to vertex positions from that ABC frame.")

print(f"Generating OBJ sequence for verification...")

# Also export OBJ files for each frame for verification
for obj in mesh_objects:
    if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1:
        # Create obj_exports directory if it doesn't exist
        import os
        obj_dir = '/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/obj_frames'
        os.makedirs(obj_dir, exist_ok=True)

        # Export each frame as OBJ
        total_exported = 0
        for shape_key in obj.data.shape_keys.key_blocks[1:]:  # Skip Basis
            try:
                # Set shape key value to 1 to see the deformation
                shape_key.value = 1.0
                bpy.context.view_layer.update()

                # Export OBJ for this frame
                obj_path = f"{obj_dir}/frame_{shape_key.name}.obj"
                bpy.ops.export_scene.obj(
                    filepath=obj_path,
                    check_existing=False,
                    use_selection=True,
                    use_materials=False,
                    use_mesh_modifiers=True
                )

                # Reset shape key
                shape_key.value = 0.0

                print(f"  Exported {obj_path}")
                total_exported += 1

            except Exception as e:
                print(f"  Failed to export frame {shape_key.name}: {e}")
                shape_key.value = 0.0

        print(f"OBJ sequence export complete: {total_exported} frames saved to obj_frames/")

# Keep the modifiers for now (uncomment to remove after export)
# for obj in mesh_objects:
#     for mod in obj.modifiers:
#         if mod.type == 'MESH_SEQUENCE_CACHE':
#             obj.modifiers.remove(mod)
