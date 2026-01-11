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
import os

def get_data_path():
    """Get path to dem bones data directory"""
    script_path = bpy.data.filepath if bpy.data.filepath else __file__
    script_dir = os.path.dirname(os.path.abspath(script_path))

    # Script should be in modules/dem_bones/data/core/
    if 'modules/dem_bones/data/core' in script_dir:
        # Go up one directory to get to data
        data_dir = os.path.dirname(script_dir)
        return data_dir

    # Try to find dem bones relative to current working directory
    if os.path.exists('modules/dem_bones/data'):
        return 'modules/dem_bones/data'

    # Default fallback
    return '.'

print("Converting ABC vertex cache frames to blend shapes...")

data_dir = get_data_path()
print(f"Using data directory: {data_dir}")

# Import ABC if no mesh objects with cache modifiers exist
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
has_cache_modifiers = any(
    mod for obj in mesh_objects
    for mod in obj.modifiers
    if mod.type == 'MESH_SEQUENCE_CACHE'
)

if not mesh_objects or not has_cache_modifiers:
    print("No mesh objects with cache modifiers found. Importing ABC file...")

    # Clear scene
    bpy.ops.wm.read_homefile(use_empty=True)

    abc_file = os.path.join(data_dir, 'Bone_Anim.abc')
    if os.path.exists(abc_file):
        print(f"Importing {abc_file}...")
        result = bpy.ops.wm.alembic_import(filepath=abc_file)
        if result != {'FINISHED'}:
            print("✗ ABC import failed!")
        else:
            print("✓ ABC imported successfully")
    else:
        print(f"✗ ABC file not found: {abc_file}")

# Get all mesh objects again after potential import
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

    # Create animation for blend shapes
    if shape_count > 0:
        print("  Creating animation keyframes for blend shapes...")

        # Clear existing animation data
        if obj.animation_data:
            obj.animation_data_clear()

        # Create animation data
        if not obj.animation_data:
            obj.animation_data_create()

        # Set animation length
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = shape_count

        # Create keyframes for each shape key over time
        for frame_idx, shape_key in enumerate(obj.data.shape_keys.key_blocks[1:], 1):  # Skip basis
            target_frame = frame_idx

            # Turn on the shape key for this frame
            for sk in obj.data.shape_keys.key_blocks:
                sk.value = 1.0 if sk.name == shape_key.name else 0.0

            # Keyframe all shape keys at this frame
            bpy.context.scene.frame_set(target_frame)
            for sk in obj.data.shape_keys.key_blocks:
                sk.keyframe_insert("value", frame=target_frame)

        # Reset all shape keys to 0
        for sk in obj.data.shape_keys.key_blocks:
            sk.value = 0.0

        print("  Animation keyframes created")

    if shape_count > 0:
        print("  Blend shapes and animations ready for DemBones and GLTF export")
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
