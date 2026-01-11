#!/usr/bin/env python3
"""
ABC to OBJ Sequence Converter for DemBones
Converts ABC animation cache directly to OBJ sequence with proper frame timing.

Usage: blender --background --python convert_abc_to_obj_sequence.py

Output: OBJ files with frame and time data in filenames, e.g.:
- frame_001_time_0.000.obj
- frame_002_time_0.033.obj
- frame_003_time_0.067.obj
"""

import bpy
import os
import sys

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

def export_mesh_obj(verts, faces, filepath, mesh_name="Mesh"):
    """Export mesh to OBJ format manually - supports polygons of any size"""
    try:
        with open(filepath, 'w') as f:
            f.write(f"# Wavefront OBJ exported from ABC animation sequence\n")
            f.write(f"# Vertex count: {len(verts)}, Face count: {len(faces)}\n")
            f.write(f"o {mesh_name}\n")

            # Write vertices
            for vert in verts:
                f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

            # Write vertex normals (simple approximation - up direction)
            for vert in verts:
                normal = [0, 0, 1]  # Up direction as simple approximation
                f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

            # Write faces (handle polygons of any size)
            for face in faces:
                if len(face) >= 3:
                    # General polygon (supports triangles, quads, ngons)
                    face_indices = []
                    for vertex_idx in face:
                        vertex_idx_1_based = vertex_idx + 1  # OBJ uses 1-based indexing
                        face_indices.append(f"{vertex_idx_1_based}//{vertex_idx_1_based}")
                    f.write(f"f {' '.join(face_indices)}\n")

        return True
    except Exception as e:
        print(f"Failed to write OBJ file: {e}")
        return False

def main():
    """Convert ABC animation cache to OBJ sequence with frame timing"""
    data_dir = get_data_path()
    print(f"ABC to OBJ Sequence Converter")
    print(f"Using data directory: {data_dir}")

    # File paths
    abc_file = os.path.join(data_dir, 'Bone_Anim.abc')
    geom_file = os.path.join(data_dir, 'Bone_Geom.fbx')
    obj_sequence_dir = os.path.join(data_dir, 'obj_sequence_timed')

    print(f"ABC file: {abc_file}")
    print(f"Geometry: {geom_file}")
    print(f"Output directory: {obj_sequence_dir}")

    if not os.path.exists(abc_file):
        print(f"ERROR: ABC file not found: {abc_file}")
        exit(1)

    if not os.path.exists(geom_file):
        print(f"ERROR: Geometry FBX not found: {geom_file}")
        exit(1)

    try:
        # Clear scene
        bpy.ops.wm.read_homefile(use_empty=True)

        # Import geometry first
        print("\nImporting FBX geometry...")
        bpy.ops.import_scene.fbx(filepath=geom_file)
        print("✓ FBX geometry imported")

        # Import ABC animation
        print("\nImporting ABC animation...")
        result = bpy.ops.wm.alembic_import(filepath=abc_file, as_background_job=False)
        if result != {'FINISHED'}:
            print("✗ ABC import failed!")
            exit(1)
        print("✓ ABC animation imported")

        # Find mesh object with cache modifiers
        mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        obj_with_cache = None
        for obj in mesh_objs:
            cache_mods = [mod for mod in obj.modifiers if mod.type == 'MESH_SEQUENCE_CACHE']
            if cache_mods:
                obj_with_cache = obj
                break

        if not obj_with_cache:
            print("✗ No mesh object with cache modifiers found!")
            exit(1)

        print(f"✓ Found cache on object: {obj_with_cache.name}")

        # Convert cache to blend shapes first
        obj = obj_with_cache
        print(f"\nConverting cache to blend shapes for {obj.name}...")

        # Initialize shape keys if needed
        if not obj.data.shape_keys:
            bpy.ops.object.shape_key_add(from_mix=False)

        # Determine animation frame range
        # Check if we can get timing info from the ABC cache
        frame_rate = 30  # Default assumption
        frame_start = 1
        frame_end = 30   # Default range

        print(f"Frame range: {frame_start}-{frame_end} (frame rate: {frame_rate} fps)")

        # Create blend shapes for each frame
        shape_count = 0
        for frame in range(frame_start, frame_end + 1):
            # Set frame to capture deformation
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

            # Create blend shape from current deformed state
            bpy.ops.object.shape_key_add(from_mix=False)
            shape_key = obj.data.shape_keys.key_blocks[-1]
            shape_key.name = str(frame)

            # Copy vertex positions
            for i, vert in enumerate(obj.data.vertices):
                shape_key.data[i].co = vert.co

            shape_count += 1
            print(f"  Processed frame {frame} → blend shape '{frame}'")

        print(f"✓ Created {shape_count} blend shapes")

        # Create OBJ sequence with timing information
        print(f"\nGenerating OBJ sequence to: {obj_sequence_dir}")
        try:
            os.makedirs(obj_sequence_dir, exist_ok=True)
            print(f"✓ Created output directory: {obj_sequence_dir}")
        except Exception as e:
            print(f"✗ Failed to create output directory: {e}")
            exit(1)

        obj_count = 0
        for frame in range(frame_start, frame_end + 1):
            # Calculate time for this frame
            frame_time = (frame - frame_start) / frame_rate  # Time in seconds

            # Set frame and activate corresponding blend shape
            bpy.context.scene.frame_set(frame)
            shape_name = str(frame)

            # Activate the shape key for this frame
            for sk in obj.data.shape_keys.key_blocks:
                sk.value = 1.0 if sk.name == shape_name else 0.0
            bpy.context.view_layer.update()

            try:
                # Get the deformed mesh
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)
                eval_mesh = eval_obj.to_mesh()

                # Collect geometry
                verts = [(v.co.x, v.co.y, v.co.z) for v in eval_mesh.vertices]
                faces = [tuple(f.vertices) for f in eval_mesh.polygons]

                print(f"    Frame {frame}: {len(verts)} verts, {len(faces)} faces")

                # Export OBJ with frame and time data in filename
                obj_filename = f"frame_{frame:03d}_time_{frame_time:.3f}.obj"
                obj_path = os.path.join(obj_sequence_dir, obj_filename)

                success = export_mesh_obj(verts, faces, obj_path, f"Bone_Anim_Frame_{frame}")
                if success:
                    obj_count += 1
                    print(f"  Exported {obj_filename} ✓")
                    # Verify file was actually created
                    if os.path.exists(obj_path):
                        file_size = os.path.getsize(obj_path)
                        print(f"    ✓ File saved: {file_size} bytes")
                    else:
                        print(f"    ✗ File NOT saved: {obj_path}")
                else:
                    print(f"  ✗ Export failed for {obj_filename}")

                # Clean up
                eval_obj.to_mesh_clear()

            except Exception as e:
                print(f"  ✗ Error processing frame {frame}: {e}")

        print(f"\n✓ Conversion complete!")
        print(f"  Generated {obj_count} OBJ files with timing data")
        print(f"  Frame rate: {frame_rate} fps")
        print(f"  Time range: 0.000s to {(shape_count-1)/frame_rate:.3f}s")
        print(f"  Output directory: {obj_sequence_dir}")

        # Show sample filenames
        if obj_count > 0:
            print(f"\nSample filenames:")
            for i in range(min(3, obj_count)):
                frame_time = i / frame_rate
                print(f"  frame_{i+1:03d}_time_{frame_time:.3f}.obj")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
