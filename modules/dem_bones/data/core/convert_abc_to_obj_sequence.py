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

        # Import ABC animation and extract timing metadata
        print("\nImporting ABC animation...")
        result = bpy.ops.wm.alembic_import(filepath=abc_file, as_background_job=False)
        if result != {'FINISHED'}:
            print("✗ ABC import failed!")
            exit(1)
        print("✓ ABC animation imported")

        # Extract ABC timing information from imported objects and cache files
        abc_frame_start = None
        abc_frame_end = None
        abc_fps = None

        print("\nExtracting ABC timing metadata...")

        # Check bpy.data.cache_files for ABC metadata
        print(f"Global cache files: {len(bpy.data.cache_files)}")
        for cache_file in bpy.data.cache_files:
            print(f"  Cache file: {cache_file.name}")
            print(f"    Path: {cache_file.filepath}")
            print(f"    All properties: {[p for p in dir(cache_file) if not p.startswith('_')]}")

            # Try to get Alembic-specific properties
            try:
                if hasattr(cache_file, 'is_sequence') and cache_file.is_sequence:
                    print(f"    Is sequence: {cache_file.is_sequence}")

                    # Try to access Alembic time sampling
                    # Blender uses OpenALEMBIC internally
                    import bpy  # Already imported

                    # Check if we can get time range from the cache
                    print("    Attempting to access Alembic time sampling...")
                    # This might not work directly, but let's try

            except Exception as e:
                print(f"    Error accessing Alembic data: {e}")

        # Also check the imported objects
        print(f"Imported objects: {len(bpy.data.objects)}")
        for obj in bpy.data.objects:
            # Check for cache modifiers
            for mod in obj.modifiers:
                if mod.type == 'MESH_SEQUENCE_CACHE':
                    cache_file = mod.cache_file
                    if cache_file and hasattr(cache_file, 'filepath'):
                        # Try to access Alembic metadata
                        print(f"  Analyzing cache: {cache_file.name}")

                        # Check cache file properties for timing info
                        if hasattr(cache_file, 'frame_start'):
                            abc_frame_start = cache_file.frame_start
                            print(f"  Frame start: {abc_frame_start}")
                        if hasattr(cache_file, 'frame_end'):
                            abc_frame_end = cache_file.frame_end
                            print(f"  Frame end: {abc_frame_end}")

                        # Look for other timing properties
                        for prop_name in dir(cache_file):
                            if 'time' in prop_name.lower() or 'frame' in prop_name.lower():
                                if prop_name.startswith('__') or prop_name in ['filepath', 'name', 'users']:
                                    continue
                                try:
                                    value = getattr(cache_file, prop_name)
                                    if isinstance(value, (int, float)):
                                        print(f"  {prop_name}: {value}")
                                except:
                                    pass

                        # Also check the modifier properties
                        print(f"  Modifier properties available: {[p for p in dir(mod) if not p.startswith('_')]}")

                        if hasattr(mod, 'frame_start'):
                            print(f"  Modifier frame_start: {mod.frame_start}")
                        if hasattr(mod, 'frame_end'):
                            print(f"  Modifier frame_end: {mod.frame_end}")

        # Check ABC animation timing information
        print("\nAnalyzing ABC animation timing...")
        scene = bpy.context.scene

        # Check if there are any animation curves or keyframes
        anim_objects = [obj for obj in bpy.data.objects if obj.animation_data]
        print(f"Objects with animation data: {len(anim_objects)}")

        # Look for cache modifiers to understand timing
        mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        print(f"Mesh objects found: {len(mesh_objs)}")

        for obj in mesh_objs:
            cache_modifiers = [mod for mod in obj.modifiers if mod.type == 'MESH_SEQUENCE_CACHE']
            if cache_modifiers:
                print(f"  {obj.name} has {len(cache_modifiers)} cache modifier(s)")
                for mod in cache_modifiers:
                    print(f"    Cache: {mod.cache_file.name}")
                    # Try to print cache timing info if available
                    if hasattr(mod, 'cache_file'):
                        print(f"    Path: {mod.cache_file.filepath}")

                    # Try to access cache properties
                    cache_props = dir(mod)
                    print(f"    Cache properties available: {[p for p in cache_props if not p.startswith('_')]}")

                    # Check for frame information
                    if hasattr(mod, 'frame_start'):
                        print(f"    Frame start: {mod.frame_start}")
                    if hasattr(mod, 'frame_end'):
                        print(f"    Frame end: {mod.frame_end}")
                    if hasattr(mod, 'frame_offset'):
                        print(f"    Frame offset: {mod.frame_offset}")

                    # Check for time information
                    if hasattr(mod, 'time_mode'):
                        print(f"    Time mode: {mod.time_mode}")
                    if hasattr(mod, 'factor'):
                        print(f"    Time factor: {mod.factor}")

                    # Check the underlying cache file
                    cache_file = mod.cache_file
                    if cache_file:
                        print(f"    Cache file properties: {[p for p in dir(cache_file) if not p.startswith('_')]}")

                        # Try to get Alembic time sampling info
                        if hasattr(cache_file, 'frame_start'):
                            print(f"    Alembic frame start: {cache_file.frame_start}")
                        if hasattr(cache_file, 'frame_end'):
                            print(f"    Alembic frame end: {cache_file.frame_end}")
                        if hasattr(cache_file, 'frame_offset'):
                            print(f"    Alembic frame offset: {cache_file.frame_offset}")

        # Find mesh object with cache modifiers
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
        # Check if we can get timing info from the ABC cache or use better defaults
        frame_rate = 30  # Default assumption

        # Determine actual ABC animation length by testing frame ranges
        # ABC animations can be of variable length - find the real extent

        # Try longer ranges to find actual animation length
        extended_ranges = [
            (0, 59, "Frame 0-59 (60 frames)"),
            (1, 60, "Frame 1-60 (60 frames)"),
            (1, 120, "Frame 1-120 (120 frames)"),
            (1, 200, "Frame 1-200 (200 frames)"),
            (0, 100, "Frame 0-100 (100 frames)"),
            (1001, 1100, "Frame 1001-1100 (Maya range)"),
        ]

        print("Determining actual ABC animation length...")
        best_frame_range = None
        max_motion_detected = 0
        animation_end_frame = None

        # Test each range for motion to find where animation actually ends
        for frame_start, frame_end, description in extended_ranges:
            print(f"  Testing {description}...")

            try:
                motion_samples = []

                # Sample motion at regular intervals across the range
                sample_points = [frame_start + i * (frame_end - frame_start) // 4 for i in range(5)]
                if len(sample_points) == 5 and sample_points[-1] < frame_end:
                    sample_points.append(frame_end)

                base_vert = None
                for frame in sample_points:
                    if frame > 100:  # Skip very high ranges after testing basic ones
                        continue

                    bpy.context.scene.frame_set(frame)
                    bpy.context.view_layer.update()

                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    if len(eval_mesh.vertices) > 0:
                        curr_vert = eval_mesh.vertices[0].co.x
                        if base_vert is None:
                            base_vert = curr_vert
                        else:
                            motion_samples.append(abs(curr_vert - base_vert))

                    eval_obj.to_mesh_clear()

                range_motion = sum(motion_samples) if motion_samples else 0
                print(f"    Total motion: {range_motion:.6f}")

                if range_motion > max_motion_detected:
                    max_motion_detected = range_motion
                    best_frame_range = (frame_start, frame_end, description)

                # If this range has significant motion at the end, record it
                if range_motion > 0.5:  # Threshold for considering animation active
                    animation_end_frame = frame_end

            except Exception as e:
                print(f"    Error: {e}")
                continue

        # Also try to find the actual animation bounds by testing consecutive frames
        print("  Testing consecutive frame motion...")

        # Test frames 1-50 to find where motion stops
        motion_over_time = []
        for test_frame in range(1, 51, 5):  # Sample every 5 frames
            try:
                bpy.context.scene.frame_set(test_frame)
                bpy.context.view_layer.update()
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)
                eval_mesh = eval_obj.to_mesh()

                if len(eval_mesh.vertices) > 0:
                    curr_pos = (eval_mesh.vertices[0].co.x, eval_mesh.vertices[0].co.y, eval_mesh.vertices[0].co.z)
                    motion_over_time.append(curr_pos)

                eval_obj.to_mesh_clear()

            except Exception as e:
                print(f"    Error at frame {test_frame}: {e}")
                continue

        # Find where the animation stabilizes (motion stops)
        stability_threshold = 0.001
        stable_frame = None

        for i in range(1, len(motion_over_time)):
            if i < len(motion_over_time):
                prev = motion_over_time[i-1]
                curr = motion_over_time[i]
                frame_diff = sum((a-b)**2 for a,b in zip(prev, curr))**0.5

                if frame_diff < stability_threshold:
                    stable_frame = (i-1) * 5 + 1
                    break

        if best_frame_range and animation_end_frame:
            # Use the shorter of the two ranges to avoid oversampling
            frame_start, frame_end, description = best_frame_range

            if stable_frame and stable_frame < frame_end:
                # We found where animation stabilizes - use that as the endpoint
                if stable_frame > 1:  # Ensure at least some animation
                    frame_end = min(frame_end, stable_frame + 5)  # Add small buffer
                    description += f" (stabilized at ~frame {stable_frame})"

            print(f"✓ Detected ABC animation: {description}")
            print(f"  Actual range: {frame_start}-{frame_end} (motion score: {max_motion_detected:.6f})")

            if stable_frame:
                print(f"  Animation stable around frame: {stable_frame}")

        else:
            print("❌ Could not determine animation length - using conservative range")
            frame_start, frame_end = 1, 30

        print(f"Frame range: {frame_start}-{frame_end} (frame rate: {frame_rate} fps)")
        print(f"Time range: 0.000s to {(frame_end-frame_start)/frame_rate:.3f}s")

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
