#!/usr/bin/env python3
"""
ABC to GLTF Blend Shape Converter for DemBones
Run with: blender --background --python convert_abc_to_gltf.py -- /path/to/godot/project
"""

import bpy
import os
import sys

def get_data_path():
    """Get path to dem bones data directory based on script location"""
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
    """Export mesh to OBJ format manually"""
    try:
        with open(filepath, 'w') as f:
            f.write(f"# Wavefront OBJ file exported from Blender\n")
            f.write(f"# Vertex count: {len(verts)}\n")
            f.write(f"# Face count: {len(faces)}\n")
            f.write(f"o {mesh_name}\n")

            # Write vertices
            for vert in verts:
                f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

            # Write vertex normals (simple approximation)
            for vert in verts:
                # Simple normal calculation (could be improved)
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
    """Main conversion function"""
    data_dir = get_data_path()
    print(f"Using data directory: {data_dir}")

    # File paths
    abc_file = os.path.join(data_dir, 'Bone_Anim.abc')
    geom_file = os.path.join(data_dir, 'Bone_Geom.fbx')
    output_file = os.path.join(data_dir, 'Bone_Anim_Converted.glb')
    obj_sequence_dir = os.path.join(data_dir, 'obj_sequence')

    print("ABC to GLTF Blend Shape Converter")
    print("=" * 35)
    print(f"ABC: {abc_file}")
    print(f"Geometry: {geom_file}")
    print(f"Output: {output_file}")

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
        print("✓ Geometry imported")

        # Import ABC animation
        print("\nImporting ABC animation...")
        result = bpy.ops.wm.alembic_import(filepath=abc_file)
        if result != {'FINISHED'}:
            print("✗ ABC import failed!")
            exit(1)
        print("✓ ABC imported successfully")

        # Check for cache modifiers
        mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        print(f"\nFound {len(mesh_objs)} mesh objects")

        obj_with_cache = None
        for obj in mesh_objs:
            cache_mods = [mod for mod in obj.modifiers if mod.type == 'MESH_SEQUENCE_CACHE']
            if cache_mods:
                obj_with_cache = obj
                print(f"✓ Found mesh cache on {obj.name} ({len(cache_mods)} modifiers)")
                break

        if not obj_with_cache:
            print("✗ No mesh sequence cache modifiers found!")
            exit(1)

        # Convert cache to blend shapes
        obj = obj_with_cache
        print(f"\nConverting cache to blend shapes for {obj.name}...")

        # Initialize shape keys
        if not obj.data.shape_keys:
            bpy.ops.object.shape_key_add(from_mix=False)
            print("✓ Initialized shape keys")

        # Sample frames 1-30
        shape_count = 0
        for frame in range(1, 31):
            try:
                # Set frame and update
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()
                obj.data.update()

                # Create blend shape
                bpy.ops.object.shape_key_add(from_mix=False)
                shape_key = obj.data.shape_keys.key_blocks[-1]
                shape_key.name = str(frame)

                # Copy vertex positions
                for i, vert in enumerate(obj.data.vertices):
                    shape_key.data[i].co = vert.co

                shape_count += 1
                print(f"  Frame {frame} → Blend shape '{frame}' ✓")

            except Exception as e:
                print(f"  ✗ Error at frame {frame}: {e}")
                break

        print(f"\n✓ Created {shape_count} blend shapes from ABC cache")

        # Create animation for blend shapes
        print(f"\nCreating animation for {shape_count} blend shapes...")

        # Set animation length
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = shape_count

        # Create animation for blend shapes - create a simple stepwise animation
        # Instead of complex keyframes, create a basic animation that cycles through shapes
        print("Creating basic blend shape animation...")

        # Create an AnimationPlayer-type animation by keyframing each shape key to be 1.0 at its frame
        # Reset all keyframes first
        if obj.animation_data:
            obj.animation_data_clear()

        # Create animation data
        if not obj.animation_data:
            obj.animation_data_create()

        # For a simple animation, animate all shape keys over time with basic on/off
        for frame in range(1, shape_count + 1):
            bpy.context.scene.frame_set(frame)

            # Turn on the shape key for this frame
            shape_name = str(frame)
            if shape_name in obj.data.shape_keys.key_blocks:
                for sk in obj.data.shape_keys.key_blocks:
                    sk.value = 1.0 if sk.name == shape_name else 0.0

                # Keyframe all shape keys at this frame
                for sk in obj.data.shape_keys.key_blocks:
                    sk.keyframe_insert("value", frame=frame)

        # Reset to frame 1
        bpy.context.scene.frame_set(1)
        for sk in obj.data.shape_keys.key_blocks:
            sk.value = 0.0

        # Reset shape key values to 0 (so morph doesn't show at rest pose)
        for shape_key in obj.data.shape_keys.key_blocks[1:]:
            shape_key.value = 0.0

        print(f"✓ Animation created with {shape_count} keyframes")

        # Export OBJ sequence for validation
        print(f"\nExporting OBJ sequence to: {obj_sequence_dir}")
        os.makedirs(obj_sequence_dir, exist_ok=True)

        obj_count = 0
        for frame in range(1, shape_count + 1):
            # Set frame to get the deformed geometry
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

            # Use blend shape for clean deformation (set value to 1.0)
            shape_key_name = str(frame)
            for sk in obj.data.shape_keys.key_blocks:
                sk.value = 1.0 if sk.name == shape_key_name else 0.0

            bpy.context.view_layer.update()

            # Export OBJ for this frame manually (since Blender OBJ exporter isn't available in background mode)
            obj_path = os.path.join(obj_sequence_dir, f"frame_{frame:03d}.obj")
            try:
                # Get current evaluator mesh for export
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)
                eval_mesh = eval_obj.to_mesh()

                # Collect vertices and faces from current deformation
                verts = [(v.co.x, v.co.y, v.co.z) for v in eval_mesh.vertices]
                faces = [tuple(f.vertices) for f in eval_mesh.polygons]


                # Export manually
                success = export_mesh_obj(verts, faces, obj_path, f"frame_{frame:03d}")
                if success:
                    obj_count += 1
                    print(f"  Exported frame_{frame:03d}.obj ✓")

                # Clean up temporary mesh
                eval_obj.to_mesh_clear()

            except Exception as e:
                print(f"  ✗ Failed to export frame {frame}: {e}")
                break

        print(f"✓ Exported {obj_count} OBJ files to obj_sequence/")

        # Export GLTF with blend shapes
        print("Exporting GLTF with blend shapes...")
        bpy.ops.export_scene.gltf(
            filepath=output_file,
            export_format='GLB',
            export_morph=True,
            export_animations=False
        )

        # Export GLTF with animations (optional)
        anim_output_file = output_file.replace('.glb', '_with_anim.glb')
        try:
            print("Exporting GLTF with blend shapes and animations...")
            bpy.ops.export_scene.gltf(
                filepath=anim_output_file,
                export_format='GLB',
                export_morph=True,
                export_animations=True  # Include any animation curves
            )
            if os.path.exists(anim_output_file):
                anim_size = os.path.getsize(anim_output_file) / 1024 / 1024
                print(f"✓ Animated GLTF exported ({anim_size:.1f} MB)")
        except Exception as e:
            print(f"Animated GLTF export skipped: {e}")

        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024 / 1024
            print(f"✓ GLTF with blend shapes exported successfully ({file_size:.1f} MB)")
            print("✓ Conversion completed successfully!")
            print(f"  • GLTF with blend shapes: {output_file}")
            print(f"  • Animated GLTF: {anim_output_file}")
            print(f"  • OBJ sequence directory: {obj_sequence_dir}")
        else:
            print("✗ GLTF export failed - file not created")
            exit(1)

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
