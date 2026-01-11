#!/usr/bin/env python3
"""
ABC to USD Converter with Metadata Extraction for DemBones
Converts ABC animation cache to USD format for better introspection.

Usage: blender --background --python convert_abc_to_usd_and_extract.py

This creates USD files with full animation data, then extracts timing metadata.
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

def extract_usd_metadata(usd_file):
    """Extract animation metadata from USD file"""
    try:
        # Try to read USD file as ASCII text to extract metadata
        with open(usd_file, 'r') as f:
            lines = f.readlines()

        print(f"  Reading USD file ({len(lines)} lines)...")

        # Look for common USD metadata
        time_range = None
        frame_rate = None
        start_time = None
        end_time = None
        mesh_data = {}

        for line in lines:
            line = line.strip()

            # Look for time-related metadata
            if 'endTimeCode' in line or 'timeSamples' in line:
                print(f"    Found timing data: {line}")

            if 'timeSamples' in line:
                # Extract time samples information
                print(f"    Time samples: {line}")

            # Look for mesh/geometry data
            if 'Mesh' in line or 'Xform' in line:
                if '/*Bone/geoms/' in line:
                    mesh_data[line] = mesh_data.get(line, 0) + 1

        print(f"  Mesh objects found: {len(mesh_data)}")
        for mesh, count in mesh_data.items():
            print(f"    {mesh}: {count} references")

        return {
            'lines': len(lines),
            'meshes': len(mesh_data),
            'mesh_names': list(mesh_data.keys())
        }

    except Exception as e:
        print(f"  ✗ Error reading USD file: {e}")
        return None

def main():
    """Convert ABC to USD and extract metadata"""
    data_dir = get_data_path()
    print(f"ABC to USD Converter with Metadata Extraction")
    print(f"Using data directory: {data_dir}")

    # File paths
    abc_file = os.path.join(data_dir, 'Bone_Anim.abc')
    geom_file = os.path.join(data_dir, 'Bone_Geom.fbx')
    usd_file = os.path.join(data_dir, 'Bone_Anim.usda')  # ASCII USD format for inspection

    print(f"ABC file: {abc_file}")
    print(f"Geometry: {geom_file}")
    print(f"Output USD: {usd_file}")

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

        # Export to USD format (ASCII for metadata inspection)
        print(f"\nExporting to USD format: {usd_file}")

        # Find the mesh object with animation cache
        mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        cache_obj = None

        for obj in mesh_objs:
            if any(mod.type == 'MESH_SEQUENCE_CACHE' for mod in obj.modifiers):
                cache_obj = obj
                break

        if not cache_obj:
            print("✗ No mesh object with cache found!")
            exit(1)

        print(f"✓ Found cache object: {cache_obj.name}")

        # Select the object for export
        bpy.ops.object.select_all(action='DESELECT')
        cache_obj.select_set(True)
        bpy.context.view_layer.objects.active = cache_obj

        # Export to USD with animation
        try:
            bpy.ops.export_scene.usd(
                filepath=usd_file,
                selected_objects_only=True,
                export_animation=True,
                start_time=1,
                end_time=30,  # Initial range, will be adjusted based on metadata
                frame_step=1,
                export_materials=False,
                export_meshes=True,
                use_usd_preview_surface=True,
                export_textures=False
            )

            if os.path.exists(usd_file):
                file_size = os.path.getsize(usd_file)
                print(f"✓ USD exported successfully ({file_size} bytes)")
            else:
                print("✗ USD export failed - file not created")
                exit(1)

        except Exception as e:
            print(f"✗ USD export failed: {e}")
            exit(1)

        # Extract metadata from USD file
        print(f"\nExtracting metadata from USD file...")
        usd_metadata = extract_usd_metadata(usd_file)

        if usd_metadata:
            print("✓ USD metadata extraction complete")
            print(f"  Lines in USD file: {usd_metadata['lines']}")
            print(f"  Mesh objects: {usd_metadata['meshes']}")
        else:
            print("⚠️  USD metadata extraction incomplete")

        print(f"\n🎯 Summary:")
        print(f"  ABC file: {os.path.getsize(abc_file)} bytes")
        print(f"  USD file: {os.path.getsize(usd_file)} bytes")
        print(f"  Converted from binary Alemkozonic to inspectable USD format")

        print("✓ ABC → USD conversion completed successfully!")
        print(f"  Check {usd_file} for full animation metadata")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
