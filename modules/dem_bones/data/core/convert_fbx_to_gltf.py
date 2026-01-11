#!/usr/bin/env python3
"""
FBX to GLTF Batch Converter for DemBones
Converts all FBX files in data directory to GLTF format.
Run with: blender --background --python convert_fbx_to_gltf.py
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
        if os.path.exists(data_dir) and 'data' in data_dir:
            return data_dir
        else:
            return os.getcwd()

    # Try to find dem bones relative to current working directory
    if os.path.exists('modules/dem_bones/data'):
        return 'modules/dem_bones/data'

    # Default fallback
    return '.'

def convert_fbx_to_gltf(fbx_path, gltf_path, export_morph=True, export_animations=True):
    """Convert a single FBX file to GLTF"""
    print(f"Converting: {os.path.basename(fbx_path)}")

    if not os.path.exists(fbx_path):
        print(f"  File not found: {fbx_path}")
        return False

    try:
        # Clear scene
        bpy.ops.wm.read_homefile(use_empty=True)

        # Import FBX
        print("  Importing FBX...")
        result = bpy.ops.import_scene.fbx(filepath=fbx_path)
        if 'FINISHED' not in str(result):
            print(f"  FBX import failed: {result}")
            return False

        print("  FBX imported successfully")
        # Count imported objects
        mesh_count = len([obj for obj in bpy.data.objects if obj.type == 'MESH'])
        total_objs = len(bpy.data.objects)
        print(f"  Imported {mesh_count} meshes, {total_objs} total objects")

        # Export GLTF
        print("  Exporting GLTF...")
        bpy.ops.export_scene.gltf(
            filepath=gltf_path,
            export_format='GLB',
            export_morph=export_morph,
            export_animations=export_animations,
            export_materials='NONE',  # Skip materials for vertex data only
            use_selection=False
        )

        if os.path.exists(gltf_path):
            file_size = os.path.getsize(gltf_path)
            print(f"  ✓ Exported {file_size:.1f} KB")
            return True
        else:
            print("  GLTF file not created!")
            return False

    except Exception as e:
        print(f"  Conversion failed: {e}")
        return False

def main():
    """Batch convert all FBX files to GLTF"""
    data_dir = get_data_path()
    print("FBX to GLTF Batch Converter for DemBones")
    print("=" * 40)
    print(f"Data directory: {data_dir}")

    # Find all FBX files
    fbx_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.lower().endswith('.fbx'):
                fbx_files.append(file)

    if not fbx_files:
        print("No FBX files found in data directory!")
        exit(1)

    print(f"Found {len(fbx_files)} FBX files:")
    for fbx in fbx_files:
        print(f"  • {fbx}")

    # Convert each FBX
    converted = 0
    skipped = 0

    for fbx_file in fbx_files:
        fbx_path = os.path.join(data_dir, fbx_file)
        gltf_file = fbx_file.replace('.fbx', '.glb')
        gltf_path = os.path.join(data_dir, gltf_file)

        # Skip if GLTF already exists (to avoid overwriting)
        if os.path.exists(gltf_path):
            print(f"Skipping {fbx_file} (GLTF exists)")
            skipped += 1
            continue

        if convert_fbx_to_gltf(fbx_path, gltf_path):
            converted += 1
        else:
            print(f"Failed to convert: {fbx_file}")

    print("\nConversion Summary:")
    print(f"  ✅ Converted: {converted}")
    print(f"  ⏭️ Skipped: {skipped}")
    print(f"  📝 Total FBX files: {len(fbx_files)}")

    if converted > 0:
        print("\nConverted files:")
        for fbx in fbx_files:
            gltf_name = fbx.replace('.fbx', '.glb')
            gltf_path = os.path.join(data_dir, gltf_name)
            if os.path.exists(gltf_path):
                size = os.path.getsize(gltf_path) / 1024
                print(f"    • {gltf_name} ({size:.1f} KB)")
    print("\n✓ FBX batch conversion completed!")

if __name__ == "__main__":
    main()
