#!/bin/bash
cd /Applications/Blender.app/Contents/MacOS

# Convert all remaining FBX files to GLTF
FILES=(
"Bone_Geom.fbx"
"Bone_Helpers.fbx" 
"Bone_PartiallySkinned.fbx"
"Bone_Skin.fbx"
"Bone_Trans.fbx"
"Decomposition_05.fbx"
"Decomposition_10.fbx"
"Decomposition_20.fbx"
"Decomposition_20_grouped.fbx"
"Optimized.fbx"
"SolvedHelpers.fbx"
"SolvedPartialWeights.fbx"
"SolvedTransformations.fbx"
"SolvedWeights.fbx"
)

for file in "${FILES[@]}"; do
    base=$(basename "$file" .fbx)
    if [ ! -f "/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/${base}.glb" ]; then
        echo "Converting $file..."
        ./Blender --background --python-expr "
import bpy
try:
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/$file\")
    bpy.ops.export_scene.gltf(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/${base}.glb\", export_format=\"GLB\")
    print(f\"Converted $file\")
except Exception as e:
    print(f\"Failed to convert $file: {e}\")
"
    else
        echo "Skipping $file (already exists)"
    fi
done

echo "Batch conversion completed!"
