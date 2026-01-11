#!/bin/bash
cd /Applications/Blender.app/Contents/MacOS
./Blender --background --python-expr "
import bpy
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.fbx(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_All.fbx\")
bpy.ops.export_scene.gltf(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_All.glb\", export_format=\"GLB\", export_morph=True)
print(\"Conversion complete\")
"
