#!/bin/bash
cd /Applications/Blender.app/Contents/MacOS
echo "Converting Bone_Geom.fbx..."
./Blender --background --python-expr "
import bpy
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.fbx(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Geom.fbx\")
bpy.ops.export_scene.gltf(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Geom2.glb\", export_format=\"GLB\")
"
echo "Converting Bone_Anim.abc..."
./Blender --background --python-expr "
import bpy
bpy.ops.wm.read_homefile(use_empty=True)
try:
    bpy.ops.wm.alembic_import(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim.abc\", is_sequencer=False)
    bpy.ops.export_scene.gltf(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim2.glb\", export_format=\"GLB\")
except:
    print(\"ABC conversion failed\")
"
echo "Conversions complete"
