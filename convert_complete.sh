#!/bin/bash
cd /Applications/Blender.app/Contents/MacOS
echo "Converting Bone_Geom.fbx + Bone_Anim.abc into single GLTF with blend shapes..."
./Blender --background --python-expr "
import bpy
import sys

# Clear scene  
bpy.ops.wm.read_homefile(use_empty=True)

# Import geometry first
print(\"Importing Bone_Geom.fbx...\")
bpy.ops.import_scene.fbx(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Geom.fbx\")

# Import animations (creates mesh cache)
print(\"Importing Bone_Anim.abc...\")  
try:
    bpy.ops.wm.alembic_import(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim.abc\", set_frame_range=True, is_sequencer=False, as_background_job=False)
    print(\"ABC imported successfully\")
except Exception as e:
    print(f\"ABC import failed: {e}\")
    sys.exit(1)

# Check if we have mesh cache modifiers
mesh_objs = [obj for obj in bpy.data.objects if obj.type == \"MESH\"]
if not mesh_objs:
    print(\"No mesh objects found!\")
    sys.exit(1)

# Select the mesh and convert cache to shape keys
mesh_obj = mesh_objs[0]
bpy.context.view_layer.objects.active = mesh_obj
mesh_obj.select_set(True)

print(f\"Converting mesh cache on {mesh_obj.name} to shape keys...\")

# Apply the conversion script logic inline
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 10  # Adjust based on animation length

# Convert mesh cache to shape keys frame by frame
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    
    # Apply mesh cache as shape key
    if mesh_obj.modifiers:
        for mod in mesh_obj.modifiers:
            if mod.type == \"MESH_SEQUENCE_CACHE\":
                bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier=mod.name)
                break

# Set up shape key animation (one shape key active per frame)
frame = bpy.context.scene.frame_start
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    
    for shapekey in bpy.data.shape_keys:
        for i, keyblock in enumerate(shapekey.key_blocks):
            if keyblock.name != \"Basis\":
                curr = i - 1
                if curr != frame:
                    keyblock.value = 0
                    if not keyblock.animation_data:
                        keyblock.keyframe_insert(\"value\", frame=frame)
                else:
                    keyblock.value = 1
                    if not keyblock.animation_data:
                        keyblock.keyframe_insert(\"value\", frame=frame)

print(\"Conversion complete, exporting GLTF...\")
bpy.ops.export_scene.gltf(filepath=\"/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data_gltf/Bone_Geom_Anim.glb\", export_format=\"GLB\", export_morph=True, export_animations=True)

print(\"GLTF export complete\")
"
echo "Conversion completed successfully!"
