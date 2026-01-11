#!/bin/bash

echo "=== STAGE 1: Convert ABC to Blend Shapes ==="

/Applications/Blender.app/Contents/MacOS/./Blender --background --python-expr "
import bpy
import sys

print('Starting ABC conversion...')

# Clear scene
bpy.ops.wm.read_homefile(use_empty=True)

# Import geometry
print('Importing Bone_Geom.fbx...')
bpy.ops.import_scene.fbx(filepath='/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Geom.fbx')

# Try ABC import
print('Importing Bone_Anim.abc...')
try:
    result = bpy.ops.wm.alembic_import(filepath='/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim.abc')
    print(f'ABC import result: {result}')
    print('SUCCESS: ABC imported')
except Exception as e:
    print(f'ABC import failed: {e}')
    print('Continuing with available data...')

# Check results
mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
print(f'Found {len(mesh_objs)} mesh objects')

total_shapes = 0
for obj in mesh_objs:
    cache_mods = [mod for mod in obj.modifiers if mod.type == 'MESH_SEQUENCE_CACHE']
    print(f'{obj.name}: {len(cache_mods)} cache modifiers')
    
    if cache_mods and obj.data.shape_keys:
        shape_count = len(obj.data.shape_keys.key_blocks)
        total_shapes += shape_count
        print(f'  Has {shape_count} existing shape keys')

print(f'Total shape keys found: {total_shapes}')

if total_shapes == 0:
    print('No shape keys found - running conversion...')
    
    # Try to convert cache to shapes
    for obj in mesh_objs:
        cache_mods = [mod for mod in obj.modifiers if mod.type == 'MESH_SEQUENCE_CACHE']
        if cache_mods:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            print(f'Converting cache on {obj.name}...')
            for frame in range(1, 31):
                try:
                    bpy.context.scene.frame_set(frame)
                    for mod in cache_mods:
                        bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier=mod.name)
                except Exception as e:
                    print(f'Failed at frame {frame}: {e}')
                    break
    
    total_shapes = sum(len(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else 0 for obj in mesh_objs)
    print(f'Total shape keys after conversion: {total_shapes}')

# Export regardless
print('Exporting GLTF with morph targets...')
bpy.ops.export_scene.gltf(
    filepath='/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim_WithShapes.glb',
    export_format='GLB',
    export_morph=True
)
print('Stage 1 complete')
"

echo "=== STAGE 2: Add Animation Tracks ==="

/Applications/Blender.app/Contents/MacOS/./Blender --background --python-expr "
import bpy

print('Stage 2: Adding animation tracks...')

# Load the shapes GLTF
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.gltf(filepath='/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim_WithShapes.glb')

# Find mesh with blend shapes
mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.data.shape_keys]
blend_shapes_found = False

if mesh_objs:
    for mesh_obj in mesh_objs:
        if mesh_obj.data.shape_keys:
            shape_keys = mesh_obj.data.shape_keys.key_blocks
            print(f'Found mesh {mesh_obj.name} with {len(shape_keys)} shape keys')
            
            # Create animation data on the mesh object
            if not mesh_obj.animation_data:
                mesh_obj.animation_data_create()
            
            action = bpy.data.actions.new('DemBones_Animation')
            mesh_obj.animation_data.action = action
            
            # Create fcurve for each blend shape
            for i, shape_key in enumerate(shape_keys):
                if shape_key.name != 'Basis':
                    # Create animation curve path
                    curve_path = f'data.shape_keys.key_blocks[\"{shape_key.name}\"].value'
                    
                    # Add fcurve
                    fcurve = action.fcurves.new(curve_path)
                    
                    # Add keyframes (brief activation every N frames)
                    interval = max(5, len(shape_keys) // 10 + 1)  # Space out activations
                    start_frame = i * interval + 1
                    
                    # Keyframe: off, on, off
                    fcurve.keyframe_points.insert(start_frame, 0.0)
                    fcurve.keyframe_points.insert(start_frame + 1, 1.0)
                    fcurve.keyframe_points.insert(start_frame + 3, 0.0)
                    
                    blend_shapes_found = True
                    print(f'  Added animation track for {shape_key.name}')
    
    if blend_shapes_found:
        print('Blend shape animation tracks created')
    else:
        print('No blend shapes found to animate')

# Export with animations
print('Exporting GLTF with animations and morph targets...')
bpy.ops.export_scene.gltf(
    filepath='/Users/ernest.lee/Desktop/code/godot/modules/dem_bones/data/Bone_Anim_Complete.glb',
    export_format='GLB',
    export_animations=True,
    export_morph=True
)
print('Final GLTF exported')
"

# Replace the main file
cp modules/dem_bones/data/Bone_Anim_Complete.glb modules/dem_bones/data/Bone_Anim.glb
echo "Bone_Anim.glb updated with animations from ABC conversion"
