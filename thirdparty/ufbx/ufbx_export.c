#include "ufbx_export.h"
#include "ufbx.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

static bool ufbx_export_grow_array(void **ptr, size_t *cap, size_t elem_size, size_t min_cap) {
    if (*cap >= min_cap) {
        return true;
    }
    
    size_t new_cap = *cap ? *cap * 2 : 16;
    if (new_cap < min_cap) {
        new_cap = min_cap;
    }
    
    void *new_ptr = realloc(*ptr, new_cap * elem_size);
    if (!new_ptr) {
        return false;
    }
    
    *ptr = new_ptr;
    *cap = new_cap;
    return true;
}

// Animation export functions

// Add an animation stack to the scene
ufbx_anim_stack *ufbx_add_animation(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->anim_stacks, &scene_imp->anim_stacks_cap,
                                sizeof(ufbx_anim_stack*), scene_imp->num_anim_stacks + 1)) {
        return NULL;
    }
    
    ufbx_anim_stack *stack = (ufbx_anim_stack*)calloc(1, sizeof(ufbx_anim_stack));
    if (!stack) {
        return NULL;
    }
    
    // Initialize animation stack
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(stack);
        return NULL;
    }
    strcpy(name_copy, name);
    stack->element.name.data = name_copy;
    stack->element.name.length = name_len;
    stack->element.element_id = scene_imp->num_anim_stacks + 5000; // Offset to avoid ID conflicts
    stack->element.type = UFBX_ELEMENT_ANIM_STACK;
    
    // Initialize time range
    stack->time_begin = 0.0;
    stack->time_end = 0.0;
    
    // Initialize layers list
    stack->layers.data = NULL;
    stack->layers.count = 0;
    
    scene_imp->anim_stacks[scene_imp->num_anim_stacks] = stack;
    scene_imp->num_anim_stacks++;
    
    scene_imp->scene.anim_stacks.data = (ufbx_anim_stack**)scene_imp->anim_stacks;
    scene_imp->scene.anim_stacks.count = scene_imp->num_anim_stacks;
    
    return stack;
}

// Add an animation layer to a stack
ufbx_anim_layer *ufbx_add_anim_layer(ufbx_export_scene *scene, ufbx_anim_stack *stack, const char *name) {
    if (!scene || !stack || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->anim_layers, &scene_imp->anim_layers_cap,
                                sizeof(ufbx_anim_layer*), scene_imp->num_anim_layers + 1)) {
        return NULL;
    }
    
    ufbx_anim_layer *layer = (ufbx_anim_layer*)calloc(1, sizeof(ufbx_anim_layer));
    if (!layer) {
        return NULL;
    }
    
    // Initialize animation layer
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(layer);
        return NULL;
    }
    strcpy(name_copy, name);
    layer->element.name.data = name_copy;
    layer->element.name.length = name_len;
    layer->element.element_id = scene_imp->num_anim_layers + 6000; // Offset to avoid ID conflicts
    layer->element.type = UFBX_ELEMENT_ANIM_LAYER;
    
    // Initialize layer properties
    layer->weight = 1.0;
    layer->weight_is_animated = false;
    layer->blended = false;
    layer->additive = false;
    layer->compose_rotation = false;
    layer->compose_scale = false;
    
    // Initialize lists
    layer->anim_values.data = NULL;
    layer->anim_values.count = 0;
    layer->anim_props.data = NULL;
    layer->anim_props.count = 0;
    
    scene_imp->anim_layers[scene_imp->num_anim_layers] = layer;
    scene_imp->num_anim_layers++;
    
    // Add layer to stack's layers list - properly grow the array
    size_t new_count = stack->layers.count + 1;
    ufbx_anim_layer **new_layers = (ufbx_anim_layer**)realloc(stack->layers.data, new_count * sizeof(ufbx_anim_layer*));
    if (!new_layers) {
        // Clean up the layer we just allocated
        free((void*)layer->element.name.data);
        free(layer);
        return NULL;
    }
    
    new_layers[stack->layers.count] = layer;
    stack->layers.data = new_layers;
    stack->layers.count = new_count;
    
    return layer;
}

// Add an animation value to a layer
ufbx_anim_value *ufbx_add_anim_value(ufbx_export_scene *scene, ufbx_anim_layer *layer, const char *name) {
    if (!scene || !layer || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->anim_values, &scene_imp->anim_values_cap,
                                sizeof(ufbx_anim_value*), scene_imp->num_anim_values + 1)) {
        return NULL;
    }
    
    ufbx_anim_value *value = (ufbx_anim_value*)calloc(1, sizeof(ufbx_anim_value));
    if (!value) {
        return NULL;
    }
    
    // Initialize animation value
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(value);
        return NULL;
    }
    strcpy(name_copy, name);
    value->element.name.data = name_copy;
    value->element.name.length = name_len;
    value->element.element_id = scene_imp->num_anim_values + 7000; // Offset to avoid ID conflicts
    value->element.type = UFBX_ELEMENT_ANIM_VALUE;
    
    // Initialize default value and curves
    value->default_value = (ufbx_vec3){{ 0, 0, 0 }};
    value->curves[0] = NULL;
    value->curves[1] = NULL;
    value->curves[2] = NULL;
    
    scene_imp->anim_values[scene_imp->num_anim_values] = value;
    scene_imp->num_anim_values++;
    
    // Add value to layer's anim_values list - properly grow the array
    size_t new_count = layer->anim_values.count + 1;
    ufbx_anim_value **new_values = (ufbx_anim_value**)realloc(layer->anim_values.data, new_count * sizeof(ufbx_anim_value*));
    if (!new_values) {
        // Clean up the value we just allocated
        free((void*)value->element.name.data);
        free(value);
        return NULL;
    }
    
    new_values[layer->anim_values.count] = value;
    layer->anim_values.data = new_values;
    layer->anim_values.count = new_count;
    
    return value;
}

// Add an animation curve to an animation value
ufbx_anim_curve *ufbx_add_anim_curve(ufbx_export_scene *scene, ufbx_anim_value *value, int component) {
    if (!scene || !value || component < 0 || component > 2) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->anim_curves, &scene_imp->anim_curves_cap,
                                sizeof(ufbx_anim_curve*), scene_imp->num_anim_curves + 1)) {
        return NULL;
    }
    
    ufbx_anim_curve *curve = (ufbx_anim_curve*)calloc(1, sizeof(ufbx_anim_curve));
    if (!curve) {
        return NULL;
    }
    
    // Initialize animation curve
    char curve_name[256];
    snprintf(curve_name, sizeof(curve_name), "%s_%c", value->element.name.data, 'X' + component);
    size_t curve_name_len = strlen(curve_name);
    char *curve_name_copy = malloc(curve_name_len + 1);
    if (!curve_name_copy) {
        free(curve);
        return NULL;
    }
    strcpy(curve_name_copy, curve_name);
    curve->element.name.data = curve_name_copy;
    curve->element.name.length = curve_name_len;
    curve->element.element_id = scene_imp->num_anim_curves + 8000; // Offset to avoid ID conflicts
    curve->element.type = UFBX_ELEMENT_ANIM_CURVE;
    
    // Initialize keyframes list
    curve->keyframes.data = NULL;
    curve->keyframes.count = 0;
    
    // Initialize extrapolation
    curve->pre_extrapolation.mode = UFBX_EXTRAPOLATION_CONSTANT;
    curve->pre_extrapolation.repeat_count = -1;
    curve->post_extrapolation.mode = UFBX_EXTRAPOLATION_CONSTANT;
    curve->post_extrapolation.repeat_count = -1;
    
    // Initialize value range
    curve->min_value = 0.0;
    curve->max_value = 0.0;
    curve->min_time = 0.0;
    curve->max_time = 0.0;
    
    scene_imp->anim_curves[scene_imp->num_anim_curves] = curve;
    scene_imp->num_anim_curves++;
    
    // Attach curve to animation value
    value->curves[component] = curve;
    
    return curve;
}

// Set animation curve keyframes
bool ufbx_set_anim_curve_keyframes(ufbx_anim_curve *curve, const ufbx_keyframe *keyframes, size_t num_keyframes, ufbx_error *error) {
    if (!curve || !keyframes || num_keyframes == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid curve, keyframes, or keyframe count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate keyframe data
    ufbx_keyframe *keyframe_data = (ufbx_keyframe*)malloc(num_keyframes * sizeof(ufbx_keyframe));
    if (!keyframe_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate keyframe data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy keyframe data
    memcpy(keyframe_data, keyframes, num_keyframes * sizeof(ufbx_keyframe));
    
    // Free existing keyframes if any
    if (curve->keyframes.data) {
        free(curve->keyframes.data);
    }
    
    // Set curve keyframe data
    curve->keyframes.data = keyframe_data;
    curve->keyframes.count = num_keyframes;
    
    // Update value and time ranges
    if (num_keyframes > 0) {
        curve->min_value = keyframes[0].value;
        curve->max_value = keyframes[0].value;
        curve->min_time = keyframes[0].time;
        curve->max_time = keyframes[0].time;
        
        for (size_t i = 1; i < num_keyframes; i++) {
            if (keyframes[i].value < curve->min_value) {
                curve->min_value = keyframes[i].value;
            }
            if (keyframes[i].value > curve->max_value) {
                curve->max_value = keyframes[i].value;
            }
            if (keyframes[i].time < curve->min_time) {
                curve->min_time = keyframes[i].time;
            }
            if (keyframes[i].time > curve->max_time) {
                curve->max_time = keyframes[i].time;
            }
        }
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Bone construction helpers

// Add a bone to a node
ufbx_bone *ufbx_add_bone(ufbx_export_scene *scene, ufbx_node *node, const char *name) {
    if (!scene || !node || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    // Grow bones array
    if (!ufbx_export_grow_array((void**)&scene_imp->bones, &scene_imp->bones_cap,
                                sizeof(ufbx_bone*), scene_imp->num_bones + 1)) {
        return NULL;
    }
    
    // Allocate new bone
    ufbx_bone *bone = (ufbx_bone*)calloc(1, sizeof(ufbx_bone));
    if (!bone) {
        return NULL;
    }
    
    // Initialize bone
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(bone);
        return NULL;
    }
    strcpy(name_copy, name);
    bone->element.name.data = name_copy;
    bone->element.name.length = name_len;
    bone->element.element_id = scene_imp->num_bones + 9000; // Offset to avoid ID conflicts
    bone->element.type = UFBX_ELEMENT_BONE;
    
    // Set default bone properties
    bone->relative_length = 1.0;
    bone->is_root = (node->parent == NULL || node->parent->bone == NULL);
    
    // Add to scene tracking
    scene_imp->bones[scene_imp->num_bones] = bone;
    scene_imp->num_bones++;
    
    // Update scene bones list
    scene_imp->scene.bones.data = (ufbx_bone**)scene_imp->bones;
    scene_imp->scene.bones.count = scene_imp->num_bones;
    
    // Attach bone to node
    node->bone = bone;
    node->attrib = &bone->element;
    node->attrib_type = UFBX_ELEMENT_BONE;
    
    return bone;
}

// Set bone properties
bool ufbx_set_bone_properties(ufbx_bone *bone, ufbx_real relative_length, ufbx_error *error) {
    if (!bone) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid bone");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    bone->relative_length = relative_length;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Add a single keyframe to an animation curve
bool ufbx_add_keyframe(ufbx_anim_curve *curve, double time, ufbx_real value, ufbx_interpolation interpolation, ufbx_error *error) {
    if (!curve) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid curve");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Grow keyframes array
    size_t new_count = curve->keyframes.count + 1;
    ufbx_keyframe *new_keyframes = (ufbx_keyframe*)realloc(curve->keyframes.data, new_count * sizeof(ufbx_keyframe));
    if (!new_keyframes) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate keyframe data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Add new keyframe
    ufbx_keyframe *keyframe = &new_keyframes[curve->keyframes.count];
    keyframe->time = time;
    keyframe->value = value;
    keyframe->interpolation = interpolation;
    keyframe->left.dx = 0.0f;
    keyframe->left.dy = 0.0f;
    keyframe->right.dx = 0.0f;
    keyframe->right.dy = 0.0f;
    
    curve->keyframes.data = new_keyframes;
    curve->keyframes.count = new_count;
    
    // Update value and time ranges
    if (curve->keyframes.count == 1) {
        curve->min_value = value;
        curve->max_value = value;
        curve->min_time = time;
        curve->max_time = time;
    } else {
        if (value < curve->min_value) curve->min_value = value;
        if (value > curve->max_value) curve->max_value = value;
        if (time < curve->min_time) curve->min_time = time;
        if (time > curve->max_time) curve->max_time = time;
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set animation curve extrapolation modes
bool ufbx_set_anim_curve_extrapolation(ufbx_anim_curve *curve, ufbx_extrapolation pre, ufbx_extrapolation post, ufbx_error *error) {
    if (!curve) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid curve");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    curve->pre_extrapolation = pre;
    curve->post_extrapolation = post;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Connect animation property to element
bool ufbx_connect_anim_prop(ufbx_export_scene *scene, ufbx_anim_layer *layer, ufbx_element *element, const char *prop_name, ufbx_anim_value *value, ufbx_error *error) {
    if (!scene || !layer || !element || !prop_name || !value) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid scene, layer, element, property name, or value");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate new animation property
    ufbx_anim_prop *anim_prop = (ufbx_anim_prop*)calloc(1, sizeof(ufbx_anim_prop));
    if (!anim_prop) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate animation property");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Initialize animation property
    anim_prop->element = element;
    size_t prop_name_len = strlen(prop_name);
    char *prop_name_copy = malloc(prop_name_len + 1);
    if (!prop_name_copy) {
        free(anim_prop);
        return false;
    }
    strcpy(prop_name_copy, prop_name);
    anim_prop->prop_name.data = prop_name_copy;
    anim_prop->prop_name.length = prop_name_len;
    anim_prop->anim_value = value;
    
    // Add to layer's anim_props list - properly grow the array
    size_t new_count = layer->anim_props.count + 1;
    ufbx_anim_prop *new_props = (ufbx_anim_prop*)realloc(layer->anim_props.data, new_count * sizeof(ufbx_anim_prop));
    if (!new_props) {
        free((void*)anim_prop->prop_name.data);
        free(anim_prop);
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to grow anim props array");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    new_props[layer->anim_props.count] = *anim_prop;
    layer->anim_props.data = new_props;
    layer->anim_props.count = new_count;
    
    free(anim_prop); // We copied the data, so free the temporary allocation
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set animation stack time range
bool ufbx_set_anim_stack_time_range(ufbx_anim_stack *stack, double time_begin, double time_end, ufbx_error *error) {
    if (!stack) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid animation stack");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    if (time_end < time_begin) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid time range: end time (%f) is before begin time (%f)", time_end, time_begin);
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    stack->time_begin = time_begin;
    stack->time_end = time_end;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

ufbx_export_scene *ufbx_create_scene(const ufbx_export_opts *opts) {
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)calloc(1, sizeof(ufbxi_export_scene));
    if (!scene_imp) {
        return NULL;
    }
    
    if (opts && opts->temp_allocator.allocator.alloc_fn) {
        scene_imp->allocator = opts->temp_allocator.allocator;
    } else {
        scene_imp->allocator.alloc_fn = (ufbx_alloc_fn*)malloc;
        scene_imp->allocator.realloc_fn = (ufbx_realloc_fn*)realloc;
        scene_imp->allocator.free_fn = (ufbx_free_fn*)free;
        scene_imp->allocator.user = NULL;
    }
    
    scene_imp->scene.metadata.creator = "ufbx_export";
    scene_imp->scene.metadata.version = 7400; // FBX 2014/2015 format
    scene_imp->scene.metadata.is_big_endian = false;
    
    // Set coordinate system (Godot uses Y-up, right-handed)
    scene_imp->scene.settings.axes.up = UFBX_COORDINATE_AXIS_POSITIVE_Y;
    scene_imp->scene.settings.axes.front = UFBX_COORDINATE_AXIS_NEGATIVE_Z;
    scene_imp->scene.settings.axes.right = UFBX_COORDINATE_AXIS_POSITIVE_X;
    scene_imp->scene.settings.unit_meters = 1.0;
    scene_imp->scene.settings.frames_per_second = 30.0;
    
    scene_imp->scene.nodes.data = NULL;
    scene_imp->scene.nodes.count = 0;
    scene_imp->scene.meshes.data = NULL;
    scene_imp->scene.meshes.count = 0;
    scene_imp->scene.materials.data = NULL;
    scene_imp->scene.materials.count = 0;
    scene_imp->scene.bones.data = NULL;
    scene_imp->scene.bones.count = 0;
    scene_imp->scene.anim_stacks.data = NULL;
    scene_imp->scene.anim_stacks.count = 0;
    
    return &scene_imp->scene;
}

void ufbx_free_export_scene(ufbx_export_scene *scene) {
    if (!scene) {
        return;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        if (scene_imp->nodes[i]) {
            free(scene_imp->nodes[i]);
        }
    }
    free(scene_imp->nodes);
    
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        if (scene_imp->meshes[i]) {
            ufbx_mesh *mesh = scene_imp->meshes[i];
            free(mesh->vertices.data);
            free(mesh->vertex_indices.data);
            free(mesh->faces.data);
            free(mesh);
        }
    }
    free(scene_imp->meshes);
    
    for (size_t i = 0; i < scene_imp->num_materials; i++) {
        if (scene_imp->materials[i]) {
            free(scene_imp->materials[i]);
        }
    }
    free(scene_imp->materials);
    
    for (size_t i = 0; i < scene_imp->num_bones; i++) {
        if (scene_imp->bones[i]) {
            free((void*)scene_imp->bones[i]->element.name.data);
            free(scene_imp->bones[i]);
        }
    }
    free(scene_imp->bones);
    
    for (size_t i = 0; i < scene_imp->num_anim_stacks; i++) {
        if (scene_imp->anim_stacks[i]) {
            free(scene_imp->anim_stacks[i]);
        }
    }
    free(scene_imp->anim_stacks);
    
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        if (scene_imp->anim_layers[i]) {
            free(scene_imp->anim_layers[i]);
        }
    }
    free(scene_imp->anim_layers);
    
    for (size_t i = 0; i < scene_imp->num_anim_values; i++) {
        if (scene_imp->anim_values[i]) {
            free(scene_imp->anim_values[i]);
        }
    }
    free(scene_imp->anim_values);
    
    for (size_t i = 0; i < scene_imp->num_anim_curves; i++) {
        if (scene_imp->anim_curves[i]) {
            ufbx_anim_curve *curve = scene_imp->anim_curves[i];
            free(curve->keyframes.data);
            free(curve);
        }
    }
    free(scene_imp->anim_curves);
    
    free(scene_imp);
}

ufbx_node *ufbx_add_node(ufbx_export_scene *scene, const char *name, ufbx_node *parent) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->nodes, &scene_imp->nodes_cap, 
                                sizeof(ufbx_node*), scene_imp->num_nodes + 1)) {
        return NULL;
    }
    
    ufbx_node *node = (ufbx_node*)calloc(1, sizeof(ufbx_node));
    if (!node) {
        return NULL;
    }
    
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(node);
        return NULL;
    }
    strcpy(name_copy, name);
    node->element.name.data = name_copy;
    node->element.name.length = name_len;
    node->element.element_id = scene_imp->num_nodes + 1;
    node->element.type = UFBX_ELEMENT_NODE;
    
    // Initialize children list
    node->children.data = NULL;
    node->children.count = 0;
    
    node->parent = parent;
    if (parent) {
        // Add to parent's children list
        size_t new_count = parent->children.count + 1;
        ufbx_node **new_children = (ufbx_node**)realloc(parent->children.data, new_count * sizeof(ufbx_node*));
        if (!new_children) {
            free((void*)node->element.name.data);
            free(node);
            return NULL;
        }
        new_children[parent->children.count] = node;
        parent->children.data = new_children;
        parent->children.count = new_count;
    }
    
    node->local_transform.translation = (ufbx_vec3){{ 0, 0, 0 }};
    node->local_transform.rotation = (ufbx_quat){{ 0, 0, 0, 1 }};
    node->local_transform.scale = (ufbx_vec3){{ 1, 1, 1 }};
    
    // Initialize other node properties
    node->mesh = NULL;
    node->light = NULL;
    node->camera = NULL;
    node->bone = NULL;
    node->attrib = NULL;
    node->attrib_type = UFBX_ELEMENT_UNKNOWN;
    node->node_depth = parent ? parent->node_depth + 1 : 0;
    
    scene_imp->nodes[scene_imp->num_nodes] = node;
    scene_imp->num_nodes++;
    
    scene_imp->scene.nodes.data = (ufbx_node**)scene_imp->nodes;
    scene_imp->scene.nodes.count = scene_imp->num_nodes;
    
    return node;
}

ufbx_mesh *ufbx_add_mesh(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->meshes, &scene_imp->meshes_cap,
                                sizeof(ufbx_mesh*), scene_imp->num_meshes + 1)) {
        return NULL;
    }
    
    ufbx_mesh *mesh = (ufbx_mesh*)calloc(1, sizeof(ufbx_mesh));
    if (!mesh) {
        return NULL;
    }
    
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(mesh);
        return NULL;
    }
    strcpy(name_copy, name);
    mesh->element.name.data = name_copy;
    mesh->element.name.length = name_len;
    mesh->element.element_id = scene_imp->num_meshes + 1000; // Offset to avoid ID conflicts
    mesh->element.type = UFBX_ELEMENT_MESH;
    
    scene_imp->meshes[scene_imp->num_meshes] = mesh;
    scene_imp->num_meshes++;
    
    scene_imp->scene.meshes.data = (ufbx_mesh**)scene_imp->meshes;
    scene_imp->scene.meshes.count = scene_imp->num_meshes;
    
    return mesh;
}

ufbx_material *ufbx_add_material(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->materials, &scene_imp->materials_cap,
                                sizeof(ufbx_material*), scene_imp->num_materials + 1)) {
        return NULL;
    }
    
    ufbx_material *material = (ufbx_material*)calloc(1, sizeof(ufbx_material));
    if (!material) {
        return NULL;
    }
    
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(material);
        return NULL;
    }
    strcpy(name_copy, name);
    material->element.name.data = name_copy;
    material->element.name.length = name_len;
    material->element.element_id = scene_imp->num_materials + 2000; // Offset to avoid ID conflicts
    material->element.type = UFBX_ELEMENT_MATERIAL;
    
    material->pbr.base_color.value_vec3 = (ufbx_vec3){{ 0.8f, 0.8f, 0.8f }};
    material->pbr.base_color.has_value = true;
    material->pbr.roughness.value_real = 0.5;
    material->pbr.roughness.has_value = true;
    material->pbr.metalness.value_real = 0.0;
    material->pbr.metalness.has_value = true;
    
    scene_imp->materials[scene_imp->num_materials] = material;
    scene_imp->num_materials++;
    
    scene_imp->scene.materials.data = (ufbx_material**)scene_imp->materials;
    scene_imp->scene.materials.count = scene_imp->num_materials;
    
    return material;
}

bool ufbx_set_mesh_vertices(ufbx_mesh *mesh, const ufbx_vec3 *vertices, size_t num_vertices, ufbx_error *error) {
    if (!mesh || !vertices || num_vertices == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, vertices, or vertex count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    ufbx_vec3 *vertex_data = (ufbx_vec3*)malloc(num_vertices * sizeof(ufbx_vec3));
    if (!vertex_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate vertex data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    memcpy(vertex_data, vertices, num_vertices * sizeof(ufbx_vec3));
    
    mesh->vertices.data = vertex_data;
    mesh->vertices.count = num_vertices;
    mesh->num_vertices = num_vertices;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

bool ufbx_set_mesh_faces(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices, 
                         const ufbx_face *face_definitions, size_t num_faces, ufbx_error *error) {
    if (!mesh || !indices || !face_definitions || num_indices == 0 || num_faces == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, indices, face definitions, or counts");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    size_t total_expected_indices = 0;
    size_t triangle_count = 0;
    for (size_t i = 0; i < num_faces; i++) {
        if (face_definitions[i].num_indices < 3) {
            if (error) {
                error->type = UFBX_ERROR_UNKNOWN;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Face %zu has invalid vertex count: %u (minimum 3)", i, face_definitions[i].num_indices);
                error->info_length = strlen(error->info);
            }
            return false;
        }
        total_expected_indices += face_definitions[i].num_indices;
        if (face_definitions[i].num_indices == 3) {
            triangle_count++;
        }
    }
    
    if (total_expected_indices != num_indices) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Face definitions expect %zu indices but got %zu", total_expected_indices, num_indices);
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    uint32_t *index_data = (uint32_t*)malloc(num_indices * sizeof(uint32_t));
    if (!index_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate index data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    memcpy(index_data, indices, num_indices * sizeof(uint32_t));
    
    ufbx_face *faces = (ufbx_face*)malloc(num_faces * sizeof(ufbx_face));
    if (!faces) {
        free(index_data);
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate face data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    memcpy(faces, face_definitions, num_faces * sizeof(ufbx_face));
    
    mesh->vertex_indices.data = index_data;
    mesh->vertex_indices.count = num_indices;
    mesh->faces.data = faces;
    mesh->faces.count = num_faces;
    mesh->num_faces = num_faces;
    mesh->num_triangles = triangle_count;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

bool ufbx_set_mesh_indices(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices, size_t vertices_per_face, ufbx_error *error) {
    if (!mesh || !indices || num_indices == 0 || vertices_per_face == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, indices, index count, or vertices per face");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    if (num_indices % vertices_per_face != 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Index count (%zu) not divisible by vertices per face (%zu)", num_indices, vertices_per_face);
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    size_t num_faces = num_indices / vertices_per_face;
    ufbx_face *face_definitions = (ufbx_face*)malloc(num_faces * sizeof(ufbx_face));
    if (!face_definitions) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate face definitions");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    for (size_t i = 0; i < num_faces; i++) {
        face_definitions[i].index_begin = i * vertices_per_face;
        face_definitions[i].num_indices = vertices_per_face;
    }
    
    bool result = ufbx_set_mesh_faces(mesh, indices, num_indices, face_definitions, num_faces, error);
    free(face_definitions);
    
    return result;
}


bool ufbx_set_material_property(ufbx_material *material, const char *property_name, 
                                int type, const void *value) {
    if (!material || !property_name || !value) {
        return false;
    }
    
    if (strcmp(property_name, "DiffuseColor") == 0) {
        material->pbr.base_color.value_vec3 = *(const ufbx_vec3*)value;
        material->pbr.base_color.has_value = true;
        return true;
    } else if (strcmp(property_name, "Roughness") == 0) {
        material->pbr.roughness.value_real = *(const ufbx_real*)value;
        material->pbr.roughness.has_value = true;
        return true;
    } else if (strcmp(property_name, "Metallic") == 0) {
        material->pbr.metalness.value_real = *(const ufbx_real*)value;
        material->pbr.metalness.has_value = true;
        return true;
    }
    
    // TODO: Support more material properties and custom properties
    return false;
}

// Forward declarations for writer functions (implemented in ufbx_export_writer.c)
extern ufbx_error ufbx_export_to_file_impl(const ufbx_export_scene *scene, const char *filename, 
                                           const ufbx_export_opts *opts);
extern ufbx_export_result ufbx_export_to_memory_impl(const ufbx_export_scene *scene, 
                                                     const ufbx_export_opts *opts);

// Validation function
static ufbx_error ufbx_validate_export_scene(const ufbx_export_scene *scene) {
    ufbx_error error = { UFBX_ERROR_NONE };
    
    if (!scene) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Scene is NULL");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Validate scene metadata
    if (!scene->metadata.creator) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Scene metadata creator is NULL");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Validate coordinate system
    if (scene->settings.axes.up == UFBX_COORDINATE_AXIS_UNKNOWN ||
        scene->settings.axes.front == UFBX_COORDINATE_AXIS_UNKNOWN ||
        scene->settings.axes.right == UFBX_COORDINATE_AXIS_UNKNOWN) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Invalid coordinate system in scene settings");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Validate unit scale
    if (scene->settings.unit_meters <= 0.0) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Invalid unit scale in scene settings");
        error.info_length = strlen(error.info);
        return error;
    }
    
    return error;
}


// Set mesh normals with error handling
bool ufbx_set_mesh_normals(ufbx_mesh *mesh, const ufbx_vec3 *normals, size_t num_normals, ufbx_error *error) {
    if (!mesh || !normals || num_normals == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, normals, or normal count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate normal data
    ufbx_vec3 *normal_data = (ufbx_vec3*)malloc(num_normals * sizeof(ufbx_vec3));
    if (!normal_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate normal data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy normal data
    memcpy(normal_data, normals, num_normals * sizeof(ufbx_vec3));
    
    // Set mesh normal data
    mesh->vertex_normal.values.data = normal_data;
    mesh->vertex_normal.values.count = num_normals;
    mesh->vertex_normal.exists = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set mesh UV coordinates with error handling
bool ufbx_set_mesh_uvs(ufbx_mesh *mesh, const ufbx_vec2 *uvs, size_t num_uvs, ufbx_error *error) {
    if (!mesh || !uvs || num_uvs == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, UVs, or UV count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate UV data
    ufbx_vec2 *uv_data = (ufbx_vec2*)malloc(num_uvs * sizeof(ufbx_vec2));
    if (!uv_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate UV data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy UV data
    memcpy(uv_data, uvs, num_uvs * sizeof(ufbx_vec2));
    
    // Set mesh UV data
    mesh->vertex_uv.values.data = uv_data;
    mesh->vertex_uv.values.count = num_uvs;
    mesh->vertex_uv.exists = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set node transform with error handling
void ufbx_set_node_transform(ufbx_node *node, const ufbx_transform *transform, ufbx_error *error) {
    if (!node || !transform) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid node or transform");
            error->info_length = strlen(error->info);
        }
        return;
    }
    
    node->local_transform = *transform;
    // TODO: Convert transform to matrix properly when ufbx is fully linked
    // For now, set identity matrix
    node->node_to_parent = ufbx_identity_matrix;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
}

// Attach mesh to node with error handling
bool ufbx_attach_mesh_to_node(ufbx_node *node, ufbx_mesh *mesh, ufbx_error *error) {
    if (!node || !mesh) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid node or mesh");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    node->mesh = mesh;
    node->attrib = &mesh->element;
    node->attrib_type = UFBX_ELEMENT_MESH;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Attach material to mesh with surface index
bool ufbx_attach_material_to_mesh(ufbx_mesh *mesh, ufbx_material *material, int surface_index, ufbx_error *error) {
    if (!mesh || !material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh or material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate materials array if not already allocated
    if (!mesh->materials.data) {
        mesh->materials.data = (ufbx_material**)malloc(sizeof(ufbx_material*));
        if (!mesh->materials.data) {
            if (error) {
                error->type = UFBX_ERROR_OUT_OF_MEMORY;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate materials array");
                error->info_length = strlen(error->info);
            }
            return false;
        }
    }
    
    // Set the material
    mesh->materials.data[0] = material;
    mesh->materials.count = 1;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set material albedo color
bool ufbx_set_material_albedo(ufbx_material *material, ufbx_real r, ufbx_real g, ufbx_real b, ufbx_real a, ufbx_error *error) {
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    material->pbr.base_color.value_vec4 = (ufbx_vec4){{ r, g, b, a }};
    material->pbr.base_color.has_value = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set material metallic and roughness
bool ufbx_set_material_metallic_roughness(ufbx_material *material, ufbx_real metallic, ufbx_real roughness, ufbx_error *error) {
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    material->pbr.metalness.value_real = metallic;
    material->pbr.metalness.has_value = true;
    material->pbr.roughness.value_real = roughness;
    material->pbr.roughness.has_value = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set material emission
bool ufbx_set_material_emission(ufbx_material *material, ufbx_real r, ufbx_real g, ufbx_real b, ufbx_error *error) {
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    material->pbr.emission_color.value_vec3 = (ufbx_vec3){{ r, g, b }};
    material->pbr.emission_color.has_value = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Add a texture to the scene
ufbx_texture *ufbx_add_texture(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    // Allocate new texture
    ufbx_texture *texture = (ufbx_texture*)calloc(1, sizeof(ufbx_texture));
    if (!texture) {
        return NULL;
    }
    
    // Initialize texture
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(texture);
        return NULL;
    }
    strcpy(name_copy, name);
    texture->element.name.data = name_copy;
    texture->element.name.length = name_len;
    texture->element.type = UFBX_ELEMENT_TEXTURE;
    
    return texture;
}

// Set texture data
bool ufbx_set_texture_data(ufbx_texture *texture, const void *data, size_t size, const char *format, ufbx_error *error) {
    if (!texture || !data || size == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid texture, data, or size");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // TODO: Implement texture data storage
    // For now, just mark as successful
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Attach texture to material
bool ufbx_attach_texture_to_material(ufbx_material *material, ufbx_texture *texture, const char *property_name, ufbx_error *error) {
    if (!material || !texture || !property_name) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material, texture, or property name");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Attach texture based on property name
    if (strcmp(property_name, "BaseColor") == 0 || strcmp(property_name, "DiffuseColor") == 0) {
        material->pbr.base_color.texture = texture;
    } else if (strcmp(property_name, "NormalMap") == 0) {
        material->pbr.normal_map.texture = texture;
    } else if (strcmp(property_name, "Metallic") == 0) {
        material->pbr.metalness.texture = texture;
    } else if (strcmp(property_name, "Roughness") == 0) {
        material->pbr.roughness.texture = texture;
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Get export buffer size
size_t ufbx_get_export_size(const ufbx_export_scene *scene, const ufbx_export_opts *opts, ufbx_error *error) {
    if (!scene) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Scene is NULL");
            error->info_length = strlen(error->info);
        }
        return 0;
    }
    
    // TODO: Calculate actual export size based on scene content
    // For now, return a reasonable estimate
    size_t estimated_size = 1024 * 1024; // 1MB base size
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return estimated_size;
}

// Export scene to memory buffer
size_t ufbx_export_to_memory(const ufbx_export_scene *scene, void *buffer, size_t buffer_size, const ufbx_export_opts *opts, ufbx_error *error) {
    if (!scene || !buffer || buffer_size == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid scene, buffer, or buffer size");
            error->info_length = strlen(error->info);
        }
        return 0;
    }
    
    // Validate scene structure
    ufbx_error validation_error = ufbx_validate_export_scene(scene);
    if (validation_error.type != UFBX_ERROR_NONE) {
        if (error) *error = validation_error;
        return 0;
    }
    
    // Use the writer implementation to export to memory
    ufbx_export_result result = ufbx_export_to_memory_impl(scene, opts);
    if (result.error.type != UFBX_ERROR_NONE) {
        if (error) *error = result.error;
        return 0;
    }
    
    // Check if buffer is large enough
    if (result.size > buffer_size) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Buffer too small: need %zu bytes, got %zu", result.size, buffer_size);
            error->info_length = strlen(error->info);
        }
        free(result.data);
        return 0;
    }
    
    // Copy data to user buffer
    memcpy(buffer, result.data, result.size);
    free(result.data);
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return result.size;
}

// Export scene to file
bool ufbx_export_to_file(const ufbx_export_scene *scene, const char *filename, const ufbx_export_opts *opts, ufbx_error *error) {
    if (!scene || !filename) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid scene or filename");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    ufbx_error validation_error = ufbx_validate_export_scene(scene);
    if (validation_error.type != UFBX_ERROR_NONE) {
        if (error) *error = validation_error;
        return false;
    }
    
    ufbx_error export_error = ufbx_export_to_file_impl(scene, filename, opts);
    if (export_error.type != UFBX_ERROR_NONE) {
        if (error) *error = export_error;
        return false;
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Skinning export functions

// Add a skin deformer to the scene
ufbx_skin_deformer *ufbx_add_skin_deformer(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->skin_deformers, &scene_imp->skin_deformers_cap,
                                sizeof(ufbx_skin_deformer*), scene_imp->num_skin_deformers + 1)) {
        return NULL;
    }
    
    // Allocate new skin deformer
    ufbx_skin_deformer *skin = (ufbx_skin_deformer*)calloc(1, sizeof(ufbx_skin_deformer));
    if (!skin) {
        return NULL;
    }
    
    // Initialize skin deformer
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(skin);
        return NULL;
    }
    strcpy(name_copy, name);
    skin->element.name.data = name_copy;
    skin->element.name.length = name_len;
    skin->element.element_id = scene_imp->num_skin_deformers + 3000; // Offset to avoid ID conflicts
    skin->element.type = UFBX_ELEMENT_SKIN_DEFORMER;
    skin->skinning_method = UFBX_SKINNING_METHOD_LINEAR;
    
    // Add to scene tracking
    scene_imp->skin_deformers[scene_imp->num_skin_deformers] = skin;
    scene_imp->num_skin_deformers++;
    
    scene_imp->scene.skin_deformers.data = (ufbx_skin_deformer**)scene_imp->skin_deformers;
    scene_imp->scene.skin_deformers.count = scene_imp->num_skin_deformers;
    
    return skin;
}

// Add a skin cluster to a skin deformer
ufbx_skin_cluster *ufbx_add_skin_cluster(ufbx_export_scene *scene, ufbx_skin_deformer *skin, ufbx_node *bone_node, const char *name) {
    if (!scene || !skin || !bone_node || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->skin_clusters, &scene_imp->skin_clusters_cap,
                                sizeof(ufbx_skin_cluster*), scene_imp->num_skin_clusters + 1)) {
        return NULL;
    }
    
    // Allocate new skin cluster
    ufbx_skin_cluster *cluster = (ufbx_skin_cluster*)calloc(1, sizeof(ufbx_skin_cluster));
    if (!cluster) {
        return NULL;
    }
    
    // Initialize skin cluster
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(cluster);
        return NULL;
    }
    strcpy(name_copy, name);
    cluster->element.name.data = name_copy;
    cluster->element.name.length = name_len;
    cluster->element.element_id = scene_imp->num_skin_clusters + 4000; // Offset to avoid ID conflicts
    cluster->element.type = UFBX_ELEMENT_SKIN_CLUSTER;
    cluster->bone_node = bone_node;
    cluster->bind_to_world = ufbx_identity_matrix;
    cluster->geometry_to_bone = ufbx_identity_matrix;
    cluster->mesh_node_to_bone = ufbx_identity_matrix;
    
    // Add to scene tracking
    scene_imp->skin_clusters[scene_imp->num_skin_clusters] = cluster;
    scene_imp->num_skin_clusters++;
    
    scene_imp->scene.skin_clusters.data = (ufbx_skin_cluster**)scene_imp->skin_clusters;
    scene_imp->scene.skin_clusters.count = scene_imp->num_skin_clusters;
    
    // Add cluster to skin deformer's clusters list
    if (!skin->clusters.data) {
        skin->clusters.data = (ufbx_skin_cluster**)malloc(sizeof(ufbx_skin_cluster*));
        if (!skin->clusters.data) {
            return NULL;
        }
        skin->clusters.count = 0;
    } else {
        ufbx_skin_cluster **new_clusters = (ufbx_skin_cluster**)realloc(skin->clusters.data, (skin->clusters.count + 1) * sizeof(ufbx_skin_cluster*));
        if (!new_clusters) {
            return NULL;
        }
        skin->clusters.data = new_clusters;
    }
    
    skin->clusters.data[skin->clusters.count] = cluster;
    skin->clusters.count++;
    
    return cluster;
}

// Set skin weight data
bool ufbx_set_skin_weights(ufbx_skin_deformer *skin, const ufbx_skin_weight *weights, size_t num_weights, ufbx_error *error) {
    if (!skin || !weights || num_weights == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid skin, weights, or weight count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate weight data
    ufbx_skin_weight *weight_data = (ufbx_skin_weight*)malloc(num_weights * sizeof(ufbx_skin_weight));
    if (!weight_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate weight data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy weight data
    memcpy(weight_data, weights, num_weights * sizeof(ufbx_skin_weight));
    
    // Set skin weight data
    skin->weights.data = weight_data;
    skin->weights.count = num_weights;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set skin vertex data
bool ufbx_set_skin_vertices(ufbx_skin_deformer *skin, const ufbx_skin_vertex *vertices, size_t num_vertices, ufbx_error *error) {
    if (!skin || !vertices || num_vertices == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid skin, vertices, or vertex count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate vertex data
    ufbx_skin_vertex *vertex_data = (ufbx_skin_vertex*)malloc(num_vertices * sizeof(ufbx_skin_vertex));
    if (!vertex_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate vertex data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy vertex data
    memcpy(vertex_data, vertices, num_vertices * sizeof(ufbx_skin_vertex));
    
    // Set skin vertex data
    skin->vertices.data = vertex_data;
    skin->vertices.count = num_vertices;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Attach skin deformer to mesh
bool ufbx_attach_skin_to_mesh(ufbx_mesh *mesh, ufbx_skin_deformer *skin, ufbx_error *error) {
    if (!mesh || !skin) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh or skin");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate skin deformers array if not already allocated
    if (!mesh->skin_deformers.data) {
        mesh->skin_deformers.data = (ufbx_skin_deformer**)malloc(sizeof(ufbx_skin_deformer*));
        if (!mesh->skin_deformers.data) {
            if (error) {
                error->type = UFBX_ERROR_OUT_OF_MEMORY;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate skin deformers array");
                error->info_length = strlen(error->info);
            }
            return false;
        }
    }
    
    // Set the skin deformer
    mesh->skin_deformers.data[0] = skin;
    mesh->skin_deformers.count = 1;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Morph target export functions

// Add a blend deformer to the scene
ufbx_blend_deformer *ufbx_add_blend_deformer(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    // Allocate new blend deformer
    ufbx_blend_deformer *blend = (ufbx_blend_deformer*)calloc(1, sizeof(ufbx_blend_deformer));
    if (!blend) {
        return NULL;
    }
    
    // Initialize blend deformer
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(blend);
        return NULL;
    }
    strcpy(name_copy, name);
    blend->element.name.data = name_copy;
    blend->element.name.length = name_len;
    blend->element.type = UFBX_ELEMENT_BLEND_DEFORMER;
    
    return blend;
}

// Add a blend channel to a blend deformer
ufbx_blend_channel *ufbx_add_blend_channel(ufbx_export_scene *scene, ufbx_blend_deformer *deformer, const char *name) {
    if (!scene || !deformer || !name) {
        return NULL;
    }
    
    // Allocate new blend channel
    ufbx_blend_channel *channel = (ufbx_blend_channel*)calloc(1, sizeof(ufbx_blend_channel));
    if (!channel) {
        return NULL;
    }
    
    // Initialize blend channel
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(channel);
        return NULL;
    }
    strcpy(name_copy, name);
    channel->element.name.data = name_copy;
    channel->element.name.length = name_len;
    channel->element.type = UFBX_ELEMENT_BLEND_CHANNEL;
    channel->weight = 0.0f;
    
    return channel;
}

// Add a blend shape to the scene
ufbx_blend_shape *ufbx_add_blend_shape(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    // Allocate new blend shape
    ufbx_blend_shape *shape = (ufbx_blend_shape*)calloc(1, sizeof(ufbx_blend_shape));
    if (!shape) {
        return NULL;
    }
    
    // Initialize blend shape
    size_t name_len = strlen(name);
    char *name_copy = malloc(name_len + 1);
    if (!name_copy) {
        free(shape);
        return NULL;
    }
    strcpy(name_copy, name);
    shape->element.name.data = name_copy;
    shape->element.name.length = name_len;
    shape->element.type = UFBX_ELEMENT_BLEND_SHAPE;
    
    return shape;
}

// Set blend shape offset data
bool ufbx_set_blend_shape_offsets(ufbx_blend_shape *shape, const ufbx_vec3 *position_offsets, const ufbx_vec3 *normal_offsets, size_t num_offsets, ufbx_error *error) {
    if (!shape || (!position_offsets && !normal_offsets) || num_offsets == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid shape, offsets, or offset count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Set position offsets if provided
    if (position_offsets) {
        ufbx_vec3 *pos_data = (ufbx_vec3*)malloc(num_offsets * sizeof(ufbx_vec3));
        if (!pos_data) {
            if (error) {
                error->type = UFBX_ERROR_OUT_OF_MEMORY;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate position offset data");
                error->info_length = strlen(error->info);
            }
            return false;
        }
        memcpy(pos_data, position_offsets, num_offsets * sizeof(ufbx_vec3));
        shape->position_offsets.data = pos_data;
        shape->position_offsets.count = num_offsets;
    }
    
    // Set normal offsets if provided
    if (normal_offsets) {
        ufbx_vec3 *norm_data = (ufbx_vec3*)malloc(num_offsets * sizeof(ufbx_vec3));
        if (!norm_data) {
            if (error) {
                error->type = UFBX_ERROR_OUT_OF_MEMORY;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate normal offset data");
                error->info_length = strlen(error->info);
            }
            return false;
        }
        memcpy(norm_data, normal_offsets, num_offsets * sizeof(ufbx_vec3));
        shape->normal_offsets.data = norm_data;
        shape->normal_offsets.count = num_offsets;
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Attach blend deformer to mesh
bool ufbx_attach_blend_to_mesh(ufbx_mesh *mesh, ufbx_blend_deformer *blend, ufbx_error *error) {
    if (!mesh || !blend) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh or blend deformer");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate blend deformers array if not already allocated
    if (!mesh->blend_deformers.data) {
        mesh->blend_deformers.data = (ufbx_blend_deformer**)malloc(sizeof(ufbx_blend_deformer*));
        if (!mesh->blend_deformers.data) {
            if (error) {
                error->type = UFBX_ERROR_OUT_OF_MEMORY;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate blend deformers array");
                error->info_length = strlen(error->info);
            }
            return false;
        }
    }
    
    // Set the blend deformer
    mesh->blend_deformers.data[0] = blend;
    mesh->blend_deformers.count = 1;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set skin cluster transform matrices
bool ufbx_set_skin_cluster_transform(ufbx_skin_cluster *cluster, const ufbx_matrix *geometry_to_bone, const ufbx_matrix *bind_to_world, ufbx_error *error) {
    if (!cluster || !geometry_to_bone || !bind_to_world) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid cluster, geometry_to_bone, or bind_to_world matrix");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Set transform matrices
    cluster->geometry_to_bone = *geometry_to_bone;
    cluster->bind_to_world = *bind_to_world;
    
    // Calculate derived matrices if needed
    cluster->mesh_node_to_bone = *geometry_to_bone; // Simplified for now
    cluster->geometry_to_world = ufbx_identity_matrix; // Will be computed later by ufbx
    cluster->geometry_to_world_transform = ufbx_identity_transform;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set skin cluster vertex indices and weights (per-cluster arrays)
bool ufbx_set_skin_cluster_vertices(ufbx_skin_cluster *cluster, const uint32_t *vertices, const ufbx_real *weights, size_t num_vertices, ufbx_error *error) {
    if (!cluster || !vertices || !weights || num_vertices == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid cluster, vertices, weights, or vertex count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate vertex indices
    uint32_t *vertex_data = (uint32_t*)malloc(num_vertices * sizeof(uint32_t));
    if (!vertex_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate vertex indices");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate weight data
    ufbx_real *weight_data = (ufbx_real*)malloc(num_vertices * sizeof(ufbx_real));
    if (!weight_data) {
        free(vertex_data);
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate vertex weights");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy data
    memcpy(vertex_data, vertices, num_vertices * sizeof(uint32_t));
    memcpy(weight_data, weights, num_vertices * sizeof(ufbx_real));
    
    // Set cluster data
    cluster->vertices.data = vertex_data;
    cluster->vertices.count = num_vertices;
    cluster->weights.data = weight_data;
    cluster->weights.count = num_vertices;
    cluster->num_weights = num_vertices;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}
