#ifndef SCENE_UTILS_H
#define SCENE_UTILS_H

#include "../../ufbx.h"
#include "../../ufbx_export.h"

// Error handling utilities
void print_error(const ufbx_error *error, const char *description);

// Scene analysis utilities
void print_warnings(ufbx_scene *scene);
void print_scene_info(ufbx_scene *scene);

// Scene copying functionality
bool copy_scene_data(ufbx_scene *source_scene, ufbx_export_scene *export_scene);

#endif // SCENE_UTILS_H
