/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/

#include "register_types.h"
#include "image_loader_svg_spatial.h"
#include "vector_graphics_adaptive_renderer.h"
#include "vector_graphics_color.h"
#include "vector_graphics_gradient.h"
#include "vector_graphics_linear_gradient.h"
#include "vector_graphics_paint.h"
#include "vector_graphics_path.h"
#include "vector_graphics_radial_gradient.h"
#include "vector_graphics_renderer.h"

#include "core/object/ref_counted.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
static void editor_init_callback() {
	Ref<EditorSceneImporterSVG> svg_spatial_loader;
	svg_spatial_loader.instantiate();
	ResourceImporterScene::add_scene_importer(svg_spatial_loader, true);
}
#endif

void initialize_svg_mesh_module(ModuleInitializationLevel p_level) {
	ClassDB::register_class<VGPath>();
	ClassDB::register_abstract_class<VGPaint>();
	ClassDB::register_class<VGColor>();
	ClassDB::register_class<VGGradient>();
	ClassDB::register_class<VGLinearGradient>();
	ClassDB::register_class<VGRadialGradient>();

	ClassDB::register_abstract_class<VGRenderer>();
	ClassDB::register_class<VGMeshRenderer>();
#ifdef TOOLS_ENABLED
	ClassDB::APIType prev_api = ClassDB::get_current_api();
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	GDREGISTER_CLASS(EditorSceneImporterSVG);
	ClassDB::set_current_api(prev_api);
	EditorNode::add_init_callback(editor_init_callback);
#endif
}

void uninitialize_svg_mesh_module(ModuleInitializationLevel p_level) {
}
