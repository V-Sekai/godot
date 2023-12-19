#include "register_types.h"

#include "thirdparty/godot/material_x_3d.h"

#include "core/io/resource.h"

#ifdef TOOLS_ENABLED
    #include "thirdparty/godot/editor_material_material_x_plugin.h"
    #include "editor/editor_node.h"
#endif

static Ref<MTLXLoader> resource_format_mtlx;

void initialize_material_x_module(ModuleInitializationLevel p_level)
{
    if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE)
    {
        GDREGISTER_CLASS(MTLXLoader);
        resource_format_mtlx.instantiate();
        ResourceLoader::add_resource_format_loader(resource_format_mtlx);
#ifdef TOOLS_ENABLED
        // Editor-specific API.
        ClassDB::APIType prev_api = ClassDB::get_current_api();
        ClassDB::set_current_api(ClassDB::API_EDITOR);

        EditorPlugins::add_by_type<MaterialXPlugin>();

        ClassDB::set_current_api(prev_api);
#endif
        return;
    }
}

void uninitialize_material_x_module(ModuleInitializationLevel p_level)
{
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
    {
        return;
    }
    ResourceLoader::remove_resource_format_loader(resource_format_mtlx);
    resource_format_mtlx.unref();
}