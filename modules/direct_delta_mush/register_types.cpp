/* register_types.cpp */

#include "register_types.h"

#include "core/object/class_db.h"
#include "direct_delta_mush.h"
#include "ddm_mesh.h"
#include "ddm_importer.h"

void initialize_direct_delta_mush_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // Register classes
    ClassDB::register_class<DDMMesh>();
    ClassDB::register_class<DDMImporter>();
    ClassDB::register_class<DirectDeltaMush>();
}

void uninitialize_direct_delta_mush_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // Cleanup if needed
}
