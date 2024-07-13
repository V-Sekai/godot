#include "register_types.h"
#include "bimdf.h"
#include "core/class_db.h"


void initialize_libsatsuma_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
			return;
	}    
    ClassDB::register_class<BIMDFSolver>();
}

void uninitialize_libsatsuma_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
			return;
	}
   // Nothing to do here in this example.
}