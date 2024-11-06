#include "gdduckdb.h"

#include "core/object/class_db.h"

#include "register_types.h"

void initialize_duckdb_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<GDDuckDB>();
}

void uninitialize_duckdb_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
