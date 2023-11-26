#include "register_types.h"

#include "core/object/class_db.h"
#include "src/godot_mvsqlite.h"

void initialize_mvsqlite_module(ModuleInitializationLevel p_level) {
  if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
    return;
  } 
  ClassDB::register_class<MVSQLite>();
  ClassDB::register_class<MVSQLiteQuery>();
}

void uninitialize_mvsqlite_module(ModuleInitializationLevel p_level) {
  if (p_level != MODULE_INITIALIZATION_LEVEL_SERVERS) {
    return;
  }
}
