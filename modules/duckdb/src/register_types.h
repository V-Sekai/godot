#ifndef DUCKDB_REGISTER_TYPES_H
#define DUCKDB_REGISTER_TYPES_H

#include <godot_cpp/core/class_db.hpp>
using namespace godot;

void initialize_duckdb_module(ModuleInitializationLevel p_level);
void uninitialize_duckdb_module(ModuleInitializationLevel p_level);

#endif // ! DUCKDB_REGISTER_TYPES_H