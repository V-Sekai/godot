#ifndef DUCKDB_REGISTER_TYPES_H
#define DUCKDB_REGISTER_TYPES_H

#include "modules/register_module_types.h"

void initialize_duckdb_module(ModuleInitializationLevel p_level);
void uninitialize_duckdb_module(ModuleInitializationLevel p_level);

#endif // ! DUCKDB_REGISTER_TYPES_H