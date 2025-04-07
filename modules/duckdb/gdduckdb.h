/**************************************************************************/
/*  gdduckdb.h                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "modules/duckdb/thirdparty/duckdb/duckdb.h"

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

class GDDuckDB : public RefCounted {
	GDCLASS(GDDuckDB, RefCounted);

private:
	duckdb_database db;
	duckdb_connection con;
	TypedArray<Dictionary> query_result = TypedArray<Dictionary>();

	std::vector<std::unique_ptr<Callable>> function_registry;
	int64_t verbosity_level = 1;

	bool read_only = false;
	const char *threads = nullptr;
	const char *max_memory = nullptr;
	const char *default_order = nullptr;

	CharString path_utf8;
	CharString threads_utf8;
	CharString max_memory_utf8;
	CharString default_order_utf8;

	const char *path = ":memory:";

	const char *duckdb_type_to_string(duckdb_type type);
	Variant map_duckdb_type_to_godot_variant(duckdb_result &result, idx_t col_idx, idx_t row_idx);
	bool is_connection_valid();

protected:
	static void _bind_methods();

public:
	enum VerbosityLevel {
		QUIET = 0,
		NORMAL = 1,
		VERBOSE = 2,
		VERY_VERBOSE = 3
	};

	GDDuckDB();
	~GDDuckDB();

	// Methods
	bool open_db();
	bool close_db();
	bool open_connection();
	bool close_connection();
	bool query(const String &sql_query);
	bool query_chunk(const String &sql_query);

	void set_query_result(const TypedArray<Dictionary> &p_query_result);

	TypedArray<Dictionary> get_query_result() const;

	// Configurations
	bool set_read_only(const bool &_read_only);
	bool get_read_only() const;

	bool set_threads(const String &_threads);
	String get_threads() const;

	bool set_max_memory(const String &_max_memory);
	String get_max_memory() const;

	bool set_default_order(const String &_default_order);
	String get_default_order() const;

	// Properties
	bool set_path(const String &_path);
	String get_path() const;
};
