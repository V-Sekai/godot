#include "gdduckdb.h"

#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>


using namespace godot;
using namespace std;


void GDDuckDB::_bind_methods() {
        // Bind the select_from_table method to be callable from Godot

        ClassDB::bind_method(D_METHOD("open_db"), &GDDuckDB::open_db);
        ClassDB::bind_method(D_METHOD("close_db"), &GDDuckDB::close_db);
        ClassDB::bind_method(D_METHOD("connect"), &GDDuckDB::connect);
        ClassDB::bind_method(D_METHOD("disconnect"), &GDDuckDB::disconnect);
        ClassDB::bind_method(D_METHOD("query", "sql_query"), &GDDuckDB::query);

        ClassDB::bind_method(D_METHOD("set_read_only", "read_only"), &GDDuckDB::set_read_only);
        ClassDB::bind_method(D_METHOD("get_read_only"), &GDDuckDB::get_read_only);
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "read_only"), "set_read_only", "get_read_only");

        ClassDB::bind_method(D_METHOD("set_path", "path"), &GDDuckDB::set_path);
        ClassDB::bind_method(D_METHOD("get_path"), &GDDuckDB::get_path);
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "path"), "set_path", "get_path");

        ClassDB::bind_method(D_METHOD("set_threads", "threads"), &GDDuckDB::set_threads);
        ClassDB::bind_method(D_METHOD("get_threads"), &GDDuckDB::get_threads);
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "threads"), "set_threads", "get_threads");     

        ClassDB::bind_method(D_METHOD("set_max_memory", "max_memory"), &GDDuckDB::set_max_memory);
        ClassDB::bind_method(D_METHOD("get_max_memory"), &GDDuckDB::get_max_memory);
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "max_memory"), "set_max_memory", "get_max_memory");     

        ClassDB::bind_method(D_METHOD("set_default_order", "default_order"), &GDDuckDB::set_default_order);
        ClassDB::bind_method(D_METHOD("get_default_order"), &GDDuckDB::get_default_order);
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "default_order"), "set_default_order", "get_default_order");                             
        
        // TODO: Implement query_chunk
        //ClassDB::bind_method(D_METHOD("query_chunk", "sql_query"), &GDDuckDB::query_chunk);

        ClassDB::bind_method(D_METHOD("set_query_result", "query_result"), &GDDuckDB::set_query_result);
        ClassDB::bind_method(D_METHOD("get_query_result"), &GDDuckDB::get_query_result);
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "query_result", PROPERTY_HINT_ARRAY_TYPE, "Dictionary"), "set_query_result", "get_query_result");
}

GDDuckDB::GDDuckDB() {
        db = nullptr;
        con = nullptr;
        query_result = TypedArray<Dictionary>();
}

GDDuckDB::~GDDuckDB() {
        /* Clean up the function_registry */
        function_registry.clear();
        function_registry.shrink_to_fit();
        /* Close the database connection if it is still open */
        if (con) {
            disconnect();
            if (db) {
                close_db();
            }
        }
        
}

bool GDDuckDB::is_connection_valid() {
    if (!db) {
        UtilityFunctions::printerr("GDDuckDB Error: Database must be opened. Hint: Use `open_db()`");
        return false;
    }

    if (!con) {
        UtilityFunctions::printerr("GDDuckDB Error: Connection must be made be opened. Hint: Use `connect()`");
        return false;
    }
    return true;
}

bool GDDuckDB::query_chunk(const String &sql_query) {
        if (!is_connection_valid()){
            return false;
        }

        duckdb_result result;
        duckdb_state status = duckdb_query(con, sql_query.utf8(), &result);

        if (status == DuckDBError) {
            UtilityFunctions::printerr("GDDuckDB Error: Failed to run query: " + String(duckdb_result_error(&result)));
            duckdb_destroy_result(&result);
            return false;
        }

        idx_t column_count = duckdb_column_count(&result);
        idx_t row_count = duckdb_row_count(&result);
        query_result.clear(); // Clear previous results

        auto chunk_count = duckdb_result_chunk_count(result);

        Dictionary data_dict;
        

        Dictionary col_dict;
        for (idx_t col_idx = 0; col_idx < column_count; ++col_idx) {
            //const char *col_name = duckdb_column_name(&result, col_idx);
            duckdb_type col_type = duckdb_column_type(&result, col_idx);
            col_dict[col_idx] = col_type;
        }

        for (idx_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
            duckdb_data_chunk chunk = duckdb_result_get_chunk(result, chunk_idx);
            for (idx_t col_idx = 0; col_idx < column_count; ++col_idx) {

                const char *col_name = duckdb_column_name(&result, col_idx);
                UtilityFunctions::print(col_name);

                duckdb_vector vector = duckdb_data_chunk_get_vector(chunk, col_idx);

                duckdb_logical_type col_type = duckdb_vector_get_column_type(vector);


                int64_t* col_value = (int64_t*)duckdb_vector_get_data(vector);

                data_dict[String(col_name)] = col_value;
                query_result.push_back(data_dict);

                UtilityFunctions::print(col_value);
            }
        }
        return true;
}

bool GDDuckDB::query(const String &sql_query) {
        
        if (!is_connection_valid()){
            return false;
        }

        duckdb_result result;
        duckdb_state status = duckdb_query(con, sql_query.utf8(), &result);

        if (status == DuckDBError) {
            UtilityFunctions::printerr("GDDuckDB Error: Failed to run query: " + String(duckdb_result_error(&result)));
            duckdb_destroy_result(&result);
            return false;
        }

        idx_t column_count = duckdb_column_count(&result);
        idx_t row_count = duckdb_row_count(&result);
        query_result.clear(); // Clear previous results


        for (idx_t row_idx = 0; row_idx < row_count; ++row_idx) {
            Dictionary row_dict;

            for (idx_t col_idx = 0; col_idx < column_count; ++col_idx) {

                const char *col_name = duckdb_column_name(&result, col_idx);
        
                Variant col_value = map_duckdb_type_to_godot_variant(result, col_idx, row_idx);

                

                row_dict[String(col_name)] = col_value;
            }

            query_result.append(row_dict);
        }

        duckdb_destroy_result(&result);
        return true;
}

Variant GDDuckDB::map_duckdb_type_to_godot_variant(duckdb_result &result, idx_t col_idx, idx_t row_idx){

        duckdb_type col_type = duckdb_column_type(&result, col_idx);
        
        if (duckdb_value_is_null(&result, col_idx, row_idx)) {
            return Variant(); // Assign null
        } 

        Variant col_value;
        switch (col_type) {
            case DUCKDB_TYPE_BOOLEAN:
                col_value = Variant((bool)duckdb_value_boolean(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_TINYINT:
                col_value = Variant((int8_t)duckdb_value_int8(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_UTINYINT:
                col_value = Variant((uint8_t)duckdb_value_uint8(&result, col_idx, row_idx));
                break;                    
            case DUCKDB_TYPE_SMALLINT:
                col_value = Variant((int16_t)duckdb_value_int16(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_USMALLINT:
                col_value = Variant((uint16_t)duckdb_value_uint16(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_INTEGER:
                col_value = Variant((int32_t)duckdb_value_int32(&result, col_idx, row_idx));    
                break;
            case DUCKDB_TYPE_UINTEGER:
                col_value = Variant((uint32_t)duckdb_value_uint32(&result, col_idx, row_idx));    
                break;                    
            case DUCKDB_TYPE_BIGINT:
                col_value = Variant((int64_t)duckdb_value_int64(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_UBIGINT:
                col_value = Variant((uint64_t)duckdb_value_uint64(&result, col_idx, row_idx));
                break;                    
            case DUCKDB_TYPE_FLOAT:
                col_value = Variant((float)duckdb_value_float(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_DOUBLE:
                col_value = Variant((double_t)duckdb_value_double(&result, col_idx, row_idx));
                break;
            case DUCKDB_TYPE_DECIMAL:
                col_value = Variant((double_t)duckdb_decimal_to_double(duckdb_value_decimal(&result, col_idx, row_idx)));
                break;
            case DUCKDB_TYPE_DATE:
            case DUCKDB_TYPE_TIME:
            case DUCKDB_TYPE_TIMESTAMP:
            case DUCKDB_TYPE_INTERVAL:
            case DUCKDB_TYPE_VARCHAR:
                {
                    char *str_value = duckdb_value_string(&result, col_idx, row_idx).data;
                    col_value = Variant(String::utf8(str_value));
                    duckdb_free(str_value);
                }
                break;
            case DUCKDB_TYPE_BLOB:
            case DUCKDB_TYPE_TIMESTAMP_TZ:
            case DUCKDB_TYPE_ENUM:
            case DUCKDB_TYPE_LIST:
            case DUCKDB_TYPE_MAP:
            case DUCKDB_TYPE_STRUCT:
            case DUCKDB_TYPE_INVALID:
            default:                       
                UtilityFunctions::printerr("Unhandled column type: (" + String(std::to_string(col_type).c_str()) + ") " + String(duckdb_type_to_string(col_type)));
                break;
        }

        return col_value;
}

const char* GDDuckDB::duckdb_type_to_string(duckdb_type type) {
        switch (type) {
            case DUCKDB_TYPE_INVALID: return "DUCKDB_TYPE_INVALID";
            case DUCKDB_TYPE_BOOLEAN: return "DUCKDB_TYPE_BOOLEAN";
            case DUCKDB_TYPE_TINYINT: return "DUCKDB_TYPE_TINYINT";
            case DUCKDB_TYPE_SMALLINT: return "DUCKDB_TYPE_SMALLINT";
            case DUCKDB_TYPE_INTEGER: return "DUCKDB_TYPE_INTEGER";
            case DUCKDB_TYPE_BIGINT: return "DUCKDB_TYPE_BIGINT";
            case DUCKDB_TYPE_UTINYINT: return "DUCKDB_TYPE_UTINYINT";
            case DUCKDB_TYPE_USMALLINT: return "DUCKDB_TYPE_USMALLINT";
            case DUCKDB_TYPE_UINTEGER: return "DUCKDB_TYPE_UINTEGER";
            case DUCKDB_TYPE_UBIGINT: return "DUCKDB_TYPE_UBIGINT";
            case DUCKDB_TYPE_FLOAT: return "DUCKDB_TYPE_FLOAT";
            case DUCKDB_TYPE_DOUBLE: return "DUCKDB_TYPE_DOUBLE";
            case DUCKDB_TYPE_DECIMAL: return "DUCKDB_TYPE_DECIMAL";
            case DUCKDB_TYPE_TIMESTAMP: return "DUCKDB_TYPE_TIMESTAMP";
            case DUCKDB_TYPE_DATE: return "DUCKDB_TYPE_DATE";
            case DUCKDB_TYPE_TIME: return "DUCKDB_TYPE_TIME";
            case DUCKDB_TYPE_INTERVAL: return "DUCKDB_TYPE_INTERVAL";
            case DUCKDB_TYPE_VARCHAR: return "DUCKDB_TYPE_STRING";
            case DUCKDB_TYPE_BLOB: return "DUCKDB_TYPE_BLOB";
            case DUCKDB_TYPE_ENUM: return "DUCKDB_TYPE_ENUM";
            case DUCKDB_TYPE_LIST: return "DUCKDB_TYPE_LIST";
            case DUCKDB_TYPE_MAP: return "DUCKDB_TYPE_MAP";
            case DUCKDB_TYPE_STRUCT: return "DUCKDB_TYPE_STRUCT";
            case DUCKDB_TYPE_TIMESTAMP_TZ: return "DUCKDB_TYPE_TIMESTAMP_TZ";
            // Add other cases as needed
            default: return "UNKNOWN_TYPE";
        }
}

bool GDDuckDB::connect() {
        if (db) {

            duckdb_state status = duckdb_connect(db, &con);
            if (status == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to open database.");
                return false;
            }
            return true;
        }

        UtilityFunctions::printerr("GDDuckDB Error: Can't connect if database is not opened. Hint: Use `open_db` first.");
        return false;
}


bool GDDuckDB::open_db() {

        duckdb_config config;
        char *err;

        if (duckdb_create_config(&config) == DuckDBError) {
            UtilityFunctions::printerr("GDDuckDB Error: Failed to create database configuration.");
            return false;
        }

        if (read_only == true) {
            if (duckdb_set_config(config, "access_mode", "READ_ONLY") == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: access_mode = READ_ONLY");
                return false;
            }
        } else {
            if (duckdb_set_config(config, "access_mode", "READ_WRITE") == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: access_mode = READ_WRITE");
                return false;
            }
        }

        if (threads) {
            if (duckdb_set_config(config, "threads", threads) == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: threads = " + String(threads));
                return false;
            }
        }

        if (max_memory) {
            if (duckdb_set_config(config, "max_memory", max_memory) == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: max_memory = " + String(max_memory));
                return false;
            }
        }

        if (default_order) {
            if (duckdb_set_config(config, "default_order", default_order) == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: default_order = " + String(default_order));
                return false;
            }
        }        

        duckdb_state open_state = duckdb_open_ext(path, &db, config, &err);
        if (open_state == DuckDBError) {
            UtilityFunctions::printerr("GDDuckDB Error: Failed to open database. Error: " + String(err));
            duckdb_destroy_config(&config);
            duckdb_free(err);
            return false;
        }
        duckdb_destroy_config(&config);
        return true;
}


void GDDuckDB::set_query_result(const TypedArray<Dictionary> &p_query_result) {
	    query_result = p_query_result;
}


TypedArray<Dictionary> GDDuckDB::get_query_result() const {
	    return query_result.duplicate(true);
}


bool GDDuckDB::disconnect() {
        if (!con) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't disconnect when no connection is connected!");
            return false;
        }
        duckdb_disconnect(&con);
        return true;
}


bool GDDuckDB::close_db() {
        if (con) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't close database if connection is still connected!");
            return false;
        }

        if (!db) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't close database when no database is currently open!");
            return false;
        }

        duckdb_close(&db);
        return true;
}


bool GDDuckDB::set_read_only(const bool &_read_only) {
        if (db) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't set read_only when database is already open!");
            return false;
        }
        read_only = _read_only;
        return true;
}

bool GDDuckDB::get_read_only() const {
        return read_only;
}

bool GDDuckDB::set_path(const String &_path) {
        if (db) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't set path when database is already open!");
            return false;
        }
        path_utf8 = _path.utf8();
        path = path_utf8;
        return true;
}

String GDDuckDB::get_path() const {
        return path;
}

bool GDDuckDB::set_threads(const String &_threads) {

        duckdb_result result;

        if (db && con) {
            String sql_query = "SET threads TO " + _threads;
            if (duckdb_query(con, sql_query.utf8(), &result) == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: threads = " + String(_threads));
                duckdb_destroy_result(&result);
                return false;
            }
            duckdb_destroy_result(&result);
        }

        if (db && !con) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't set database configuration: threads = " + String(_threads) + " when no connection is connected!");
            return false;
        }

        threads_utf8 = _threads.utf8();
        threads = threads_utf8;
        return true;
}

String GDDuckDB::get_threads() const {
        return threads;
}

bool GDDuckDB::set_max_memory(const String &_max_memory) {

        duckdb_result result;

        if (db && con) {
            String sql_query = "SET memory_limit = '" + _max_memory + "'";
            if (duckdb_query(con, sql_query.utf8(), &result) == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: memory_limit = " + String(_max_memory));
                duckdb_destroy_result(&result);
                return false;
            }
            duckdb_destroy_result(&result);
        }

        if (db && !con) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't set database configuration: memory_limit = " + String(_max_memory) + " when no connection is connected!");
            return false;
        }


        max_memory_utf8 = _max_memory.utf8();
        max_memory = max_memory_utf8;
        return true;
}

String GDDuckDB::get_max_memory() const {
        return max_memory;
}

bool GDDuckDB::set_default_order(const String &_default_order) {

        duckdb_result result;

        if (db && con) {
            String sql_query = "SET default_order = '" + _default_order.to_upper() + "'";
            if (duckdb_query(con, sql_query.utf8(), &result) == DuckDBError) {
                UtilityFunctions::printerr("GDDuckDB Error: Failed to set database configuration: default_order = " + String(_default_order));
                duckdb_destroy_result(&result);
                return false;
            }
            duckdb_destroy_result(&result);
        }

        if (db && !con) {
            UtilityFunctions::printerr("GDDuckDB Error: Can't set database configuration: default_order = " + String(_default_order) + " when no connection is connected!");
            return false;
        }

        default_order_utf8 = _default_order.utf8();
        default_order = default_order_utf8;
        return true;
}

String GDDuckDB::get_default_order() const {
        return default_order;
}