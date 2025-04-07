# godot-duckdb

This C++ module provides a custom wrapper that makes the OLAP SQL engine [DuckDB](https://duckdb.org) available in Godot 4.0+. It is heavily inspired by the [godot-duckdb](https://github.com/mrjsj/godot-duckdb) wrapper. **A lot** of the code base (even this README.md!) from the godot-duckdb repository has been used in this project.

# How to use?

Built-in functionality is very limited at the moment, however, it is possible to fully utilize DuckDB, as everything from installing duckdb-extensions to storing secrets can be controlled with SQL queries.

```gdscript
var db = GDDuckDB.new()
# db.set_path("path/to/db") # Optionally set the path to the database; otherwise, opens an in-memory database
# db.set_read_only(true) # Optionally set read-only to true/false. Default is false.
db.open_db()
db.connect()

sql_query = "SELECT 'Hello, world!' AS msg"

db.query(sql_query)

var result = db.get_query_result()
print(result)
# prints "[{ "msg": "Hello, world!" }]"

db.disconnect()
db.close_db()
```

## Variables

-   **query_result** (Array, default=[])

    Contains the results from the latest query **by value**; meaning that this property is safe to use when executing successive queries as it does not get overwritten by any future queries.

## Functions

-   Boolean success = **open_db()**

-   Boolean success = **close_db()**

-   Boolean success = **connect()**

-   Boolean success = **disconnect()**

-   Boolean success = **query(** String sql_string **)**

-   Boolean success = **set_path(** String path **)**

-   String path = **get_path()**

-   Boolean success = **set_read_only(** bool read_only **)**

-   Boolean read_only = **get_read_only()**
