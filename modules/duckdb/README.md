

![gdduckdb](logo/gdduckdb-128x128.png?raw=true  "Godot DuckDB Icon")

# godot-duckdb

This GDNative script aims to serve as a custom wrapper that makes the OLAP SQL engine [DuckDB](https://duckdb.org) available in Godot 4.0+.
It is heavily inspired by the [godot-sqlite](https://github.com/2shady4u/godot-sqlite) wrapper. **Alot** of the code base (even this README.md!) from the godot-sqlite repository has been used in this project.

### Supported operating systems:
- Mac OS X (universal)
- Windows
- Linux

#### Planned supported operating systems
- Android (arm64)
- iOS (arm64)
- HTML5 (stability depends on browser)

# How to install?

Re-building Godot from scratch is **NOT** required, the proper way of installing this plugin is to either install it through the Asset Library or to just manually download the build files yourself.

### Godot Asset Library

**Godot-DuckDB** is available through the official Godot Asset Library, and can be installed in the following way:

- Click on the 'AssetLib' button at the top of the editor.
- Search for 'godot-duckdb' and click on the resulting element.
- In the dialog pop-up, click 'Download'.
- Once the download is complete, click on the install button...
- Once more, click on the 'Install' button.
- Activate the plugin in the 'Project Settings/Plugins'-menu.
- All done!

### Manually

It's also possible to manually download the build files found in the [releases](https://github.com/mrjsj/godot-duckdb/releases) tab, extract them on your system and run the supplied demo-project. Make sure that Godot is correctly loading the `gdduckdb.gdextension`-resource and that it is available in the `res://`-environment.

An example project, named "demo", can also be downloaded from the releases tab.

# How to use?

Built-in functionality is very limited at the moment, however, it is possible to fully utilize DuckDB, as everything from installing duckdb-extensions to storing secrets can be controlled with sql-queries.


```gdscript
var db = GDDuckDB.new()
# db.set_path("path/to/db") # Optionally set path to database, otherwise opens in-memory database
# db.set_read_only(true) # Optionally set read_only to true/false. Default is false.
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

- **query_result** (Array, default=[])

    Contains the results from the latest query **by value**; meaning that this property is safe to use when looping successive queries as it does not get overwritten by any future queries.


## Functions

- Boolean success = **open_db()**

- Boolean success = **close_db()**

- Boolean success = **connect()**

- Boolean success = **disconnect()**

- Boolean success = **query(** String sql_string **)**

- Boolean success = **set_path(** String path **)**

- String path = **get_path()**

- Boolean success = **set_read_only(** bool read_only **)**

- Boolean read_only = **get_read_only()**

