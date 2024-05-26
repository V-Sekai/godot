extends Node3D


func _ready():
	var db : MVSQLite = MVSQLite.new();
	if (!db.open("mvsqlite")):
		print("Failed opening database.");
		return;
	var create_entity_table : String = """
CREATE TABLE IF NOT EXISTS entity (
	id TEXT PRIMARY KEY NOT NULL CHECK(LENGTH(id) = 36),
	user_data blob NOT NULL CHECK( LENGTH(user_data) = 16) DEFAULT (zeroblob(16)),
	reserved blob NOT NULL CHECK( LENGTH(reserved) = 48)  DEFAULT (zeroblob(48)),
	shard	INTEGER NOT NULL,
	code	INTEGER NOT NULL,
	flags	INTEGER	NOT NULL,
	past_pending	BLOB NOT NULL CHECK( LENGTH(past_pending) <= 1024) DEFAULT (zeroblob(64)),
	past_posted BLOB NOT NULL CHECK( LENGTH(past_posted) <= 1024) DEFAULT (zeroblob(64)),
	current_pending BLOB NOT NULL CHECK( LENGTH(current_pending) <= 1024) DEFAULT (zeroblob(64)),
	current_posted	BLOB NOT NULL CHECK( LENGTH(current_posted) <= 1024) DEFAULT (zeroblob(64)),
	timestamp INTEGER NOT NULL
) WITHOUT ROWID, STRICT;
"""
	var query: MVSQLiteQuery = db.create_query(create_entity_table)
	if query == null:
		print("Failed to create query.")
		return
	for i in range(32):
		var node_3d : Node3D = Node3D.new()
		var script = load("res://sqlite_write/sqlite_write_scene.gd")
		node_3d.set_script(script)
		add_child(node_3d, true)
		node_3d.owner = self

