extends Node

class PlayerState:
	var id : String
	var state : PackedByteArray
	
var sql: String
var db: MVSQLite

var insert_query: MVSQLiteQuery = null
var player
var timer: Timer = Timer.new()

var player_count = 1


func _ready() -> void:
	player = PlayerState.new()
	player.id = generate_uuid()
	player.state = PackedByteArray()
	
	var http_request: HTTPRequest = HTTPRequest.new()
	add_child(http_request)
	http_request.connect("request_completed", _on_request_completed)
	var err = http_request.request("http://localhost:7001/api/create_namespace",
								   ["Content-Type: application/json"],
								   HTTPClient.METHOD_POST,
								   '{"key":"' + "player_server_01" + '"}')
	if err == OK:
		print("Database created")

	db = MVSQLite.new()
	if not db.open("player_server_01"):
		return
	var query = db.create_query("DROP TABLE IF EXISTS players")
	query.execute()
	query = db.create_query("CREATE TABLE IF NOT EXISTS players (id TEXT PRIMARY KEY, state BLOB, tick INTEGER) WITHOUT ROWID, STRICT")
	query.execute()
	var placeholders: PackedStringArray
	placeholders.resize(player_count)
	placeholders.fill("(?, ?, ?)")
	sql = "INSERT INTO players(id, state, tick) VALUES " + ", ".join(placeholders) + " ON CONFLICT(id) DO UPDATE SET state = excluded.state, tick = excluded.tick RETURNING *"
	insert_query = db.create_query(sql)
	
var crypto: Crypto = Crypto.new()

func _process(_delta):
	var states: Array
	for i in range(player_count):
		player.state = crypto.generate_random_bytes(100)
		states.append_array([player.id, player.state, Engine.get_process_frames()])
	var results = insert_query.execute(states)
	if !results:
		return
	var error = insert_query.get_last_error_message()
	if error != "not an error":
		print(error)

func _on_request_completed(_result, response_code, _headers, body):
	var string = body.get_string_from_utf8()
	if response_code == 200:
		print("Request completed successfully")
	else:
		print("Request failed with response code: ", response_code)
		print("Response body: ", string)
		

func generate_uuid() -> String:
	var uuid: PackedByteArray = crypto.generate_random_bytes(16)
	return "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x" % [
		uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
		uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
	]

func _exit_tree():
	db.close()
