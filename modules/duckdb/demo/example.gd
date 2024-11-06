extends Node

var db : GDDuckDB

func _ready():
	
	db = GDDuckDB.new()
	db.open_db()
	db.connect()
	
	var sql_query = "SELECT 'Hello, world!' as msg;"
	db.query(sql_query)
	
	var result = db.get_query_result()
	print(result)
	
	var data_query = """
	-- Create a new table with the specified columns
	CREATE TABLE sample_table (
		id UBIGINT,
		flag BOOLEAN,
		quantity INT,
		price FLOAT,
		event_time TIMESTAMP,
		description STRING,
		my_list TIME
	);

	-- Insert some sample rows into the table
	INSERT INTO sample_table (id, flag, quantity, price, event_time, description, my_list) VALUES
	(1, true, 10, 19.99, CURRENT_TIMESTAMP, 'First item description', CURRENT_TIME),
	(2, false, 5, 29.99, '2023-04-01T14:00:00.000Z', 'Second item description',CURRENT_TIME),
	(3, true, 20, 9.99, '2023-04-01T14:00:00.000Z', 'Third item description', CURRENT_TIME),
	(4, false, 15, 49.99, '2023-04-01T14:00:00.000Z', 'Fourth item description', CURRENT_TIME);
	
	SELECT * FROM sample_table;
	"""
	db.query(data_query)
	var data_result = db.get_query_result()
	
	print(data_result)

	db.disconnect()
	db.close_db()
