extends Node

func _ready() -> void:
	solve()
	
func solve():
	var start_nodes = [0, 0, 1, 1, 2, 2]
	var end_nodes = [1, 2, 3, 4, 3, 4]
	var capacities = [1, 1, 1, 1, 1, 1]
	var costs = [0, 0, 10, 0, 30, 0]

	var source = 0
	var sink = 5
	var tasks = 2
	var supplies = [tasks, 0, 0, 0, 0, -tasks]
	var job_names = ["Engineer", "Designer"]
	var student_names = ["Alice", "Bob"]

	var mdf = MinimumDeviationFlow.new()

	mdf.add_node("Source")

	for i in range(len(job_names)):
		mdf.add_node(job_names[i])

	for i in range(len(student_names)):
		mdf.add_node(student_names[i])

	mdf.add_node("Sink")

	for i in range(len(start_nodes)):
		mdf.add_edge_abs(start_nodes[i], end_nodes[i], costs[i], 1.0, 0, capacities[i], true, false)

	mdf.add_edge_abs(3, sink, 0, 1.0, 0, 1, true, false)
	mdf.add_edge_abs(4, sink, 0, 1.0, 0, 1, true, false)

	mdf.solve()
