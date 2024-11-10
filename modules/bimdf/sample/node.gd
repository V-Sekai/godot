extends Node

func _ready() -> void:
	solve()
	
func solve():
	# Suppose that a taxi firm has one taxi (the agent) available, and one customer (the task) wishing to be picked up as soon as possible. The firm prides itself on speedy pickups, so for the taxi the "cost" of picking up the customer will depend on the time taken for the taxi to reach the pickup point. This is a balanced assignment problem. Its solution is whichever combination of taxi and customer results in the least total cost.
	
	var taxis = ["Taxi1"]
	var customers = ["Customer1"]
	var travel_times = [
		[10]  # Taxi1
	]

	var taxi_nodes = []
	var customer_nodes = []

	var mdf = MinimumDeviationFlow.new()

	for taxi in taxis:
		taxi_nodes.append(mdf.add_node(taxi))

	for customer in customers:
		customer_nodes.append(mdf.add_node(customer))

	var source = mdf.add_node("source")
	var sink = mdf.add_node("sink")

	var cost = 0
	var capacity = 1
	var lower_bound = 0
	var upper_bound = 1
	var u_tail = false
	var v_tail = false

	for taxi_node in taxi_nodes:
		mdf.add_edge_abs(source, taxi_node, cost, capacity, lower_bound, upper_bound, u_tail, v_tail)

	for customer_node in customer_nodes:
		mdf.add_edge_abs(customer_node, sink, cost, capacity, lower_bound, upper_bound, u_tail, v_tail)

	cost = travel_times[0][0] if travel_times[0][0] != 0 else 0
	u_tail = true
	v_tail = true
	mdf.add_edge_abs(taxi_nodes[0], customer_nodes[0], cost, capacity, lower_bound, upper_bound, u_tail, v_tail)

	mdf.solve()
