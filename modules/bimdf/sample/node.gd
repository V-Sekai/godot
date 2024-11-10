extends Node

func _ready() -> void:
	solve()

func solve():
	var mdf = MinimumDeviationFlow.new()
	var start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4]
	var end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2]
	var capacities = [15, 8, 20, 4, 10, 15, 4, 20, 5]
	var unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3]
	var supplies = [20, 0, 0, -5, -15]

	for i in range(len(supplies)):
		mdf.add_node(str(i), supplies[i])
	
	for i in range(len(start_nodes)):
		var start_index = mdf.get_node_index(str(start_nodes[i]))
		var end_index = mdf.get_node_index(str(end_nodes[i]))
		mdf.add_edge_abs(start_index, end_index, unit_costs[i], 1, 0, capacities[i], true, false)

	var start_index = mdf.get_node_index("0")
	var end_index = mdf.get_node_index("2")
	mdf.add_edge_abs(start_index, end_index, 0, 1, 0, 20, false, true)

	mdf.solve()
	
# output:
# graph TD
#     0 -->|0_1 0/15 target: 4 weight: 1| 1
#     0 -->|0_2 0/8 target: 4 weight: 1| 2
#     1 -->|1_2 0/20 target: 2 weight: 1| 2
#     1 -->|1_3 0/4 target: 2 weight: 1| 3
#     1 -->|1_4 0/10 target: 6 weight: 1| 4
#     2 -->|2_3 1/15 target: 1 weight: 1| 3
#     2 -->|2_4 3/4 target: 3 weight: 1| 4
#     3 -->|3_4 1/20 target: 2 weight: 1| 4
#     4 -->|4_2 4/5 target: 3 weight: 1| 2
#
# Textbook answer.
# Minimum cost: 150
#
#   Arc    Flow / Capacity  Cost
# 0 -> 1    12  /  15        48
# 0 -> 2     8  /   8        32
# 1 -> 2     8  /  20        16
# 1 -> 3     4  /   4         8
# 1 -> 4     0  /  10         0
# 2 -> 3    12  /  15        12
# 2 -> 4     4  /   4        12
# 3 -> 4    11  /  20        22
# 4 -> 2     0  /   5         0