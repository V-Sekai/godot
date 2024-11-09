extends Node

func _ready() -> void:
	solve()

func solve():
	var bimdf : MinimumDeviationFlow = MinimumDeviationFlow.new()
	var state : MinimumDeviationFlowState = MinimumDeviationFlowState.new()

	var x = bimdf.add_node(state)
	var a = bimdf.add_node(state)
	var b = bimdf.add_node(state)
	var c = bimdf.add_node(state)

	bimdf.add_edge_zero(state, x, x, 0, 1, false, false)
	bimdf.add_edge_quad(state, x, a, 4, 1, 1, 5, true, true)
	bimdf.add_edge_abs(state, a, b, 0.7, 1, 0, 2, false, false)
	bimdf.add_edge_abs(state, a, c, 0.4, 1, 0, 2, false, false)
	bimdf.add_edge_abs(state, b, c, 0.2, 1, 0, 2, true, true)
	bimdf.solve(state)
