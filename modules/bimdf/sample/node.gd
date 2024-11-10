extends Node

func _ready() -> void:
	solve()

func solve():
	var bimdf : MinimumDeviationFlow = MinimumDeviationFlow.new()
	var x = bimdf.add_node("x")
	var a = bimdf.add_node("a")
	var b = bimdf.add_node("b")
	var c = bimdf.add_node("c")

	bimdf.add_edge_zero(x, x, 0, 1, false, false)
	bimdf.add_edge_quad(x, a, 4, 1, 1, 5, true, true)
	bimdf.add_edge_abs(a, b, 0.7, 1, 0, 2, false, false)
	bimdf.add_edge_abs(a, c, 0.4, 1, 0, 2, false, false)
	bimdf.add_edge_abs(b, c, 0.2, 1, 0, 2, true, true)
	bimdf.solve()
	bimdf.clear()
