extends Node

func _ready() -> void:
	solve()

func solve():
	var mdf = MinimumDeviationFlow.new()
	
	var x = mdf.add_node("X")
	var a = mdf.add_node("A")
	var b = mdf.add_node("B")
	var c = mdf.add_node("C")
	
	mdf.add_edge_zero(x, x, false, false) 
	mdf.add_edge_quad(x, a, 4.0, 1, 1, 2147483647, true, true)
	mdf.add_edge_abs(a, b, 0.7, 0, 0, 2147483647, false, false)
	mdf.add_edge_abs(a, c, 0.4, 0, 0, 2147483647, false, false)
	mdf.add_edge_abs(b, c, 0.2, 0, 0, 2147483647, true, true)
	
	mdf.solve()