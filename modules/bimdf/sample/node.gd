extends Node

func _ready() -> void:
    solve()

func solve():
    var mdf = MinimumDeviationFlow.new()
    var taxis = ["Taxi1", "Taxi2", "Taxi3"]
    var passengers = ["Passenger1", "Passenger2", "Passenger3", "Passenger4"]
    var preferences = [
        [4, 3, 2, 1],  # Taxi1
        [1, 2, 3, 4],  # Taxi2
        [2, 3, 4, 1],  # Taxi3
    ]
    
    for taxi in taxis:
        mdf.add_node(taxi)

    for passenger in passengers:
        mdf.add_node(passenger)

    var source = mdf.add_node("source")
    var sink = mdf.add_node("sink")
    
    # Connect source to taxis with capacity 1
    for i in range(len(taxis)):
        mdf.add_edge_abs(source, i, 1, 0, 0, 1, true, false)

    # Connect passengers to sink with capacity 1
    for j in range(len(passengers)):
        mdf.add_edge_abs(len(taxis) + j, sink, 1, 0, 0, 1, true, false)

    # Normalize preferences so that the sum of preferences[i][j] for each taxi is 1
    for i in range(len(taxis)):
        var total_preference = 0.0
        for j in range(len(passengers)):
            total_preference += preferences[i][j]
        for j in range(len(passengers)):
            preferences[i][j] /= total_preference

    # Connect taxis to passengers with normalized preferences
    for i in range(len(taxis)):
        for j in range(len(passengers)):
            mdf.add_edge_abs(i, len(taxis) + j, preferences[i][j], 1, 0, 1, true, false)

    mdf.solve()