# Plan Explanation and Debugging Demo

This demo showcases the plan explanation and debugging features of the goal_task_planner module.

## Features Demonstrated

1. **Plan Explanation** - Understand why a plan was chosen
2. **Node Explanations** - Get detailed explanations for individual nodes
3. **Alternative Methods** - See what other methods were considered
4. **Decision Path** - Trace the decision path from root to any node
5. **Graph Visualization** - Export plans as DOT and JSON for visualization

## Domain

The demo uses a simple "go to store and buy item" domain with:
- **Actions**: `walk` (move between locations), `buy` (purchase items)
- **Tasks**: `go_to` (with two methods: direct path or via landmark), `buy` (purchase an item)

## Running the Demo

### Option 1: Run as a Godot Script

```bash
godot --headless --script modules/goal_task_planner/tests/domains/plan_explanation_demo/test.gd
```

### Option 2: Run in Godot Editor

1. Open the project in Godot
2. Create a new scene with a Node
3. Attach the `test.gd` script to the node
4. Run the scene

## Output

The demo will:
1. Create a plan to go to the store and buy an apple
2. Display plan explanation with statistics
3. Show node explanations for key decision points
4. List alternative methods that were considered
5. Display the decision path
6. Export the plan graph to:
   - `user://plan_graph.dot` - DOT format (for Graphviz)
   - `user://plan_graph.json` - JSON format (for web visualization)

## Visualizing the Graph

### Using Graphviz (DOT format)

```bash
# Install Graphviz if needed
# macOS: brew install graphviz
# Linux: sudo apt-get install graphviz
# Windows: Download from https://graphviz.org/

# Render to PNG
dot -Tpng user://plan_graph.dot -o plan_graph.png

# Render to SVG
dot -Tsvg user://plan_graph.dot -o plan_graph.svg

# Render to PDF
dot -Tpdf user://plan_graph.dot -o plan_graph.pdf
```

### Using Web Visualization (JSON format)

The JSON output can be used with visualization libraries:

- **D3.js**: Use the JSON with D3's force-directed graph layout
- **Cytoscape.js**: Load the JSON directly into Cytoscape
- **vis.js**: Use with vis-network for interactive graphs

Example with Cytoscape.js:
```javascript
fetch('plan_graph.json')
  .then(response => response.json())
  .then(data => {
    cytoscape({
      container: document.getElementById('cy'),
      elements: [
        ...data.nodes.map(n => ({ data: n })),
        ...data.edges.map(e => ({ data: e }))
      ]
    });
  });
```

## Graph Format

### DOT Format

The DOT graph includes:
- **Node colors**: Different colors for different node types (Root, Action, Task, Goal)
- **Status indicators**: Green (completed), Red (failed), Yellow (open)
- **Node labels**: Node ID and information
- **Edges**: Parent-child relationships

### JSON Format

The JSON includes:
- `nodes`: Array of node objects with id, type, status, info, decision_info
- `edges`: Array of edge objects with from/to node IDs
- `success`: Whether planning succeeded

## Example Output

```
=== Plan Explanation and Debugging Demo ===

Initial State:
  Location: home
  Money: 20
  Inventory: []

Goal: [["go_to", "store"], ["buy", "apple", "store"]]

✓ Plan found successfully!

Plan Actions:
  1.  ["walk", "home", "store"]
  2.  ["buy", "apple", "store"]

==================================================
1. PLAN EXPLANATION
==================================================
Success: true
Total Nodes: 6
Plan Length: 2 actions

Node Counts:
  Actions: 2
  Tasks: 2
  Failed: 0
  Closed: 4

Decision Points: 2
  1. Node 1: ["go_to", "store"]
     Selected: method_go_to_direct
     Score: 10.5
     Alternatives: 2
...

✓ DOT graph saved to: user://plan_graph.dot
  Render with: dot -Tpng [path] -o plan_graph.png
✓ JSON graph saved to: user://plan_graph.json
```
