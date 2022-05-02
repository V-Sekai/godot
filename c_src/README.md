# Unifex Godot Engine

```bash
sudo apt install -y elixir erlang-dev
mix deps.get
mix compile
```

Launch.

Create a script in the fs.


```
extends SceneTree

func _init():
	var gltf_doc = GLTFDocument.new()
	var packed_scene = PackedScene.new()
	var node = gltf_doc.import_scene("Fox.glb")
	packed_scene.pack(node)
	print("test")
	ResourceSaver.save("saved.tscn", packed_scene)
	quit()
```

```elixir
# iex -S mix
iex -S mix run --no-compile
{:ok, pid} = Godot.start_link(["--headless --path ../../workspace/groups --editor"])
GenServer.call(pid, {:call, ["get_method_list", :nil, "", :string, "", :string, "", :string, "", :string, ""]}) 
GenServer.call(pid, {:call, ["get_node_count", :nil, "", :string, "", :string, "", :string, "", :string, ""]}) 
GenServer.stop(pid)
```