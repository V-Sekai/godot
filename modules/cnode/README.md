# Godot CNode

Erlang/Elixir CNode interface for Godot Engine. This allows Elixir/Erlang nodes to communicate with Godot using the Erlang distribution protocol.

## Building Without Mix

The CNode can be built **standalone without requiring mix or Elixir**. There are two ways to build:

### Method 1: Direct SCons Build (Recommended)

```bash
# From the Godot project root
scons library_type=executable extra_suffix=cnode dev_build=yes debug_symbols=yes
```

This will build the CNode executable to `bin/godot.{platform}.{target}.dev.{arch}.cnode`

### Method 2: Using the Helper Script

```bash
# From the Godot project root
python3 modules/cnode/build_cnode.py
```

This script will:
1. Build Godot executable (if needed)
2. Generate extension API
3. Build Godot static library (if needed)
4. Build the CNode executable


## Prerequisites

- SCons 4.0+
- Python 3.8+
- Erlang/OTP with erl_interface library (in `thirdparty/erl_interface`)
- Godot static library (built with `library_type=static_library`)

**Note:** The CNode uses native Godot headers directly (libgodot API), not godot-cpp. No godot-cpp dependency is required.


## Running the CNode

After building, run the CNode:

```bash
./bin/godot.{platform}.{target}.dev.{arch}.cnode -name godot@127.0.0.1 -setcookie godotcookie
```

## API

The CNode supports the following RPC calls:

- `ping` - Health check
- `godot_version` - Get Godot version
- `create_instance` - Create a Godot instance
- `start_instance` - Start a Godot instance
- `stop_instance` - Stop a Godot instance
- `destroy_instance` - Destroy a Godot instance
- `iteration` - Run one frame iteration
- `get_scene_tree_root` - Get the root node
- `find_node` - Find a node by path
- `get_current_scene` - Get the current scene
- `get_node_property` - Get a node property
- `set_node_property` - Set a node property
- `call_node_method` - Call a method on a node
- `get_node_children` - Get children of a node
- `get_node_class` - Get node's class name

All Variant data is encoded/decoded using BERT (Binary ERlang Term) format.

