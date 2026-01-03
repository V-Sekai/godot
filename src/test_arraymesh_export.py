#!/usr/bin/env python3
"""
Test script for creating ArrayMesh and exporting as ESCN via MCP server.
This script tests the complete workflow:
1. Create MeshInstance3D node
2. Set SafeGDScript to create ArrayMesh (setting mesh directly, not @export)
3. Export as ESCN
"""

import json
import requests
import sys
import time

# MCP server URL (script server runs on port 8087)
MCP_SERVER_URL = "http://localhost:8087"

def make_mcp_request(method, params=None):
    """Make a JSON-RPC 2.0 request to the MCP server"""
    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000),
        "method": method
    }
    if params:
        payload["params"] = params
    
    try:
        response = requests.post(
            MCP_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}

def test_arraymesh_export():
    """Test the complete ArrayMesh creation and export workflow"""
    print("=" * 60)
    print("Testing ArrayMesh Creation and ESCN Export")
    print("=" * 60)
    
    # Step 1: Check if server is available
    print("\n[1/4] Checking MCP server availability...")
    tools_result = make_mcp_request("tools/list")
    if "error" in tools_result:
        print(f"❌ Server not available: {tools_result.get('error')}")
        print(f"   Make sure Godot MCP server is running on {MCP_SERVER_URL}")
        return False
    
    print("✅ Server is available")
    if "result" in tools_result and "tools" in tools_result["result"]:
        tools = tools_result["result"]["tools"]
        print(f"   Available tools: {', '.join([t.get('name', '') for t in tools])}")
    
    # Step 2: Create MeshInstance3D node
    print("\n[2/4] Creating MeshInstance3D node...")
    create_node_params = {
        "name": "execute_action_sequence",
        "arguments": {
            "actions": [{
                "action": "command_call",
                "method": "create_node",
                "node_type": "MeshInstance3D",
                "name": "ArrayMeshNode",
                "parent_path": "/root"
            }]
        }
    }
    
    create_result = make_mcp_request("tools/call", create_node_params)
    if "error" in create_result:
        print(f"❌ Failed to create node: {create_result.get('error')}")
        return False
    
    if "result" in create_result:
        result_content = create_result["result"].get("content", [])
        if result_content:
            content_text = result_content[0].get("text", "{}")
            result_data = json.loads(content_text)
            if result_data.get("success"):
                print(f"✅ Node created: {result_data.get('message', '')}")
            else:
                print(f"❌ Failed: {result_data.get('message', 'Unknown error')}")
                return False
    
    # Step 3: Set script to create ArrayMesh (NO @export var mesh - set mesh directly)
    print("\n[3/4] Setting SafeGDScript to create ArrayMesh...")
    script_code = """func _ready():
    # Create ArrayMesh with a simple cube
    var array_mesh = ArrayMesh.new()
    
    # Define vertices for a cube
    var vertices = PackedVector3Array([
        # Front face
        Vector3(-1, -1, 1),  # 0
        Vector3(1, -1, 1),   # 1
        Vector3(1, 1, 1),    # 2
        Vector3(-1, 1, 1),   # 3
        # Back face
        Vector3(-1, -1, -1), # 4
        Vector3(1, -1, -1),  # 5
        Vector3(1, 1, -1),   # 6
        Vector3(-1, 1, -1),  # 7
    ])
    
    # Define indices for triangles (clockwise winding)
    var indices = PackedInt32Array([
        # Front face
        0, 1, 2,  2, 3, 0,
        # Back face
        5, 4, 7,  7, 6, 5,
        # Top face
        3, 2, 6,  6, 7, 3,
        # Bottom face
        4, 0, 3,  3, 7, 4,
        # Right face
        1, 5, 6,  6, 2, 1,
        # Left face
        4, 5, 1,  1, 0, 4,
    ])
    
    # Create arrays for mesh
    var arrays = []
    arrays.resize(Mesh.ARRAY_MAX)
    arrays[Mesh.ARRAY_VERTEX] = vertices
    arrays[Mesh.ARRAY_INDEX] = indices
    
    # Add surface to mesh
    array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
    
    # Set mesh directly on MeshInstance3D (not @export var)
    mesh = array_mesh
    
    print("ArrayMesh created and set on MeshInstance3D!")"""
    
    set_script_params = {
        "name": "execute_action_sequence",
        "arguments": {
            "actions": [{
                "action": "command_set_script",
                "target_path": "/root/ArrayMeshNode",
                "script": script_code
            }]
        }
    }
    
    script_result = make_mcp_request("tools/call", set_script_params)
    if "error" in script_result:
        print(f"❌ Failed to set script: {script_result.get('error')}")
        return False
    
    if "result" in script_result:
        result_content = script_result["result"].get("content", [])
        if result_content:
            content_text = result_content[0].get("text", "{}")
            result_data = json.loads(content_text)
            if result_data.get("success"):
                print(f"✅ Script set: {result_data.get('message', '')}")
            else:
                print(f"❌ Failed: {result_data.get('message', 'Unknown error')}")
                return False
    
    # Step 4: Export as ESCN
    print("\n[4/4] Exporting scene as ESCN...")
    export_params = {
        "name": "execute_action_sequence",
        "arguments": {
            "actions": [{
                "action": "command_export_scene",
                "node_path": "/root/ArrayMeshNode",
                "format": "escn"
            }]
        }
    }
    
    export_result = make_mcp_request("tools/call", export_params)
    if "error" in export_result:
        print(f"❌ Failed to export scene: {export_result.get('error')}")
        return False
    
    if "result" in export_result:
        # Handle action sequence response format
        result = export_result["result"]
        
        # Check if it's an action sequence result
        if "results" in result and isinstance(result["results"], list) and len(result["results"]) > 0:
            action_result = result["results"][0]
            if action_result.get("success"):
                exported_data = action_result.get("exported_scene_data", "")
                if exported_data:
                    print("✅ Scene exported successfully!")
                    print(f"   Format: {action_result.get('file_extension', 'unknown')}")
                    print(f"   MIME type: {action_result.get('mime_type', 'unknown')}")
                    print(f"   Data length: {len(exported_data)} characters")
                    
                    # Save to file
                    output_file = "test_arraymesh_output.escn"
                    with open(output_file, "w") as f:
                        f.write(exported_data)
                    print(f"   Saved to: {output_file}")
                    
                    # Check if mesh is in the exported file
                    if "ArrayMesh" in exported_data or "mesh" in exported_data.lower():
                        print("   ✅ Mesh appears to be in exported file")
                    else:
                        print("   ⚠️  Warning: Mesh may not be in exported file")
                    
                    # Show first few lines
                    lines = exported_data.split("\n")[:15]
                    print("\n   First 15 lines of exported ESCN:")
                    for i, line in enumerate(lines, 1):
                        print(f"   {i:2d}: {line}")
                    
                    return True
                else:
                    print("❌ Export succeeded but no data returned")
                    return False
            else:
                print(f"❌ Export failed: {action_result.get('message', 'Unknown error')}")
                return False
        # Fallback: try content format (for direct tool calls)
        elif "content" in result:
            result_content = result.get("content", [])
            if result_content:
                content_text = result_content[0].get("text", "{}")
                result_data = json.loads(content_text)
                if result_data.get("success"):
                    exported_data = result_data.get("exported_scene_data", "")
                    if exported_data:
                        print("✅ Scene exported successfully!")
                        print(f"   Format: {result_data.get('file_extension', 'unknown')}")
                        print(f"   MIME type: {result_data.get('mime_type', 'unknown')}")
                        print(f"   Data length: {len(exported_data)} characters")
                        
                        # Save to file
                        output_file = "test_arraymesh_output.escn"
                        with open(output_file, "w") as f:
                            f.write(exported_data)
                        print(f"   Saved to: {output_file}")
                        
                        # Check if mesh is in the exported file
                        if "ArrayMesh" in exported_data or "mesh" in exported_data.lower():
                            print("   ✅ Mesh appears to be in exported file")
                        else:
                            print("   ⚠️  Warning: Mesh may not be in exported file")
                        
                        # Show first few lines
                        lines = exported_data.split("\n")[:15]
                        print("\n   First 15 lines of exported ESCN:")
                        for i, line in enumerate(lines, 1):
                            print(f"   {i:2d}: {line}")
                        
                        return True
    
    print("❌ Unexpected response format")
    print(f"   Response: {json.dumps(export_result, indent=2)}")
    return False

if __name__ == "__main__":
    print("\nStarting ArrayMesh Export Test...")
    print(f"MCP Server: {MCP_SERVER_URL}")
    print("\nNote: Make sure the Godot MCP server is running:")
    print("  ./bin/godot.macos.editor.arm64 --path modules/mcp/test_project --mcp-auto-start")
    print()
    
    success = test_arraymesh_export()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test PASSED")
        sys.exit(0)
    else:
        print("❌ Test FAILED")
        sys.exit(1)

