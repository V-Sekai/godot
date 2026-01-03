#!/usr/bin/env python3
import json
import requests
import sys
import time

MCP_SERVER_URL = "http://localhost:8083"

def get_response_data(response):
    if response.headers.get('Content-Type') == 'text/event-stream':
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                return json.loads(line[6:])
    else:
        return response.json()
    return None

def test_mcp():
    print("=" * 60)
    print("Godot MCP Full Test")
    print("=" * 60)
    
    # 1. Initialize
    print("1. Testing initialize...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    
    try:
        response = requests.post(MCP_SERVER_URL, json=init_request, timeout=5, stream=True)
        if response.status_code == 200:
            data = get_response_data(response)
            if data:
                print(f"   ✓ Protocol: {data.get('result', {}).get('protocolVersion')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # 2. Initialized notification
    print("2. Testing notifications/initialized...")
    requests.post(MCP_SERVER_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=2)
    print("   ✓ Sent")

    # 3. List Tools
    print("3. Testing tools/list...")
    response = requests.post(MCP_SERVER_URL, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, timeout=5)
    data = get_response_data(response)
    if data:
        tools = data.get('result', {}).get('tools', [])
        print(f"   ✓ Found {len(tools)} tools")

    # 4. List Resources
    print("4. Testing resources/list...")
    response = requests.post(MCP_SERVER_URL, json={"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}}, timeout=5)
    data = get_response_data(response)
    if data:
        resources = data.get('result', {}).get('resources', [])
        print(f"   ✓ Found {len(resources)} resources")
        for res in resources:
            print(f"     - {res.get('uri')}: {res.get('name')}")

    # 5. Read Resource
    print("5. Testing resources/read...")
    response = requests.post(MCP_SERVER_URL, json={"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "resource://server_info"}}, timeout=5)
    data = get_response_data(response)
    if data:
        contents = data.get('result', {}).get('contents', [])
        if contents:
            print(f"   ✓ Resource content: {contents[0].get('text')}")

    # 6. List Prompts
    print("6. Testing prompts/list...")
    response = requests.post(MCP_SERVER_URL, json={"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}}, timeout=5)
    data = get_response_data(response)
    if data:
        prompts = data.get('result', {}).get('prompts', [])
        print(f"   ✓ Found {len(prompts)} prompts")
        for p in prompts:
            print(f"     - {p.get('name')}: {p.get('description')}")

    # 7. Get Prompt
    print("7. Testing prompts/get...")
    response = requests.post(MCP_SERVER_URL, json={"jsonrpc": "2.0", "id": 6, "method": "prompts/get", "params": {"name": "save_mesh_to_escn"}}, timeout=5)
    data = get_response_data(response)
    if data:
        messages = data.get('result', {}).get('messages', [])
        if messages:
            print(f"   ✓ Prompt has {len(messages)} messages")

    # 8. Execute Script
    print("8. Testing tools/call (execute_script)...")
    execute_request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "execute_script",
            "arguments": {
                "code": "func run():\n    return 42",
                "language": "safegdscript",
                "method": "run"
            }
        }
    }
    response = requests.post(MCP_SERVER_URL, json=execute_request, timeout=5)
    data = get_response_data(response)
    if data:
        result = data.get('result', {})
        # The tool returns a list of content items in MCP
        content = result.get('content', [])
        if content and len(content) > 0:
            text = content[0].get('text', '')
            print(f"   ✓ Execution result: {text}")
            if "42" in text:
                print("   ✓ Correct value returned")

    print("=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_mcp()
