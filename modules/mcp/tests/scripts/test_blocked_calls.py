import requests
import json
import time

def test_blocked_calls():
    url = "http://localhost:8083"
    
    # 1. Initialize
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }
    requests.post(url, json=init_payload)
    requests.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    tests = [
        ("return OS.get_name()", "potentially dangerous operations: OS.get_name"),
        ("var d = DirAccess.open('res://')", "potentially dangerous operations: DirAccess.open"),
        ("var f = FileAccess.open('res://test.txt', FileAccess.WRITE)", "potentially dangerous operations: FileAccess.open"),
        ("var s = load('res://dangerous.gd')", "potentially dangerous operations: load"),
        ("var e = Engine.get_singleton('OS')", "potentially dangerous operations: Engine.get_singleton"),
        ("var g = GDScript.new()", "potentially dangerous operations: GDScript.new"),
        ("var h = HTTPClient.new()", "potentially dangerous operations: HTTPClient.new"),
        ("var x = AudioServer.get_bus_count()", "SafeGDScript compilation failed"), # Blocked by sandbox class whitelist
    ]

    for code, expected_error in tests:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "execute_script",
                "arguments": {
                    "code": f"func run():\n    {code}",
                    "language": "safegdscript",
                    "method": "run"
                }
            }
        }
        
        response = requests.post(url, json=payload).json()
        print(f"Testing: {code}")
        
        is_error = response.get("result", {}).get("isError", False)
        # The server might return success: false or isError: true depending on how it's mapped
        # In our current implementation, it returns success: false in the tool result
        
        content = response.get("result", {}).get("content", [])
        text = content[0].get("text", "") if content else ""
        
        if "Error" in text and expected_error in text:
            print(f"  ✅ Correctly blocked: {text}")
        else:
            print(f"  ❌ FAILED to block or wrong error: {text}")

if __name__ == "__main__":
    test_blocked_calls()
