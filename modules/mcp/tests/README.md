# MCP Test Fixtures Taxonomy

This directory contains comprehensive test fixtures for the Godot MCP (Model Context Protocol) implementation, organized by taxonomy for clarity and maintainability.

## Directory Structure

### 🔒 `security/`
Security vulnerability testing for SafeGDScript implementation.

- **`safegdscript_comprehensive_security_tests.jsonl`** (284 lines)
  - 71 security-critical operations across 4 categories
  - Tests both single-line and multi-line code formats
  - Categories: System, File, Resource, Reflection operations

- **`safegdscript_comprehensive_security_results.jsonl`** (142 lines)
  - Execution results from security tests
  - Current results: 57/142 operations allowed (40.1% security coverage)
  - Documents security vulnerabilities and SafeGDScript limitations

### 🔍 `fuzz/` (Future Use)
Prepared for comprehensive fuzz testing infrastructure.

## Test Categories

### System Operations (26 tests)
- OS environment access, process execution, system information
- Examples: `OS.execute()`, `OS.get_environment()`, `OS.get_name()`

### File Operations (20 tests)
- File system access, directory operations, file manipulation
- Examples: `FileAccess.open()`, `DirAccess.remove()`, `FileAccess.get_file_as_string()`

### Resource Operations (8 tests)
- Resource loading, asset management, dynamic imports
- Examples: `load()`, `ResourceLoader.load()`, `ResourceSaver.save()`

### Reflection Operations (17 tests)
- Runtime type inspection, dynamic instantiation, method enumeration
- Examples: `ClassDB.instantiate()`, `ClassDB.class_get_method_list()`, `ClassDB.class_call_static()`

## File Format

All test fixtures use JSONL (JSON Lines) format:
- Each line contains one JSON object
- Fields: `test_id`, `category`, `operation`, `format`, `code`, `language`, `executed`, `error`, `timestamp`
- Compatible with standard JSON processing tools

## Usage

### Running Tests
```bash
# Run security tests
cd modules/mcp/tests
python3 run_security_tests.py
```

### Analyzing Results
```bash
# Count total test cases
wc -l modules/mcp/tests/*/*.jsonl

# Search for specific operations
grep "OS.execute" modules/mcp/tests/security/*.jsonl

# Analyze security coverage
python3 -c "
import json
results = []
for line in open('modules/mcp/tests/security/safegdscript_comprehensive_security_results.jsonl'):
    results.append(json.loads(line.strip()))
allowed = sum(1 for r in results if r.get('executed', False))
print(f'Security coverage: {allowed}/{len(results)} ({allowed/len(results)*100:.1f}%)')
"
```

## Security Status

**Current SafeGDScript Security Coverage: 40.1%**
- 57 out of 142 security-critical operations are allowed
- Major vulnerabilities in System, File, and Network operations
- Reflection operations largely unrestricted

## Maintenance

- **Regeneration**: Test fixtures are regenerated from live MCP server testing
- **Updates**: Files are updated when SafeGDScript implementation changes
- **Validation**: Used for regression testing and security audits

## Contributing

When adding new test fixtures:
1. Choose appropriate taxonomy directory
2. Follow JSONL format with consistent field naming
3. Include comprehensive test coverage
4. Update this README with new categories or significant changes
