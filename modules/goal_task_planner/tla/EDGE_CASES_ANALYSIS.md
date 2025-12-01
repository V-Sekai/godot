# Edge Cases Analysis for Failing Tests

## Backtracking Test 4

### Test Setup
- **Initial state**: `flag = -1`
- **Tasks**: `[put_it, need1]`
- **put_it methods**:
  - `m_err`: `[action_putv(0), action_getv(1)]` - Sets flag=0, then checks flag==1 (FAILS)
  - `m0`: `[action_putv(0), action_getv(0)]` - Sets flag=0, then checks flag==0 (SUCCEEDS)
  - `m1`: `[action_putv(1), action_getv(1)]` - Sets flag=1, then checks flag==1 (SUCCEEDS)
- **need1 method**:
  - `m_need1`: `[action_getv(1)]` - Checks flag==1 (SUCCEEDS if flag==1)

### Expected Execution Flow

1. **Initial**: `flag = -1`, tasks `put_it` and `need1` are OPEN
2. **Process put_it**: Try methods in order
   - Try `m_err`: 
     - Execute `action_putv(0)` → `flag = 0`
     - Execute `action_getv(1)` → FAILS (flag=0, not 1)
     - Backtrack: blacklist `m_err` array `[[action_putv, 0], [action_getv, 1]]`
     - Reopen `put_it`
   - Try `m0`:
     - Execute `action_putv(0)` → `flag = 0`
     - Execute `action_getv(0)` → SUCCEEDS (flag=0)
     - `put_it` CLOSED
3. **Process need1**:
   - Execute `action_getv(1)` → FAILS (flag=0, not 1)
   - Backtrack: blacklist `m0` array `[[action_putv, 0], [action_getv, 0]]`
   - Reopen `put_it`
4. **Process put_it again** (reopened):
   - `m_err` is blacklisted → skip
   - `m0` is blacklisted → skip
   - Try `m1`:
     - Execute `action_putv(1)` → `flag = 1`
     - Execute `action_getv(1)` → SUCCEEDS (flag=1)
     - `put_it` CLOSED
5. **Process need1 again**:
   - Execute `action_getv(1)` → SUCCEEDS (flag=1)
   - `need1` CLOSED
6. **Planning succeeds**: Final plan `[action_putv(1), action_getv(1), action_getv(1)]`

### Key Edge Cases

#### Edge Case 1: Nested Array Comparison
**Issue**: When comparing blacklisted arrays, we need to compare nested arrays element-by-element.

**Current Code** (`_is_command_blacklisted`):
```cpp
bool match = true;
for (int j = 0; j < action_arr.size(); j++) {
    if (action_arr[j] != blacklisted_arr[j]) {
        match = false;
        break;
    }
}
```

**Problem**: This uses `!=` operator which may not work correctly for nested arrays in Godot. We need explicit nested comparison.

**Fix Applied**: Added nested array comparison:
```cpp
if (action_elem.get_type() == Variant::ARRAY && blacklisted_elem.get_type() == Variant::ARRAY) {
    Array action_elem_arr = action_elem;
    Array blacklisted_elem_arr = blacklisted_elem;
    if (action_elem_arr.size() != blacklisted_elem_arr.size()) {
        match = false;
        break;
    }
    for (int k = 0; k < action_elem_arr.size(); k++) {
        if (action_elem_arr[k] != blacklisted_elem_arr[k]) {
            match = false;
            break;
        }
    }
}
```

#### Edge Case 2: Method Array Blacklisting After Reopening
**Issue**: When a task is reopened after backtracking, its previously used method array must be blacklisted.

**Current Behavior**: 
- When `put_it` is reopened after `need1` fails, `m0`'s array `[[action_putv, 0], [action_getv, 0]]` is blacklisted
- When `put_it` tries methods again, it should skip `m_err` (already blacklisted) and `m0` (just blacklisted), and try `m1`

**Potential Issue**: The blacklist comparison might not be working correctly, causing all methods to appear blacklisted even when they shouldn't be.

#### Edge Case 3: Array Reference vs Value
**Issue**: When storing `created_subtasks` in a node, we store a reference. When blacklisting, we need to ensure we're comparing values, not references.

**Current Code**:
```cpp
curr_node["created_subtasks"] = subtasks.duplicate(true);  // Deep copy
```

**In Backtracking**:
```cpp
Array subtasks_copy = created_subtasks.duplicate(true);  // Deep copy
updated_blacklist.push_back(subtasks_copy);
```

**Potential Issue**: Even with deep copies, the comparison might fail if the array structure is different (e.g., different object instances with same values).

## Sample Test 1

### Test Setup
- **Initial state**: `flag[0] = True`, `flag[1..7] = False`
- **Tasks**: `[task_method_1, task_method_2]`
- **task_method_1 methods**:
  - `m1_1`: `[transfer_flag(0,1), transfer_flag(1,2), transfer_flag(3,4)]`
  - `m1_2`: `[transfer_flag(0,1), transfer_flag(1,2), transfer_flag(2,3)]`
  - `m1_3`: `[transfer_flag(0,1), transfer_flag(1,2), transfer_flag(2,3), transfer_flag(3,4)]`
- **task_method_2 methods**:
  - `m2_1`: `[transfer_flag(3,4), transfer_flag(4,5), transfer_flag(5,6), transfer_flag(6,7)]`
  - `m2_2`: `[transfer_flag(4,5), transfer_flag(5,6), transfer_flag(6,7)]`

### Expected Execution Flow

1. **Process task_method_1**: Must set `flag[3] = True` for `task_method_2` to succeed
2. **Process task_method_2**: Must set `flag[7] = True`

### Key Edge Case

**Issue**: The planner stops early after processing only one task, generating only 3 actions instead of 7.

**Potential Causes**:
1. `original_todo_list` not being set correctly (FIXED)
2. Task completion check failing
3. Solution graph extraction missing actions

## Recommendations

1. **Add detailed logging** to see exactly what arrays are being compared in `_is_command_blacklisted`
2. **Verify deep copy behavior** - ensure `duplicate(true)` creates truly independent arrays
3. **Test array comparison directly** - create a unit test that verifies nested array comparison works correctly
4. **Check for test contamination** - ensure tests don't share state between runs

