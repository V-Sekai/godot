# Plan A: Add Y-Branching Support to Base ManyBoneIK3D

## Goal
Enhance base ManyBoneIK3D with Y-branching capabilities while preserving existing API and ensuring all IK solvers (CCDIK3D, EWBIK3D, future solvers) benefit automatically.

## Overview
This plan migrates the sophisticated bone list splitter algorithm from EWBIK3D's `IKBoneSegment3D` system to the base ManyBoneIK3D class, enabling automatic Y-branching detection and coordination while maintaining the existing `root_bone_name` + `end_bone_name` API contract.

---

## Phase A1: Migrate Bone List Splitter Algorithm

### A1.1 Extract Skeleton Analysis from EWBIK3D

**Files to modify:**
- `scene/3d/many_bone_ik_3d.h`
- `scene/3d/many_bone_ik_3d.cpp`

**Extract from EWBIK3D:**
- `IKBoneSegment3D::generate_default_segments()` logic
- `IKBoneSegment3D::_has_multiple_children_or_pinned()` detection
- `IKBoneSegment3D::_process_children()` branching logic
- `IKBoneSegment3D::_is_parent_of_tip()` hierarchy analysis

**Add to base ManyBoneIK3D:**
```cpp
class ManyBoneIK3D {
private:
    // Y-branching support structures
    struct SharedSegment {
        int root_bone_id;
        Vector<int> setting_indices;  // Settings that share this root
        Vector<int> shared_bone_chain; // Common parent bones
        HashMap<int, Vector<int>> branch_points; // bone_id -> child_settings
    };
    
    HashMap<int, Vector<int>> branching_map;  // bone_id -> [setting_indices]
    Vector<SharedSegment> shared_segments;
    bool has_branching = false;
    
    // Skeleton analysis methods
    void analyze_skeleton_branching();
    void detect_shared_parent_bones();
    void build_shared_segments();
    bool _has_multiple_children_or_pinned(int bone_id, const Vector<int> &children);
    Vector<int> _find_branching_points();
    void _trace_bone_chains();
    
public:
    bool has_y_branching() const { return has_branching; }
    Vector<SharedSegment> get_shared_segments() const { return shared_segments; }
};
```

### A1.2 Add Y-Branching Detection to Base Class

**Implementation details:**
```cpp
void ManyBoneIK3D::analyze_skeleton_branching() {
    Skeleton3D *skeleton = get_skeleton();
    if (!skeleton) return;
    
    // Clear previous analysis
    branching_map.clear();
    shared_segments.clear();
    has_branching = false;
    
    // Build map of bone_id -> settings that use it as root
    for (int i = 0; i < settings.size(); i++) {
        int root_bone = settings[i]->root_bone;
        if (root_bone >= 0) {
            if (!branching_map.has(root_bone)) {
                branching_map[root_bone] = Vector<int>();
            }
            branching_map[root_bone].push_back(i);
        }
    }
    
    // Detect Y-branching (multiple settings sharing same root)
    for (const KeyValue<int, Vector<int>> &pair : branching_map) {
        if (pair.value.size() > 1) {
            has_branching = true;
            _create_shared_segment(pair.key, pair.value);
        }
    }
}

void ManyBoneIK3D::_create_shared_segment(int root_bone, const Vector<int> &setting_indices) {
    SharedSegment segment;
    segment.root_bone_id = root_bone;
    segment.setting_indices = setting_indices;
    
    // Find common parent bones for all chains in this segment
    _build_shared_bone_chain(segment);
    
    // Identify branch points
    _identify_branch_points(segment);
    
    shared_segments.push_back(segment);
}
```

### A1.3 Skeleton Topology Analysis

**Key algorithms to implement:**
- **Shared bone detection**: Find bones that are parents to multiple end bones
- **Branch point identification**: Detect where chains diverge
- **Optimal solving order**: Determine sequence for coordinated solving

---

## Phase A2: Enhanced Coordination Framework

### A2.1 Shared Bone Coordination

**Add coordination logic:**
```cpp
void ManyBoneIK3D::_solve_with_branching_coordination(double p_delta) {
    // Solve shared segments first (coordinate multiple targets)
    for (const SharedSegment &segment : shared_segments) {
        _solve_shared_segment(segment, p_delta);
    }
    
    // Solve independent linear chains normally
    for (int i = 0; i < settings.size(); i++) {
        if (!_is_setting_in_shared_segment(i)) {
            _solve_linear_chain(settings[i], p_delta);
        }
    }
}

void ManyBoneIK3D::_solve_shared_segment(const SharedSegment &segment, double p_delta) {
    // Coordinate solving for multiple targets sharing parent bones
    Vector<Vector3> destinations;
    Vector<float> weights;
    
    // Collect all targets for this segment
    for (int setting_idx : segment.setting_indices) {
        Node3D *target = Object::cast_to<Node3D>(get_node_or_null(settings[setting_idx]->target_node));
        if (target) {
            destinations.push_back(target->get_global_position());
            weights.push_back(1.0f); // TODO: Add weight support
        }
    }
    
    // Solve with multiple destinations
    _solve_multi_target_iteration(segment, destinations, weights, p_delta);
}
```

### A2.2 Enhanced _process_modification()

**Update main processing loop:**
```cpp
void ManyBoneIK3D::_process_modification(double p_delta) {
    Skeleton3D *skeleton = get_skeleton();
    if (!skeleton) return;
    
    // Analyze skeleton for Y-branching on each frame
    // (could be optimized to only run when settings change)
    analyze_skeleton_branching();
    
    min_distance_squared = min_distance * min_distance;
    
    if (has_y_branching()) {
        // Use coordinated solving for Y-branching
        _solve_with_branching_coordination(p_delta);
    } else {
        // Use existing linear chain solving
        _solve_linear_chains(p_delta);
    }
}
```

### A2.3 Solver Contract Preservation

**Ensure backward compatibility:**
- Each `_solve_iteration()` call still receives single linear chain data
- Y-branching coordination happens at higher level
- CCDIK3D and EWBIK3D solvers require no changes
- New virtual method for multi-target solving (optional override)

```cpp
// New optional override for advanced solvers
virtual void _solve_multi_target_iteration(const SharedSegment &segment, 
                                         const Vector<Vector3> &destinations,
                                         const Vector<float> &weights,
                                         double p_delta) {
    // Default implementation: solve each target separately
    for (int i = 0; i < segment.setting_indices.size(); i++) {
        int setting_idx = segment.setting_indices[i];
        if (i < destinations.size()) {
            _solve_single_target(settings[setting_idx], destinations[i], p_delta);
        }
    }
}
```

---

## Phase A3: Enhanced Property System

### A3.1 Auto-Generation Helpers

**Add convenience methods:**
```cpp
class ManyBoneIK3D {
public:
    // Helper methods for Y-branching setup
    void add_target(NodePath target_node, const String &end_bone_name);
    void auto_generate_settings_from_targets(const Vector<NodePath> &targets);
    Vector<String> find_optimal_root_bones(const Vector<String> &end_bone_names);
    void optimize_settings_for_branching();
    
    // Analysis and validation
    bool validate_branching_configuration();
    Vector<String> get_branching_warnings();
    void suggest_branching_improvements();
};
```

**Implementation:**
```cpp
void ManyBoneIK3D::add_target(NodePath target_node, const String &end_bone_name) {
    Skeleton3D *skeleton = get_skeleton();
    if (!skeleton) return;
    
    int end_bone = skeleton->find_bone(end_bone_name);
    if (end_bone < 0) return;
    
    // Find optimal root bone for this target
    String optimal_root = _find_optimal_root_for_end_bone(end_bone_name);
    
    // Create new setting
    int new_idx = settings.size();
    set_setting_count(new_idx + 1);
    set_root_bone_name(new_idx, optimal_root);
    set_end_bone_name(new_idx, end_bone_name);
    set_target_node(new_idx, target_node);
}

String ManyBoneIK3D::_find_optimal_root_for_end_bone(const String &end_bone_name) {
    Skeleton3D *skeleton = get_skeleton();
    if (!skeleton) return "";
    
    int end_bone = skeleton->find_bone(end_bone_name);
    if (end_bone < 0) return "";
    
    // Traverse up skeleton to find good root candidates
    // Prefer bones that are:
    // 1. Not too close to end bone (need some chain length)
    // 2. Natural branching points (spine, pelvis, etc.)
    // 3. Shared with other existing chains (for Y-branching)
    
    Vector<int> candidates;
    int current = skeleton->get_bone_parent(end_bone);
    int chain_length = 0;
    
    while (current >= 0 && chain_length < 10) { // Max chain length
        candidates.push_back(current);
        
        // Check if this bone is already used as root (good for Y-branching)
        for (int i = 0; i < settings.size(); i++) {
            if (settings[i]->root_bone == current) {
                return skeleton->get_bone_name(current); // Prefer shared roots
            }
        }
        
        current = skeleton->get_bone_parent(current);
        chain_length++;
    }
    
    // Return a reasonable root (not too close, not too far)
    if (candidates.size() >= 3) {
        return skeleton->get_bone_name(candidates[2]); // 3 bones up
    } else if (!candidates.is_empty()) {
        return skeleton->get_bone_name(candidates[0]);
    }
    
    return "";
}
```

### A3.2 Pin-Style Property Support

**Add alternative property interface:**
```cpp
// Enhanced property system supporting both styles:
// Traditional: settings/0/root_bone_name, settings/0/end_bone_name
// Pin-style: pins/0/bone_name, pins/0/target_node

bool ManyBoneIK3D::_set(const StringName &p_path, const Variant &p_value) {
    String path = p_path;
    
    // Handle pin-style properties
    if (path.begins_with("pins/")) {
        return _handle_pin_property(path, p_value);
    }
    
    // Handle existing settings properties
    if (path.begins_with("settings/")) {
        return _handle_setting_property(path, p_value);
    }
    
    return false;
}

bool ManyBoneIK3D::_handle_pin_property(const String &path, const Variant &p_value) {
    int pin_idx = path.get_slicec('/', 1).to_int();
    String prop = path.get_slicec('/', 2);
    
    if (prop == "bone_name") {
        // Auto-create setting for this pin
        String end_bone_name = p_value;
        NodePath target_path; // Get from existing or default
        add_target(target_path, end_bone_name);
        return true;
    } else if (prop == "target_node") {
        // Update target for corresponding setting
        // Find setting that corresponds to this pin index
        if (pin_idx < settings.size()) {
            set_target_node(pin_idx, p_value);
            return true;
        }
    }
    
    return false;
}
```

### A3.3 Enhanced Property Validation

**Add validation and warnings:**
```cpp
Vector<String> ManyBoneIK3D::get_branching_warnings() {
    Vector<String> warnings;
    
    analyze_skeleton_branching();
    
    // Check for suboptimal configurations
    for (const SharedSegment &segment : shared_segments) {
        if (segment.setting_indices.size() > 4) {
            warnings.push_back("Segment with root '" + 
                get_skeleton()->get_bone_name(segment.root_bone_id) + 
                "' has many branches (" + itos(segment.setting_indices.size()) + 
                "). Consider splitting into multiple segments.");
        }
        
        // Check for very short chains
        for (int setting_idx : segment.setting_indices) {
            if (_get_chain_length(setting_idx) < 2) {
                warnings.push_back("Setting " + itos(setting_idx) + 
                    " has very short chain. Consider different root bone.");
            }
        }
    }
    
    return warnings;
}
```

---

## Phase A4: Universal Benefits

### A4.1 CCDIK3D Automatic Enhancement

**Zero changes required to CCDIK3D:**
- CCDIK3D automatically gains Y-branching when multiple settings share root bones
- Base class coordination handles the complexity
- CCDIK3D's `_solve_iteration()` continues to work on linear chains
- Example: CCDIK3D can now solve both arms reaching different targets simultaneously

**Testing scenarios:**
```cpp
// Test case: CCDIK3D with Y-branching
ccdik.set_setting_count(2);
ccdik.set_root_bone_name(0, "spine_base");  // Shared root
ccdik.set_end_bone_name(0, "hand_L");
ccdik.set_target_node(0, NodePath("target_L"));
ccdik.set_root_bone_name(1, "spine_base");  // Shared root
ccdik.set_end_bone_name(1, "hand_R");
ccdik.set_target_node(1, NodePath("target_R"));

// Base class automatically detects Y-branching and coordinates solving
```

### A4.2 Enhanced Base Class Capabilities

**All future IK solvers gain:**
- Automatic Y-branching detection and coordination
- Rich skeleton analysis capabilities
- Optimal solving coordination for shared parent bones
- Enhanced property system with validation

### A4.3 Backward Compatibility

**Guaranteed compatibility:**
- Existing single-chain setups work unchanged
- Y-branching only activates when multiple settings share root bones
- No breaking changes to current API
- Performance impact minimal for non-branching cases

---

## Implementation Checklist

### Phase A1: Skeleton Analysis
- [ ] Extract `IKBoneSegment3D` algorithms to base class
- [ ] Add Y-branching detection structures
- [ ] Implement skeleton topology analysis
- [ ] Add shared bone detection logic

### Phase A2: Coordination Framework
- [ ] Implement shared segment solving
- [ ] Enhance `_process_modification()` with branching support
- [ ] Add multi-target iteration support
- [ ] Preserve solver contract compatibility

### Phase A3: Property System
- [ ] Add auto-generation helper methods
- [ ] Implement pin-style property support
- [ ] Add validation and warning system
- [ ] Create optimization suggestions

### Phase A4: Testing and Validation
- [ ] Test CCDIK3D with Y-branching scenarios
- [ ] Verify backward compatibility
- [ ] Performance testing for non-branching cases
- [ ] Integration testing with complex skeletons

## Success Criteria

1. **CCDIK3D Enhancement**: CCDIK3D can solve Y-branching scenarios without code changes
2. **API Preservation**: All existing code continues to work unchanged
3. **Performance**: Minimal impact on non-branching scenarios
4. **Robustness**: Handles complex skeleton topologies correctly
5. **Usability**: Helper methods make Y-branching setup easy

## Risk Mitigation

1. **Backward Compatibility**: Extensive testing with existing setups
2. **Performance**: Lazy evaluation of branching analysis
3. **Complexity**: Clear separation between analysis and solving
4. **Debugging**: Rich validation and warning system

This plan establishes the foundation for universal Y-branching support across all ManyBoneIK solvers while maintaining full backward compatibility.
