# Plan B: Remove Y-Branching Support from EWBIK3D

## Goal
Simplify EWBIK3D by removing duplicate Y-branching systems and migrating to use the enhanced base ManyBoneIK3D capabilities from Plan A.

## Overview
This plan removes EWBIK3D's parallel pin/constraint/segmentation systems and migrates to use the base class Y-branching support, pin system, and constraint system. EWBIK3D becomes a focused solver that leverages the base class infrastructure.

## Prerequisites
- Plan A must be completed first
- Base ManyBoneIK3D has Y-branching support
- Base ManyBoneIK3D has enhanced property system
- `IKKusudama3D` properly inherits from `JointLimitation3D`

---

## Phase B1: Migrate Pin System to Base Class

### B1.1 Remove EWBIK3D Pins Array

**Files to modify:**
- `modules/many_bone_ik/src/many_bone_ik_3d.h`
- `modules/many_bone_ik/src/many_bone_ik_3d.cpp`

**Remove from EWBIK3D:**
```cpp
// DELETE these from EWBIK3D
Vector<Ref<IKEffectorTemplate3D>> pins;
int pin_count = 0;

// DELETE pin management methods
void set_pin_count(int p_count);
int get_pin_count() const;
void set_pin_bone_name(int p_pin_index, const String &p_bone_name);
String get_pin_bone_name(int p_pin_index) const;
void set_pin_target_node(int p_pin_index, const NodePath &p_target_node);
NodePath get_pin_target_node(int p_pin_index) const;
// ... other pin methods
```

### B1.2 Map Pin Properties to Base Class Settings

**Update EWBIK3D property system:**
```cpp
bool EWBIK3D::_set(const StringName &p_path, const Variant &p_value) {
    String path = p_path;
    
    // Map pin properties to base class settings
    if (path.begins_with("pins/")) {
        return _handle_pin_to_setting_mapping(path, p_value);
    }
    
    // Handle remaining EWBIK3D-specific properties
    return ManyBoneIK3D::_set(p_path, p_value);
}

bool EWBIK3D::_handle_pin_to_setting_mapping(const String &path, const Variant &p_value) {
    int pin_idx = path.get_slicec('/', 1).to_int();
    String prop = path.get_slicec('/', 2);
    
    if (prop == "bone_name") {
        // Create/update setting for this pin
        String end_bone_name = p_value;
        _ensure_setting_for_pin(pin_idx, end_bone_name);
        return true;
    } else if (prop == "target_node") {
        // Update target for corresponding setting
        if (pin_idx < get_setting_count()) {
            set_target_node(pin_idx, p_value);
            return true;
        }
    } else if (prop == "weight") {
        // Store weight in setting (need to add weight support to base class)
        _set_setting_weight(pin_idx, p_value);
        return true;
    } else if (prop == "direction_priorities") {
        // Store direction priorities in setting
        _set_setting_direction_priorities(pin_idx, p_value);
        return true;
    }
    
    return false;
}

void EWBIK3D::_ensure_setting_for_pin(int pin_idx, const String &end_bone_name) {
    // Ensure we have enough settings
    if (pin_idx >= get_setting_count()) {
        set_setting_count(pin_idx + 1);
    }
    
    // Use base class helper to find optimal root bone
    String optimal_root = _find_optimal_root_for_end_bone(end_bone_name);
    
    set_root_bone_name(pin_idx, optimal_root);
    set_end_bone_name(pin_idx, end_bone_name);
}
```

### B1.3 Auto-Population from Pin Interface

**Maintain EWBIK3D's user-friendly pin interface:**
```cpp
void EWBIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
    // Expose pin-style properties that map to base class settings
    int pin_count = get_setting_count(); // Use settings count as pin count
    
    p_list->push_back(PropertyInfo(Variant::INT, "pin_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Pins,pins/,static,const"));
    
    for (int i = 0; i < pin_count; i++) {
        String pin_path = "pins/" + itos(i) + "/";
        
        // Map to base class setting properties
        p_list->push_back(PropertyInfo(Variant::STRING, pin_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, get_skeleton_bone_names()));
        p_list->push_back(PropertyInfo(Variant::NODE_PATH, pin_path + "target_node"));
        p_list->push_back(PropertyInfo(Variant::FLOAT, pin_path + "weight", PROPERTY_HINT_RANGE, "0,1,0.01"));
        p_list->push_back(PropertyInfo(Variant::VECTOR3, pin_path + "direction_priorities"));
        p_list->push_back(PropertyInfo(Variant::FLOAT, pin_path + "motion_propagation_factor", PROPERTY_HINT_RANGE, "0,1,0.01"));
    }
    
    // Add base class properties
    ManyBoneIK3D::_get_property_list(p_list);
}

bool EWBIK3D::_get(const StringName &p_path, Variant &r_ret) const {
    String path = p_path;
    
    if (path.begins_with("pins/")) {
        return _handle_pin_property_get(path, r_ret);
    }
    
    return ManyBoneIK3D::_get(p_path, r_ret);
}

bool EWBIK3D::_handle_pin_property_get(const String &path, Variant &r_ret) const {
    int pin_idx = path.get_slicec('/', 1).to_int();
    String prop = path.get_slicec('/', 2);
    
    if (pin_idx >= get_setting_count()) {
        return false;
    }
    
    if (prop == "bone_name") {
        r_ret = get_end_bone_name(pin_idx);
        return true;
    } else if (prop == "target_node") {
        r_ret = get_target_node(pin_idx);
        return true;
    } else if (prop == "weight") {
        r_ret = _get_setting_weight(pin_idx);
        return true;
    } else if (prop == "direction_priorities") {
        r_ret = _get_setting_direction_priorities(pin_idx);
        return true;
    }
    
    return false;
}
```

---

## Phase B2: Remove Segmented Skeletons System

### B2.1 Delete Segmented Skeletons

**Remove from EWBIK3D:**
```cpp
// DELETE these from EWBIK3D
Vector<Ref<IKBoneSegment3D>> segmented_skeletons;

// DELETE segment management methods
void _create_segmented_skeletons();
void _update_segmented_skeletons();
Ref<IKBoneSegment3D> _get_segment_for_bone(int bone_id);
void _solve_segments(double p_delta);
```

**Remove segment creation logic:**
```cpp
// DELETE from EWBIK3D::_process_modification()
void EWBIK3D::_process_modification(double p_delta) {
    // DELETE: _create_segmented_skeletons();
    // DELETE: _update_segmented_skeletons();
    // DELETE: _solve_segments(p_delta);
    
    // Use base class processing instead
    ManyBoneIK3D::_process_modification(p_delta);
}
```

### B2.2 Remove Segment-Based Solving

**Simplify EWBIK3D solving:**
```cpp
// DELETE complex segment iteration
// REPLACE with simple linear chain solving

void EWBIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination) {
    // Focus on sophisticated solving algorithm for single linear chain
    // Remove all segment coordination logic
    // Use base class Y-branching coordination instead
    
    // EWBIK3D's advanced solving algorithm here
    _solve_ewbik_linear_chain(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_destination);
}

void EWBIK3D::_solve_ewbik_linear_chain(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination) {
    // Implement EWBIK3D's sophisticated solving for linear chain
    // Use QCP solver, constraint application, etc.
    // Work on single chain - base class handles Y-branching
}
```

### B2.3 Simplify EWBIK3D Initialization

**Remove complex setup:**
```cpp
// DELETE from EWBIK3D
void EWBIK3D::_make_all_joints_dirty() {
    // DELETE: segmented skeleton rebuilding
    // USE: base class joint management
    ManyBoneIK3D::_make_all_joints_dirty();
}

void EWBIK3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
    // DELETE: segment rebuilding logic
    // USE: base class skeleton change handling
    ManyBoneIK3D::_skeleton_changed(p_old, p_new);
}
```

---

## Phase B3: Migrate Constraint System

### B3.1 Remove Constraint Arrays

**Delete from EWBIK3D:**
```cpp
// DELETE these constraint arrays
PackedStringArray constraint_names;
PackedFloat32Array joint_twist;
TypedArray<IKLimitCone3D> kusudama_open_cones;
int constraint_count = 0;

// DELETE constraint management methods
void set_constraint_count(int p_count);
int get_constraint_count() const;
void set_constraint_name(int p_constraint_index, const String &p_constraint_name);
String get_constraint_name(int p_constraint_index) const;
void set_joint_twist(int p_constraint_index, float p_twist);
float get_joint_twist(int p_constraint_index) const;
// ... other constraint methods
```

### B3.2 Map Constraints to JointLimitation3D

**Update constraint property system:**
```cpp
bool EWBIK3D::_set(const StringName &p_path, const Variant &p_value) {
    String path = p_path;
    
    // Map constraint properties to base class joint limitations
    if (path.begins_with("constraints/")) {
        return _handle_constraint_to_limitation_mapping(path, p_value);
    }
    
    return ManyBoneIK3D::_set(p_path, p_value);
}

bool EWBIK3D::_handle_constraint_to_limitation_mapping(const String &path, const Variant &p_value) {
    int constraint_idx = path.get_slicec('/', 1).to_int();
    String prop = path.get_slicec('/', 2);
    
    // Find which setting/joint this constraint applies to
    int setting_idx, joint_idx;
    if (!_find_joint_for_constraint(constraint_idx, setting_idx, joint_idx)) {
        return false;
    }
    
    if (prop == "twist_start" || prop == "twist_range") {
        // Create/update IKKusudama3D for this joint
        Ref<IKKusudama3D> kusudama = _ensure_kusudama_for_joint(setting_idx, joint_idx);
        if (prop == "twist_start") {
            kusudama->set_axial_limits(p_value, kusudama->get_range_angle());
        } else if (prop == "twist_range") {
            kusudama->set_axial_limits(kusudama->get_min_axial_angle(), p_value);
        }
        return true;
    } else if (prop == "open_cones") {
        // Update kusudama open cones
        Ref<IKKusudama3D> kusudama = _ensure_kusudama_for_joint(setting_idx, joint_idx);
        kusudama->set_open_cones(p_value);
        return true;
    }
    
    return false;
}

Ref<IKKusudama3D> EWBIK3D::_ensure_kusudama_for_joint(int setting_idx, int joint_idx) {
    Ref<JointLimitation3D> existing = get_joint_limitation(setting_idx, joint_idx);
    Ref<IKKusudama3D> kusudama = Object::cast_to<IKKusudama3D>(existing.ptr());
    
    if (kusudama.is_null()) {
        kusudama = Ref<IKKusudama3D>(memnew(IKKusudama3D));
        set_joint_limitation(setting_idx, joint_idx, kusudama);
    }
    
    return kusudama;
}
```

### B3.3 Enhanced Constraint Integration

**Expose constraint properties through base class:**
```cpp
void EWBIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
    // Add constraint properties that map to joint limitations
    int total_joints = _count_total_joints();
    
    p_list->push_back(PropertyInfo(Variant::INT, "constraint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Constraints,constraints/,static,const"));
    
    for (int i = 0; i < total_joints; i++) {
        String constraint_path = "constraints/" + itos(i) + "/";
        
        p_list->push_back(PropertyInfo(Variant::STRING, constraint_path + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
        p_list->push_back(PropertyInfo(Variant::FLOAT, constraint_path + "twist_start", PROPERTY_HINT_RANGE, "-180,180,0.1,radians_as_degrees"));
        p_list->push_back(PropertyInfo(Variant::FLOAT, constraint_path + "twist_range", PROPERTY_HINT_RANGE, "0,360,0.1,radians_as_degrees"));
        p_list->push_back(PropertyInfo(Variant::ARRAY, constraint_path + "open_cones", PROPERTY_HINT_TYPE_STRING, "IKLimitCone3D"));
        p_list->push_back(PropertyInfo(Variant::FLOAT, constraint_path + "resistance", PROPERTY_HINT_RANGE, "0,1,0.01"));
    }
    
    // Add pin properties and base class properties
    _add_pin_properties(p_list);
    ManyBoneIK3D::_get_property_list(p_list);
}
```

---

## Phase B4: EWBIK3D Simplification

### B4.1 Focus on Solving Algorithm

**Simplified EWBIK3D structure:**
```cpp
class EWBIK3D : public ManyBoneIK3D {
    GDCLASS(EWBIK3D, ManyBoneIK3D);

private:
    // EWBIK3D-specific solving parameters
    float default_damp = Math::PI;
    int stabilizing_pass_count = 0;
    bool constraint_mode = false;
    
    // Remove all parallel systems - use base class instead
    // DELETE: Vector<Ref<IKEffectorTemplate3D>> pins;
    // DELETE: Vector<Ref<IKBoneSegment3D>> segmented_skeletons;
    // DELETE: PackedStringArray constraint_names;
    // DELETE: PackedFloat32Array joint_twist;
    // DELETE: TypedArray<IKLimitCone3D> kusudama_open_cones;

protected:
    // Focus on solving algorithm
    virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination) override;
    
    // Property mapping for user-friendly interface
    virtual bool _set(const StringName &p_path, const Variant &p_value) override;
    virtual bool _get(const StringName &p_path, Variant &r_ret) const override;
    virtual void _get_property_list(List<PropertyInfo> *p_list) const override;

public:
    // EWBIK3D-specific methods
    void set_default_damp(float p_damp);
    float get_default_damp() const;
    void set_stabilizing_pass_count(int p_count);
    int get_stabilizing_pass_count() const;
    void set_constraint_mode(bool p_enabled);
    bool is_constraint_mode() const;
};
```

### B4.2 Code Reduction

**Massive simplification:**
- Remove ~70% of EWBIK3D code
- Eliminate all parallel data structures
- Remove complex segment management
- Remove duplicate constraint system
- Remove duplicate pin system

**Before (complex):**
```cpp
// EWBIK3D had parallel systems for everything
Vector<Ref<IKEffectorTemplate3D>> pins;                    // Duplicate of base settings
Vector<Ref<IKBoneSegment3D>> segmented_skeletons;         // Duplicate of base Y-branching
PackedStringArray constraint_names;                        // Duplicate of base constraints
PackedFloat32Array joint_twist;                           // Duplicate of base constraints
TypedArray<IKLimitCone3D> kusudama_open_cones;           // Duplicate of base constraints

// Complex segment management
void _create_segmented_skeletons();
void _update_segmented_skeletons();
void _solve_segments(double p_delta);
// ... hundreds of lines of duplicate logic
```

**After (simplified):**
```cpp
// EWBIK3D uses base class systems directly
// No parallel data structures
// Focus purely on solving algorithm

virtual void _solve_iteration(...) override {
    // Sophisticated EWBIK solving for linear chain
    // Base class handles Y-branching coordination
}
```

### B4.3 Enhanced Capabilities

**EWBIK3D gains from base class:**
- Automatic Y-branching coordination
- Enhanced property system with validation
- Optimal skeleton analysis
- Universal constraint system
- Better performance through shared optimization

---

## Phase B5: Migration and Testing

### B5.1 Migration Strategy

**Step-by-step migration:**
1. **Backup**: Create backup of current EWBIK3D implementation
2. **Pin migration**: Replace pin system with base class mapping
3. **Segment removal**: Remove segmented skeleton system
4. **Constraint migration**: Replace constraint arrays with JointLimitation3D
5. **Testing**: Verify all EWBIK3D features work with base class
6. **Cleanup**: Remove unused code and simplify structure

### B5.2 Compatibility Testing

**Test scenarios:**
```cpp
// Test 1: Simple linear chain (should work unchanged)
ewbik.set_pin_count(1);
ewbik.set_pin_bone_name(0, "hand_L");
ewbik.set_pin_target_node(0, NodePath("target"));

// Test 2: Y-branching (should use base class coordination)
ewbik.set_pin_count(2);
ewbik.set_pin_bone_name(0, "hand_L");
ewbik.set_pin_bone_name(1, "hand_R");
// Base class should auto-detect shared spine root

// Test 3: Complex constraints (should use IKKusudama3D)
ewbik.set_constraint_count(5);
ewbik.set_joint_twist(0, Math::PI / 4);
ewbik.set_kusudama_open_cones(1, cone_array);
// Should map to appropriate joint limitations
```

### B5.3 Performance Validation

**Verify improvements:**
- Reduced memory usage (no duplicate data structures)
- Faster initialization (no segment building)
- Better solving coordination (base class Y-branching)
- Simplified debugging (single source of truth)

---

## Implementation Checklist

### Phase B1: Pin System Migration
- [ ] Remove pins array from EWBIK3D
- [ ] Map pin properties to base class settings
- [ ] Implement auto-population from pin interface
- [ ] Test pin-style property access

### Phase B2: Remove Segmented Skeletons
- [ ] Delete segmented_skeletons system
- [ ] Remove segment-based solving logic
- [ ] Simplify EWBIK3D initialization
- [ ] Use base class Y-branching coordination

### Phase B3: Constraint System Migration
- [ ] Remove constraint arrays
- [ ] Map constraints to JointLimitation3D
- [ ] Implement IKKusudama3D integration
- [ ] Test constraint property mapping

### Phase B4: EWBIK3D Simplification
- [ ] Focus on solving algorithm only
- [ ] Remove duplicate systems
- [ ] Simplify class structure
- [ ] Verify enhanced capabilities

### Phase B5: Migration and Testing
- [ ] Step-by-step migration
- [ ] Compatibility testing
- [ ] Performance validation
- [ ] Documentation updates

## Success Criteria

1. **Code Reduction**: ~70% reduction in EWBIK3D code size
2. **Feature Preservation**: All EWBIK3D capabilities maintained
3. **Performance**: Equal or better performance than before
4. **Compatibility**: Existing EWBIK3D setups continue to work
5. **Simplicity**: EWBIK3D becomes focused solver

## Risk Mitigation

1. **Feature Loss**: Comprehensive testing of all EWBIK3D features
2. **Performance Regression**: Benchmarking before/after migration
3. **Compatibility**: Extensive testing with existing projects
4. **Complexity**: Clear migration path and documentation

## Dependencies

- **Plan A Complete**: Base class has Y-branching support
- **IKKusudama3D**: Properly inherits from JointLimitation3D
- **Base Class Enhanced**: Property system supports pin-style access
- **Testing Framework**: Comprehensive test suite for validation

This plan transforms EWBIK3D from a complex system with parallel data structures into a focused, efficient solver that leverages the enhanced base class capabilities.
