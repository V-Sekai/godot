# AriaEwbik MultiIK Design Strategy

## Context and Background

This document outlines our strategy for implementing MultiIK (Multi-Effector Inverse Kinematics) in the AriaEwbik system, based on insights from the Godot ManyBoneIK3D implementation and the broader EWBIK (Entirely Wahba's-problem Based Inverse Kinematics) research.

### Current Godot ManyBoneIK3D Implementation Insights

**Existing GUI System:**
- **Sphere Visualization**: Current GUI uses small spheres (0.02f scale) to represent kusudama constraints
- **Shader Rendering**: Custom shader system for cone visualization with RGBA color data
- **Bone Selection**: Click-to-select bones with yellow/blue visual feedback
- **Edit Mode**: Toggle button for switching between view and edit modes
- **Real-time Updates**: Gizmos update dynamically with bone transformations

**Technical Architecture:**
- **Gizmo Plugin System**: EditorNode3DGizmoPlugin with Node3DEditor integration
- **Mesh Generation**: Procedural sphere creation with 8×8 ring/radial segments
- **Coordinate Transforms**: Complex skeleton ↔ gizmo ↔ constraint space transformations
- **Material System**: ShaderMaterial with custom parameters for constraint visualization

**Current Limitations:**
- Basic sphere visualization (not advanced cone rendering)
- Limited multi-effector management
- No pole target visualization
- Manual constraint setup process

## Current Status

### EWBIK Implementation Progress

- ✅ **Phase 1**: Internal modules complete (Segmentation, Solver, Kusudama, Propagation)
- ✅ **Phase 1.5**: External API implementation (IN PROGRESS - Critical Priority)
- 🔄 **MultiIK Design**: Strategy documented and ready for implementation

### Technical Foundation

- **37 tests passing** across 4 modules
- **Decomposition algorithm** implemented for multi-effector coordination
- **AriaJoint integration** for optimized transform calculations
- **QCP algorithm** ready for Wahba's problem solving

**🔥 NEW CAPABILITY UNLOCKED: Godot + Elixir Side-by-Side Development**

We now have full code access to both:
- **Godot ManyBoneIK3D**: Complete C++ implementation with GUI, shader system, and gizmo plugins
- **AriaEwbik (Elixir)**: EWBIK algorithm foundation with AriaJoint/AriaQCP integration

This enables powerful cross-platform development workflows:
- **Algorithm Migration**: Port proven EWBIK algorithms from Elixir to Godot C++
- **GUI Enhancement**: Leverage Godot's visualization system for advanced constraint editing
- **Performance Optimization**: Combine Elixir's mathematical precision with Godot's real-time performance
- **Standards Compliance**: Ensure VRM1, glTF 2.0, and IEEE-754 compatibility across platforms

## MultiIK Design Strategy

### Core Principles (from Godot ManyBoneIK3D)

> "And after then I will continue to experiment with MultiIK in https://github.com/TokageItLab/godot/tree/multi-ik-3d. The things that have been almost decided are: Having a GUI that allows you to set one Root and multiple End Effectors, thereby eliminating duplicated joint and generating a split list at each junction, and allowing the use to set pole targets and limitations within that list. Then, there are too many things to discuss about the GUI, so we should set up a meeting somewhere and share mockups in the near future."

### Key Design Decisions

#### 1. Single Root, Multiple End Effectors

- **Root Node**: Single skeleton root for the entire IK chain
- **End Effectors**: Multiple target points (hands, feet, head, etc.)
- **Automatic Junction Detection**: System identifies branch points in skeleton hierarchy

#### 2. Junction-Based Chain Splitting

- **Branch Detection**: Automatic identification of skeleton junctions
- **Chain Segmentation**: Split effector lists at each junction
- **Dependency Management**: Ensure proper solve order (parents before children)

#### 3. Pole Target and Limitation Support

- **Pole Targets**: Control twist/swing orientation at each junction
- **Joint Limitations**: Per-joint angle constraints (Kusudama cones)
- **Priority Weighting**: Effector opacity and influence control

## Implementation Architecture

### Multi-Effector Solver Hierarchy

```elixir
# ChainIK → ManyBoneIK → BranchIK progression
defmodule AriaEwbik.MultiEffectorSolver do
  def solve_chain_ik(skeleton, single_chain) do
    # Simple chain solving (no branching)
    solve_simple_chain(skeleton, single_chain)
  end

  def solve_many_bone_ik(skeleton, effector_targets) do
    # Complex multi-effector solving
    {groups, effector_groups} = Decomposition.decompose_multi_effector(skeleton, effector_targets)
    solve_with_groups(skeleton, groups, effector_groups)
  end

  def solve_branch_ik(skeleton, effector_targets, branch_points) do
    # Extended ManyBoneIK for branched skeletons
    solve_branched_multi_effector(skeleton, effector_targets, branch_points)
  end
end
```

### GUI Design Requirements (Based on Current Godot Implementation)

#### Core Features (Existing in Godot ManyBoneIK3D)

1. **Sphere-based Kusudama Visualization**: 0.02f scale spheres with 8×8 ring/radial segments for smooth constraint rendering
2. **Bone Selection System**: Click-to-select bones with visual feedback (yellow selected, blue unselected)
3. **Edit Mode Toggle**: Button to switch between view and edit modes with joint handle display
4. **Shader-based Constraint Rendering**: Custom shader with RGBA color data for kusudama cone visualization
5. **Real-time Gizmo Updates**: Dynamic constraint visualization that updates with bone transformations

#### Advanced Features (To Be Implemented - Focused on Core IK)

1. **Root Selection Interface**: Visual picker for IK chain root with skeleton hierarchy display
2. **Multi-Effector Management**: Add/remove multiple end effectors with drag-and-drop
3. **Junction Visualization**: Show automatic branch detection with colored indicators
4. **Dual Constraint Editing**: Interactive editing of twist fins and swing cones
5. **Current Position Tracking**: Real-time display of current vs limit positions
6. **Constraint Library**: Preset anatomical constraints (humanoid, quadruped, etc.)
7. **Real-time IK Preview**: Live solution visualization with performance metrics

## Technical Implementation Plan

### Phase 0: QCP Algorithm Migration (FOUNDATIONAL TASK)

- [ ] **Migrate QCP Insights**: Port Quaternion Characteristic Polynomial algorithm insights from Elixir (69/69 tests passing) to C++ Many Bone IK
- [ ] **Test Suite Translation**: Convert comprehensive Elixir QCP test suite to C++ unit tests
- [ ] **Performance Benchmarking**: Establish baseline performance metrics for C++ QCP implementation
- [ ] **Integration Validation**: Ensure C++ QCP produces identical results to Elixir reference implementation
- [ ] **Documentation**: Document QCP algorithm insights and mathematical foundations for C++ implementation

### Phase 1: Core MultiIK Algorithm

- [ ] Implement junction detection algorithm
- [ ] Create effector list splitting logic
- [ ] Add pole target support to solver
- [ ] Integrate with existing decomposition algorithm

**Phase 1.6: Cross-Platform Algorithm Migration (NEW - Godot + Elixir Synergy)**

- [ ] **QCP Algorithm Migration**: Port Elixir AriaQCP (69/69 tests) to Godot C++ ManyBoneIK
  - [ ] Analyze Elixir QCP implementation for algorithmic insights
  - [ ] Translate quaternion mathematics to C++ with IEEE-754 compliance
  - [ ] Implement comprehensive test suite translation (69 tests)
  - [ ] Validate numerical accuracy between Elixir and C++ implementations
  - [ ] Performance benchmark C++ QCP against Elixir reference
- [ ] **EWBIK Decomposition Algorithm**: Migrate multi-effector coordination from Elixir to Godot
  - [ ] Port effector group creation and solve order determination
  - [ ] Implement junction detection and effector list splitting in C++
  - [ ] Translate priority weighting and opacity coordination patterns
  - [ ] Cross-validate algorithm results between platforms
- [ ] **Constraint System Integration**: Unified Kusudama implementation across platforms
  - [ ] Standardize cone-based constraint mathematics (Elixir ↔ C++)
  - [ ] Implement shared constraint serialization format
  - [ ] Validate constraint application consistency
  - [ ] Performance optimize constraint evaluation in both environments
- [ ] **Dual Constraint Visualization**: Implement twist fins + swing cones in Godot
  - [ ] **Twist Axis Limits**: Fin-based visualization showing current position between min/max limits
  - [ ] **Swing Kusudama Limits**: Cone visualization showing current vs allowable swing angles
  - [ ] Update shader system for dual constraint rendering
  - [ ] Add real-time current position tracking for both constraint types
- [ ] **Cross-Platform Testing Framework**: Ensure algorithm consistency
  - [ ] Create shared test data format (skeleton definitions, effector targets)
  - [ ] Implement result comparison utilities between Elixir and Godot
  - [ ] Establish performance benchmarking across platforms
  - [ ] Validate standards compliance (VRM1, glTF 2.0, IEEE-754)

### Investigation: Running Single Scripts from GDScript During SCons Compilation

**Objective**: Enable execution of individual GDScript files during the Godot build process for testing, validation, and utility purposes.

**Current Godot Build System Analysis:**

#### **SCons Build Integration Methods:**

1. **Custom Build Commands in SCsub**
   ```python
   # SCsub - Module build configuration
   Import('env')

   # Add custom command to run GDScript during build
   def run_gdscript_test(target, source, env):
       """Execute GDScript test file during compilation"""
       import subprocess
       import os

       # Path to Godot executable (built or system)
       godot_exe = env.get('GODOT_EXE', 'godot')

       # Run specific GDScript file
       test_script = str(source[0])
       cmd = [godot_exe, '--script', test_script, '--no-window']

       # Execute and capture output
       result = subprocess.run(cmd, capture_output=True, text=True)

       # Check for errors
       if result.returncode != 0:
           print(f"GDScript test failed: {test_script}")
           print(f"STDOUT: {result.stdout}")
           print(f"STDERR: {result.stderr}")
           return 1

       print(f"GDScript test passed: {test_script}")
       return 0

   # Register build command
   env.Command('test_result', 'test_script.gd', run_gdscript_test)
   ```

2. **GDScript Execution via Command Line**
   ```bash
   # Run single GDScript file
   godot --script path/to/script.gd --no-window

   # With custom arguments
   godot --script test_validation.gd --test-case kusudama_constraints

   # Headless mode for CI/CD
   godot --script benchmark.gd --headless --quiet
   ```

3. **Module-Specific Test Runner**
   ```python
   # many_bone_ik/SCsub
   Import('env')

   # Test runner for Many Bone IK module
   def run_many_bone_ik_tests(target, source, env):
       """Run all Many Bone IK tests during build"""
       import subprocess
       import glob

       godot_exe = env.get('GODOT_EXE', 'godot')

       # Find all test scripts in module
       test_files = glob.glob('tests/*.gd')

       for test_file in test_files:
           print(f"Running test: {test_file}")
           cmd = [godot_exe, '--script', test_file, '--no-window']
           result = subprocess.run(cmd, capture_output=True, text=True)

           if result.returncode != 0:
               print(f"Test failed: {test_file}")
               print(f"Output: {result.stdout}")
               print(f"Errors: {result.stderr}")
               return 1

       print("All Many Bone IK tests passed!")
       return 0

   # Add to build targets
   env.Command('many_bone_ik_tests', Glob('tests/*.gd'), run_many_bone_ik_tests)
   ```

#### **GDScript Test Script Structure:**

```gdscript
# tests/test_kusudama_constraints.gd
extends SceneTree

func _init():
    print("Running Kusudama constraint tests...")

    # Test constraint validation
    var result = test_constraint_validation()
    if not result:
        print("Constraint validation test failed!")
        quit(1)

    # Test limit enforcement
    result = test_limit_enforcement()
    if not result:
        print("Limit enforcement test failed!")
        quit(1)

    print("All Kusudama constraint tests passed!")
    quit(0)

func test_constraint_validation():
    # Test implementation
    return true

func test_limit_enforcement():
    # Test implementation
    return true
```

#### **Build System Integration Options:**

1. **Pre-build Validation**
   ```python
   # Run before main compilation
   env.AddPreAction('many_bone_ik', run_many_bone_ik_tests)
   ```

2. **Post-build Testing**
   ```python
   # Run after successful compilation
   env.AddPostAction('many_bone_ik', run_many_bone_ik_tests)
   ```

3. **Conditional Execution**
   ```python
   # Only run if GODOT_TEST=1 environment variable is set
   if env.get('GODOT_TEST', '0') == '1':
       env.Command('test_results', test_files, run_tests)
   ```

#### **Investigation Tasks:**

- [ ] **Analyze Current SCsub**: Examine existing build configuration for Many Bone IK module
- [ ] **Test Godot CLI Options**: Verify `--script` and `--no-window` functionality
- [ ] **Implement Test Runner**: Create SCsub integration for running GDScript tests
- [ ] **Validate Build Integration**: Ensure tests run correctly during scons compilation
- [ ] **Document Usage**: Create guidelines for developers on running individual scripts
- [ ] **CI/CD Integration**: Set up automated testing in build pipeline

#### **Benefits:**

- **Early Error Detection**: Catch issues during compilation rather than runtime
- **Automated Testing**: Ensure code quality with every build
- **Validation Scripts**: Run mathematical validation against known test cases
- **Performance Benchmarks**: Automated performance testing during build
- **Cross-Platform Consistency**: Validate behavior across different platforms

#### **Challenges to Investigate:**

- **Godot Executable Path**: How to reliably locate Godot binary during build
- **Headless Mode**: Ensuring proper headless execution without display
- **Error Handling**: Proper error reporting and build failure on test failures
- **Performance Impact**: Minimizing build time impact of running scripts
- **Dependency Management**: Ensuring test scripts have access to required modules

### Phase 2: GUI Framework (Based on Godot Gizmo Plugin Architecture)

- [ ] **Implement Gizmo Plugin System**: Create EditorNode3DGizmoPlugin with gizmo registration
- [ ] **Dual Constraint Mesh Generation**: Generate both fin-based twist indicators and cone-based swing limits
- [ ] **Enhanced Shader Material System**: Custom ShaderMaterial with RGBA color data for dual constraint rendering
- [ ] **Bone Selection Interface**: Click-to-select with visual feedback (yellow/blue color coding)
- [ ] **Edit Mode Toggle**: Button integration with Node3DEditor menu panel
- [ ] **Transform Coordinate Handling**: Complex skeleton ↔ gizmo ↔ constraint space transformations
- [ ] **Real-time Gizmo Updates**: Dynamic constraint visualization with bone pose changes
- [ ] **Dual Handle System**: Interactive handles for both twist fin and swing cone editing

### Phase 3: Advanced Features

- [ ] Effector priority weighting system
- [ ] Enhanced dual constraint editing tools
- [ ] Constraint library and presets
- [ ] Real-time IK performance monitoring

### Phase 4: Optimization and Testing

- [ ] Performance optimization for complex rigs
- [ ] Comprehensive test suite for multi-effector scenarios
- [ ] Integration testing with AriaJoint and AriaQCP
- [ ] User experience validation

## Use Cases and Applications

### Primary Scenarios

1. **Character Animation**: Full-body procedural IK for games
2. **Bouldering/Climbing**: Complex hand-foot coordination
3. **Foot Placement**: Automatic foot positioning on uneven terrain
4. **Interactive Characters**: Real-time response to environmental changes

### Technical Requirements

1. **Real-time Performance**: 30+ FPS for character animation
2. **Complex Rigs**: Support for 100+ joint skeletons
3. **Stability**: Robust convergence for edge cases
4. **Flexibility**: Easy setup for different character types

## Integration with Existing Systems

### AriaJoint Integration

- **HierarchyManager**: Optimized transform calculations
- **Batch Updates**: Efficient dirty flag propagation
- **Nested Sets**: Fast subtree operations

### AriaQCP Integration

- **Wahba's Problem**: Multi-effector coordinate solving
- **Quaternion Mathematics**: Stable orientation calculations
- **Performance**: Optimized for real-time use

### AriaMath Integration

- **IEEE-754 Compliance**: Numerical stability
- **Quaternion Operations**: Dot, angle, normalize functions
- **Matrix Operations**: Transform calculations

## Challenges and Solutions

### Technical Challenges

1. **Branch Detection**: Identifying skeleton junctions automatically
2. **Solve Order**: Determining optimal processing sequence
3. **Convergence**: Ensuring stable solutions for complex scenarios
4. **Performance**: Maintaining real-time performance with multiple effectors

### GUI Challenges

1. **Complex Visualization**: Showing multi-effector relationships
2. **User Experience**: Intuitive controls for complex IK setup
3. **Real-time Feedback**: Live preview of IK solutions
4. **Constraint Editing**: Visual tools for Kusudama cones

## Success Criteria

### Functional Requirements

- [ ] Single root with multiple end effectors
- [ ] Automatic junction detection and chain splitting
- [ ] Dual constraint visualization (twist fins + swing cones)
- [ ] Real-time IK solving at 30+ FPS
- [ ] Stable convergence for complex character rigs

### User Experience Requirements

- [ ] Intuitive GUI for IK chain setup
- [ ] Visual feedback for junction detection
- [ ] Easy constraint editing and visualization
- [ ] Integration with existing animation workflows

## Future Considerations

### Extended Features

1. **Animation Baking**: Convert procedural IK to keyframe animation
2. **Motion Capture Integration**: Use IK for retargeting mocap data
3. **Physics Integration**: Combine IK with physical simulation
4. **Machine Learning**: AI-assisted IK solving for complex poses

### Research Opportunities

1. **Advanced Constraints**: Soft constraints and spring systems
2. **Predictive IK**: Anticipate and prevent unnatural poses
3. **Adaptive IK**: Learn from user corrections and preferences
4. **Multi-character IK**: Coordination between multiple characters

## Conclusion

The MultiIK design strategy provides a solid foundation for implementing sophisticated multi-effector inverse kinematics in the AriaEwbik system. By following the Godot ManyBoneIK3D approach of single root with multiple end effectors, automatic junction detection, and dual constraint visualization (twist fins + swing cones), we can create a powerful and user-friendly IK system for complex character animation scenarios.

The cross-platform synergy between Godot and Elixir enables us to leverage the best of both worlds: Elixir's mathematical precision and comprehensive testing for algorithm development, combined with Godot's real-time performance and advanced GUI capabilities for production deployment.

The phased implementation approach ensures we build a robust foundation before adding advanced GUI features and optimization, resulting in a production-ready MultiIK system that serves the needs of game developers and animation professionals.
