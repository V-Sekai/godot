# MultiIK Examples and Documentation

This directory contains examples and documentation for using the MultiIK (Multi-Effector Inverse Kinematics) system in Godot.

## Overview

The MultiIK system extends the existing EWBIK3D solver to support multiple coordinated effectors. Key features include:

- **Automatic Junction Detection**: System automatically identifies skeleton branch points
- **Chain Splitting**: Effector lists are split at junctions for proper solve ordering
- **Multi-Effector Coordination**: Multiple effectors solved simultaneously with proper dependencies
- **Pole Target Support**: Control twist orientation at junction points
- **Priority Weighting**: Each effector can have different influence weights

## Basic Usage

### Enabling MultiIK

```gdscript
# Get the IK solver node
var ik_solver = $EWBIK3D

# Enable MultiIK mode
ik_solver.set_multi_ik_enabled(true)

# Set the root bone (optional, will auto-detect if not set)
ik_solver.set_root_bone_name("Hips")
```

### Adding Effectors

```gdscript
# Add effectors with different weights
ik_solver.add_effector("LeftHand", $LeftHandTarget, 1.0)
ik_solver.add_effector("RightHand", $RightHandTarget, 1.0)
ik_solver.add_effector("Head", $HeadTarget, 0.8)
ik_solver.add_effector("LeftFoot", $LeftFootTarget, 0.6)
ik_solver.add_effector("RightFoot", $RightFootTarget, 0.6)
```

### Managing Effectors

```gdscript
# Modify effector properties
ik_solver.set_effector_target(0, $NewLeftHandTarget)
ik_solver.set_effector_weight(0, 0.9)

# Remove effectors
ik_solver.remove_effector(2)  # Remove head effector

# Clear all effectors
ik_solver.clear_effectors()
```

### Pole Targets

```gdscript
# Set pole targets for elbow/knee control
ik_solver.set_pole_target("LeftArm", $LeftElbowPole)
ik_solver.set_pole_target("RightArm", $RightElbowPole)
ik_solver.set_pole_target("LeftThigh", $LeftKneePole)
ik_solver.set_pole_target("RightThigh", $RightKneePole)

# Check if pole target exists
if ik_solver.has_pole_target("LeftArm"):
    var pole_path = ik_solver.get_pole_target("LeftArm")

# Remove pole target
ik_solver.remove_pole_target("LeftArm")
```

### Chain Priorities

```gdscript
# Set chain priorities (higher values = higher priority)
ik_solver.set_chain_priority(0, 2.0)  # Left arm high priority
ik_solver.set_chain_priority(1, 2.0)  # Right arm high priority
ik_solver.set_chain_priority(2, 1.5)  # Head medium priority
ik_solver.set_chain_priority(3, 1.0)  # Left leg normal priority
ik_solver.set_chain_priority(4, 1.0)  # Right leg normal priority

# Get current priority
var priority = ik_solver.get_chain_priority(0)
```

## Advanced Usage

### Junction Analysis

```gdscript
# Get information about skeleton structure
var junctions = ik_solver.get_junction_bones()
var chains = ik_solver.get_effector_chains()

print("Junction bones: ", junctions)
print("Effector chains: ", chains.size())
```

### Integration with Animation

```gdscript
# MultiIK works with Godot's animation system
# Set up IK during animation playback
func _process(delta):
    if animation_player.is_playing():
        # Update effector targets based on animation or game logic
        update_effector_targets()

        # IK will automatically solve each frame
        # No additional calls needed
```

### Performance Optimization

```gdscript
# Adjust solver parameters for performance
ik_solver.set_iterations_per_frame(10)  # Fewer iterations for better performance
ik_solver.set_default_damp(deg_to_rad(3.0))  # Tighter damping for stability

# Use lower weights for less critical effectors
ik_solver.add_effector("LeftFoot", $LeftFootTarget, 0.3)  # Low influence
ik_solver.add_effector("RightFoot", $RightFootTarget, 0.3)  # Low influence
```

## Example Scenes

### 1. Basic Biped Character

```gdscript
# Simple biped setup
ik_solver.set_multi_ik_enabled(true)
ik_solver.set_root_bone_name("Hips")

# Hands and feet
ik_solver.add_effector("LeftHand", $LeftHandTarget, 1.0)
ik_solver.add_effector("RightHand", $RightHandTarget, 1.0)
ik_solver.add_effector("LeftFoot", $LeftFootTarget, 0.8)
ik_solver.add_effector("RightFoot", $RightFootTarget, 0.8)

# Pole targets for natural poses
ik_solver.set_pole_target("LeftArm", $LeftElbowPole)
ik_solver.set_pole_target("RightArm", $RightElbowPole)
```

### 2. Quadruped Character

```gdscript
# Quadruped setup
ik_solver.set_multi_ik_enabled(true)
ik_solver.set_root_bone_name("Spine")

# All four legs
ik_solver.add_effector("FrontLeftFoot", $FrontLeftTarget, 1.0)
ik_solver.add_effector("FrontRightFoot", $FrontRightTarget, 1.0)
ik_solver.add_effector("BackLeftFoot", $BackLeftTarget, 1.0)
ik_solver.add_effector("BackRightFoot", $BackRightTarget, 1.0)

# Head for orientation
ik_solver.add_effector("Head", $HeadTarget, 0.6)
```

### 3. Procedural Animation

```gdscript
# Dynamic effector management
func grab_object(object_position):
    # Add temporary effector for grabbing
    ik_solver.add_effector("RightHand", object_position, 1.0)

    # Wait for IK to settle
    await get_tree().create_timer(0.5).timeout

    # Adjust weight for holding
    ik_solver.set_effector_weight(ik_solver.get_effector_count() - 1, 0.8)

func release_object():
    # Remove temporary effector
    ik_solver.remove_effector(ik_solver.get_effector_count() - 1)
```

## Best Practices

### 1. Effector Placement
- Place effector targets slightly in front of actual contact points
- Use consistent coordinate spaces for all targets
- Consider effector reach when positioning targets

### 2. Weight Management
- Use weights between 0.0 and 1.0
- Higher weights (closer to 1.0) for primary effectors
- Lower weights for secondary effectors that provide hints

### 3. Pole Targets
- Position pole targets to control bend direction
- Use pole targets for elbows, knees, and other joint chains
- Pole targets work best when positioned perpendicular to the chain

### 4. Performance
- Limit effector count based on performance requirements
- Use lower iteration counts for real-time applications
- Consider using MultiIK only when multiple effectors are needed

### 5. Stability
- Start with higher damping values for stability
- Gradually reduce damping as needed for responsiveness
- Use pole targets to prevent unnatural joint configurations

## Troubleshooting

### Common Issues

1. **Effectors not reaching targets**
   - Check if targets are within reachable distance
   - Verify bone chain connectivity
   - Ensure proper skeleton hierarchy

2. **Unstable solutions**
   - Increase damping values
   - Add pole targets for better control
   - Reduce effector weights

3. **Poor performance**
   - Reduce iterations per frame
   - Limit number of effectors
   - Use lower weights for less critical effectors

4. **Unexpected behavior**
   - Verify MultiIK is enabled
   - Check effector bone names match skeleton
   - Ensure targets are properly positioned

### Debug Information

```gdscript
# Get debug information
print("MultiIK enabled: ", ik_solver.get_multi_ik_enabled())
print("Effector count: ", ik_solver.get_effector_count())
print("Junction bones: ", ik_solver.get_junction_bones())
print("Chain count: ", ik_solver.get_effector_chains().size())
```

## API Reference

### Core Methods
- `set_multi_ik_enabled(bool)` - Enable/disable MultiIK mode
- `add_effector(bone_name, target_path, weight)` - Add effector
- `remove_effector(index)` - Remove effector by index
- `clear_effectors()` - Remove all effectors

### Configuration
- `set_root_bone_name(bone_name)` - Set root bone for IK chain
- `set_pole_target(bone_name, pole_path)` - Set pole target
- `set_chain_priority(chain_index, priority)` - Set chain priority

### Queries
- `get_effector_count()` - Get number of effectors
- `get_effector_bone_name(index)` - Get effector bone name
- `get_junction_bones()` - Get detected junction bones
- `get_effector_chains()` - Get effector chains

## Migration from Single Effector

If migrating from single-effector IK:

1. Enable MultiIK mode
2. Replace single pin setup with multiple effectors
3. Adjust weights as needed
4. Add pole targets for better control
5. Test and tune performance settings

The MultiIK system is designed to be a drop-in enhancement that maintains backward compatibility while providing advanced multi-effector capabilities.
