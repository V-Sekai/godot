extends SceneTree

# Fuzzing sculpted_primitives CSG nodes for non-manifold mesh cases
# Run with: bin/godot.windows.editor.x86_64.console.exe --script modules/sculpted_primitives/tests/gdscript/fuzz_sculpted_primitives.gd

var log_file = null

func log_msg(msg):
    print(msg)
    if log_file:
        log_file.store_line(str(msg))


func fuzz_primitive(type_name, param_sets):
    for params in param_sets:
        var inst = ClassDB.instantiate(type_name)
        if inst == null:
            log_msg("[ERROR] Could not instantiate " + type_name)
            continue
        # Set primitive-specific parameters
        for k in params:
            if inst.has_method("set_" + k):
                inst.call("set_" + k, params[k])
        # Set base parameters (example: profile_curve, hollow, etc.)
        if inst.has_method("set_profile_curve"): inst.set_profile_curve(0)
        if inst.has_method("set_hollow"): inst.set_hollow(0.0)
        # ...add more base param sweeps/randomization here...
        var mesh = inst.mesh if inst.has_method("get_mesh") else null
        if mesh:
            var result = null
            if "validate_manifold_mesh" in CSGShape3D:
                result = CSGShape3D.validate_manifold_mesh(mesh)
            elif Engine.has_singleton("CSGShape3D"):
                result = Engine.get_singleton("CSGShape3D").validate_manifold_mesh(mesh)
            if result and not result["is_manifold"]:
                log_msg("Non-manifold " + type_name + ", params=" + str(params) + ": " + str(result))

# Fuzz CSG operations (add, subtract, overlap) between two primitives
func fuzz_csg_ops(type_a, param_sets_a, type_b, param_sets_b):
    var ops = [CSGShape3D.OPERATION_UNION, CSGShape3D.OPERATION_SUBTRACTION, CSGShape3D.OPERATION_INTERSECTION]
    var op_names = ["add", "sub", "overlap"]
    for params_a in param_sets_a:
        for params_b in param_sets_b:
            for i in range(ops.size()):
                var op = ops[i]
                var op_name = op_names[i]
                var root = ClassDB.instantiate("CSGCombiner3D")
                var a = ClassDB.instantiate(type_a)
                var b = ClassDB.instantiate(type_b)
                if a == null or b == null or root == null:
                    log_msg("[ERROR] Could not instantiate " + type_a + " or " + type_b)
                    continue
                for k in params_a:
                    if a.has_method("set_" + k):
                        a.call("set_" + k, params_a[k])
                for k in params_b:
                    if b.has_method("set_" + k):
                        b.call("set_" + k, params_b[k])
                if a.has_method("set_operation"): a.set_operation(CSGShape3D.OPERATION_UNION)
                if b.has_method("set_operation"): b.set_operation(op)
                root.add_child(a)
                root.add_child(b)
                # Try to get the mesh from the combiner
                var mesh = root.mesh if root.has_method("get_mesh") else null
                if mesh:
                    var result = null
                    if "validate_manifold_mesh" in CSGShape3D:
                        result = CSGShape3D.validate_manifold_mesh(mesh)
                    elif Engine.has_singleton("CSGShape3D"):
                        result = Engine.get_singleton("CSGShape3D").validate_manifold_mesh(mesh)
                    if result and not result["is_manifold"]:
                        log_msg("Non-manifold CSG op " + op_name + ": " + type_a + str(params_a) + " + " + type_b + str(params_b) + ": " + str(result))

func _init():
    var log_path = "user://fuzz_sculpted_primitives.log"
    log_file = FileAccess.open(log_path, FileAccess.WRITE)

    log_msg("[FUZZ] Starting sculpted_primitives CSG fuzzing...")

    var rng = RandomNumberGenerator.new()
    rng.randomize()

    # Parameter sweeps and randomization for all primitives
    var box_param_sets = [
        {"size": Vector3.ZERO},
        {"size": Vector3(1,1,1)},
        {"size": Vector3(0.01,0.01,0.01)},
        {"size": Vector3(10,1,1)}
    ]
    # Add random box sizes
    for i in range(20):
        box_param_sets.append({"size": Vector3(rng.randf_range(-2, 20), rng.randf_range(-2, 20), rng.randf_range(-2, 20))})

    fuzz_primitive("CSGSculptedBox3D", box_param_sets)
    fuzz_primitive("CSGSculptedSphere3D", sphere_param_sets)
    # Fuzz CSG add, sub, overlap between box and sphere
    fuzz_csg_ops("CSGSculptedBox3D", box_param_sets, "CSGSculptedSphere3D", sphere_param_sets)

    var sphere_param_sets = [
        {"radius": 0.0},
        {"radius": 0.01},
        {"radius": 1.0},
        {"radius": 10.0}
    ]
    for i in range(20):
        sphere_param_sets.append({"radius": rng.randf_range(-2, 20)})
    fuzz_primitive("CSGSculptedSphere3D", sphere_param_sets)

    var cylinder_param_sets = [
        {"radius": 0.0, "height": 0.0},
        {"radius": 1.0, "height": 1.0},
        {"radius": 0.01, "height": 10.0}
    ]
    for i in range(20):
        cylinder_param_sets.append({"radius": rng.randf_range(-2, 20), "height": rng.randf_range(-2, 20)})
    fuzz_primitive("CSGSculptedCylinder3D", cylinder_param_sets)

    var prism_param_sets = [
        {"size": Vector3.ZERO},
        {"size": Vector3(1,1,1)},
        {"size": Vector3(0.01,0.01,0.01)},
        {"size": Vector3(10,1,1)}
    ]
    for i in range(20):
        prism_param_sets.append({"size": Vector3(rng.randf_range(-2, 20), rng.randf_range(-2, 20), rng.randf_range(-2, 20))})
    fuzz_primitive("CSGSculptedPrism3D", prism_param_sets)

    var torus_param_sets = [
        {"inner_radius": 0.0, "outer_radius": 0.0},
        {"inner_radius": 1.0, "outer_radius": 2.0},
        {"inner_radius": 0.01, "outer_radius": 10.0}
    ]
    for i in range(20):
        torus_param_sets.append({"inner_radius": rng.randf_range(-2, 20), "outer_radius": rng.randf_range(-2, 20)})
    fuzz_primitive("CSGSculptedTorus3D", torus_param_sets)

    var tube_param_sets = [
        {"inner_radius": 0.0, "outer_radius": 0.0, "height": 0.0},
        {"inner_radius": 1.0, "outer_radius": 2.0, "height": 1.0},
        {"inner_radius": 0.01, "outer_radius": 10.0, "height": 10.0}
    ]
    for i in range(20):
        tube_param_sets.append({"inner_radius": rng.randf_range(-2, 20), "outer_radius": rng.randf_range(-2, 20), "height": rng.randf_range(-2, 20)})
    fuzz_primitive("CSGSculptedTube3D", tube_param_sets)

    var ring_param_sets = [
        {"inner_radius": 0.0, "outer_radius": 0.0, "height": 0.0},
        {"inner_radius": 1.0, "outer_radius": 2.0, "height": 1.0},
        {"inner_radius": 0.01, "outer_radius": 10.0, "height": 10.0}
    ]
    for i in range(20):
        ring_param_sets.append({"inner_radius": rng.randf_range(-2, 20), "outer_radius": rng.randf_range(-2, 20), "height": rng.randf_range(-2, 20)})
    fuzz_primitive("CSGSculptedRing3D", ring_param_sets)

    # TODO: Add CSGSculptedTexture3D and base parameter sweeps

    log_msg("[FUZZ] Done.")
    if log_file:
        log_file.close()
    quit()
