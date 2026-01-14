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
        var mesh = null
        if inst.has_method("get_mesh"):
            mesh = inst.get_mesh()
        if mesh != null:
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
                var mesh = null
                if root.has_method("get_mesh"):
                    mesh = root.get_mesh()
                if mesh != null:
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

    log_msg("[FUZZ] Spawning sculpted_primitives with default parameters...")

    var types = [
        "CSGSculptedBox3D",
        "CSGSculptedSphere3D",
        "CSGSculptedCylinder3D",
        "CSGSculptedPrism3D",
        "CSGSculptedTorus3D",
        "CSGSculptedTube3D",
        "CSGSculptedRing3D"
        # TODO: Add "CSGSculptedTexture3D" if available
    ]
    # Create a CSGCombiner3D as the root node for the scene tree
    var csg_root = ClassDB.instantiate("CSGCombiner3D")
    set_current_scene(csg_root)
    for type_name in types:
        var inst = ClassDB.instantiate(type_name)
        if inst == null:
            log_msg("[ERROR] Could not instantiate " + type_name)
            continue
        csg_root.add_child(inst)
        await self.process_frame
        var mesh = null
        if inst.has_method("get_mesh"):
            mesh = inst.get_mesh()
        if mesh != null:
            var result = CSGShape3D.validate_manifold_mesh(mesh)
            if result:
                log_msg("[" + type_name + "] valid=" + str(result["valid"]) + ", errors=" + str(result["errors"]))
            else:
                log_msg("[" + type_name + "] validate_manifold_mesh returned null")
        else:
            log_msg("[" + type_name + "] has no mesh (no mesh means manifold validation failed or mesh generation failed)")
        inst.queue_free()
        await self.process_frame

    log_msg("[INFO] Run with --verbose for more details on leaks and errors.")

    log_msg("[FUZZ] Done.")
    if log_file:
        log_file.close()
    quit()
