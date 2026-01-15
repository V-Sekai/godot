extends SceneTree

func _init():
    print("[FUZZ CYLINDER] Testing CSGSculptedCylinder3D with default parameters...")

    var csg_root = ClassDB.instantiate("CSGCombiner3D")
    set_current_scene(csg_root)
    var inst = ClassDB.instantiate("CSGSculptedCylinder3D")
    if inst:
        csg_root.add_child(inst)
        await self.process_frame
        var mesh = inst.get_mesh() if inst.has_method("get_mesh") else null
        if mesh:
            var result = CSGShape3D.validate_manifold_mesh(mesh)
            print("[CSGSculptedCylinder3D] valid=" + str(result["valid"]) + ", errors=" + str(result["errors"]))
        else:
            print("[CSGSculptedCylinder3D] has no mesh (no mesh means manifold validation failed or mesh generation failed)")
        inst.queue_free()
        await self.process_frame

    print("[FUZZ CYLINDER] Done.")
    quit()
