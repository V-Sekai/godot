extends SceneTree

func _init():
    print("[FUZZ BOX] Testing CSGSculptedBox3D with default parameters...")

    var csg_root = ClassDB.instantiate("CSGCombiner3D")
    set_current_scene(csg_root)
    var inst = ClassDB.instantiate("CSGSculptedBox3D")
    if inst:
        csg_root.add_child(inst)
        await self.process_frame
        var mesh = inst.get_mesh() if inst.has_method("get_mesh") else null
        if mesh:
            var result = CSGShape3D.validate_manifold_mesh(mesh)
            print("[CSGSculptedBox3D] valid=" + str(result["valid"]) + ", errors=" + str(result["errors"]))
        else:
            print("[CSGSculptedBox3D] has no mesh (no mesh means manifold validation failed or mesh generation failed)")
        inst.queue_free()
        await self.process_frame

    print("[FUZZ BOX] Done.")
    quit()
