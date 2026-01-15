extends SceneTree

var log_file = null

func log_msg(msg):
    print(msg)
    if log_file:
        log_file.store_line(str(msg))

func _init():
    var log_path = "user://fuzz_ring.log"
    log_file = FileAccess.open(log_path, FileAccess.WRITE)

    log_msg("[FUZZ RING] Testing CSGSculptedRing3D with default parameters...")

    var csg_root = ClassDB.instantiate("CSGCombiner3D")
    set_current_scene(csg_root)
    var inst = ClassDB.instantiate("CSGSculptedRing3D")
    if inst:
        csg_root.add_child(inst)
        await self.process_frame
        var mesh = inst.get_mesh() if inst.has_method("get_mesh") else null
        if mesh:
            var result = CSGShape3D.validate_manifold_mesh(mesh)
            log_msg("[CSGSculptedRing3D] valid=" + str(result["valid"]) + ", errors=" + str(result["errors"]))
        else:
            log_msg("[CSGSculptedRing3D] has no mesh (no mesh means manifold validation failed or mesh generation failed)")
        inst.queue_free()
        await self.process_frame

    log_msg("[FUZZ RING] Done.")
    if log_file:
        log_file.close()
    quit()
