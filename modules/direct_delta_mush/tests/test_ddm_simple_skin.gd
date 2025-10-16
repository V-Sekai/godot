# test_ddm_simple_skin.gd - glTF Simple Skin specification test for Direct Delta Mush
extends Node3D

func _ready():
    # Create mesh following glTF Simple Skin specification
    var skin_mesh = create_gltf_simple_skin_mesh()

    # Create skeleton with 2 joints (bones) as per glTF spec
    var skeleton = create_gltf_simple_skin_skeleton()

    # Setup DirectDeltaMushDeformer
    var deformer = $DirectDeltaMushDeformer
    deformer.mesh = skin_mesh
    deformer.skeleton_path = skeleton.get_path()
    deformer.iterations = 10
    deformer.smooth_lambda = 0.5
    deformer.use_compute = true

    # Trigger precomputation
    deformer.precompute()

    # Add simple animation to test deformation
    animate_skeleton(skeleton)

func create_gltf_simple_skin_mesh() -> ArrayMesh:
    var mesh = ArrayMesh.new()

    # Following glTF Simple Skin specification:
    # 10 vertices forming a simple geometry (from the spec example)
    # POSITION attribute: 10 vertices with bounds [-0.5, 2.0, 0.0] to [0.5, 0.0, 0.0]
    var vertices = PackedVector3Array([
        Vector3(-0.5, 0.0, 0.0),   # 0
        Vector3(0.5, 0.0, 0.0),    # 1
        Vector3(-0.5, 2.0, 0.0),   # 2
        Vector3(0.5, 2.0, 0.0),    # 3
        Vector3(0.0, 1.0, -0.5),   # 4
        Vector3(0.0, 1.0, 0.5),    # 5
        Vector3(-0.25, 1.5, 0.0),  # 6
        Vector3(0.25, 1.5, 0.0),   # 7
        Vector3(0.0, 0.5, 0.0),    # 8
        Vector3(0.0, 1.5, 0.0)     # 9
    ])

    # Indices: 24 indices (12 triangles) as per glTF spec
    var indices = PackedInt32Array([
        0, 1, 3, 3, 2, 0,    # Base quad
        4, 5, 7, 7, 6, 4,    # Middle section
        8, 9, 6, 6, 4, 8,    # Left side
        1, 5, 9, 9, 8, 1     # Right side
    ])

    # JOINTS_0: 4-component joint indices per vertex (VEC4 as per glTF)
    # Following glTF spec: joints [1, 2] (0-based indices into skin.joints)
    var joints_0 = PackedInt32Array()
    for i in vertices.size():
        if i < 5:  # Lower vertices -> joint 0 (root)
            joints_0.append_array([0, -1, -1, -1])
        else:      # Upper vertices -> joint 1 (child)
            joints_0.append_array([1, -1, -1, -1])

    # WEIGHTS_0: 4-component weights per vertex (VEC4 as per glTF)
    # Full weight to assigned joint
    var weights_0 = PackedFloat32Array()
    for i in vertices.size():
        if i < 5:  # Lower vertices -> full weight to joint 0
            weights_0.append_array([1.0, 0.0, 0.0, 0.0])
        else:      # Upper vertices -> full weight to joint 1
            weights_0.append_array([0.0, 1.0, 0.0, 0.0])

    # Create surface arrays following glTF mesh primitive structure
    var arrays = []
    arrays.resize(Mesh.ARRAY_MAX)
    arrays[Mesh.ARRAY_VERTEX] = vertices
    arrays[Mesh.ARRAY_INDEX] = indices
    arrays[Mesh.ARRAY_BONES] = joints_0      # JOINTS_0 -> ARRAY_BONES
    arrays[Mesh.ARRAY_WEIGHTS] = weights_0   # WEIGHTS_0 -> ARRAY_WEIGHTS

    mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
    return mesh

func create_gltf_simple_skin_skeleton() -> Skeleton3D:
    var skeleton = Skeleton3D.new()

    # Following glTF Simple Skin specification:
    # Node 1: Root joint (no transformation)
    skeleton.add_bone("joint_0")
    skeleton.set_bone_rest(0, Transform3D())

    # Node 2: Child joint at [0.0, 1.0, 0.0] with identity rotation
    skeleton.add_bone("joint_1")
    skeleton.set_bone_parent(1, 0)
    skeleton.set_bone_rest(1, Transform3D(Basis(), Vector3(0.0, 1.0, 0.0)))

    add_child(skeleton)
    return skeleton

func animate_skeleton(skeleton: Skeleton3D):
    # Simple rotation animation following glTF spec
    var tween = create_tween()
    tween.set_loops()

    # Animate joint 1 (child) rotation like glTF example
    var start_rot = Quaternion(0, 0, 0, 1)  # Identity
    var end_rot = Quaternion(0, 0, 0.707, 0.707)  # 90-degree rotation

    tween.tween_method(
        func(t: float):
            var current_rot = start_rot.slerp(end_rot, t)
            skeleton.set_bone_pose_rotation(1, current_rot),
        0.0, 1.0, 2.0
    ).set_trans(Tween.TRANS_SINE)
