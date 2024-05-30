class igl:
    func point_mesh_squared_distance(P: Vector3, V: Array, F: Array) -> float:
        return 0.0

    func barycentric_coordinates_tri(C: Vector3, V1: Vector3, V2: Vector3, V3: Vector3) -> Vector3:
        return Vector3()

    func cotmatrix(V2: Array, F2: Array) -> Array:
        return Array()

    func massmatrix(V2: Array, F2: Array, MASSMATRIX_TYPE_VORONOI: int) -> Array:
        return Array()

    func min_quad_with_fixed(Q: Array, B: Array, b: Array, bc: Array, Aeq: Array, Beq: Array, is_true: bool) -> Array:
        return Array()

    func adjacency_list(F2: Array) -> Array:
        return Array()

class numpy:
    func linalg_norm(v: Vector3) -> float:
        return 0.0

    func zeros(shape: int, dtype: String) -> Array:
        return Array()

    func array_range(start: int, end: int, dtype: String) -> Array:
        return Array()

    func array_copy(NotMatched: Array, copy: bool) -> Array:
        return Array()

    func isnan(sparse_matrix_data: Array) -> bool:
        return false

    func isinf(sparse_matrix_data: Array) -> bool:
        return false

class Math:
    func degrees(rad_angle: float) -> float:
        return 0.0

class scipy:
    func sparse_diags(diagonal: Array) -> Array:
        return Array()

    func sparse_csc_matrix(size: Vector2) -> Array:
        return Array()
