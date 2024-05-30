extends Node

var igl = igl.new()
var numpy = numpy.new()
var Math = Math.new()
var scipy = scipy.new()

## This function performs a robust skin weight transfer between two meshes.
##
## Parameters:
## v1 (Array): An array of 3D coordinates representing the vertices of the source mesh. Each element in the array is a tuple or list of three numbers, representing the x, y, and z coordinates of a vertex.
## f1 (Array): An array of face indices for the source mesh. Each element in the array is a tuple or list of three integers, representing the indices of the vertices that make up a face.
## w (Array): An array of weights for the source mesh. The weights represent how much influence each bone has on each vertex. Each element in the array is a list of numbers, where each number represents the weight of a bone for a particular vertex.
## v2 (Array): Similar to `v1`, this is an array of 3D coordinates representing the vertices of the target mesh.
## f2 (Array): Similar to `f1`, this is an array of face indices for the target mesh.
##
## The function first calculates normals for both meshes. It then registers the source and target meshes along with their normals.
## A closest point matching is performed where for every vertex on the target mesh, the closest point on the source mesh is found and weights are copied over.
## An inpainting process is then carried out to fill in missing data in the skin weights.
## Finally, an optional smoothing operation can be applied to the weights.
##
## Returns:
## None
func robust_skin_weight_transfer(v1, f1, w, v2, f2):
    var n1 = calculate_normals(v1, f1)
    var n2 = calculate_normals(v2, f2)
    var num_bones = w.size() 

    # Register source and target Mesh geometries, plus their Normals
    register_surface_mesh("SourceMesh", v1, f1)
    register_surface_mesh("TargetMesh", v2, f2)
    add_vector_quantity("SourceMesh", "Normals", n1)
    add_vector_quantity("TargetMesh", "Normals", n2)

    # Section 3.1 Closest Point Matching
    var distance_threshold = 0.05 * bounding_box_diagonal(v2) # threshold distance D
    var distance_threshold_squared = distance_threshold * distance_threshold
    var angle_threshold_degrees = 30 # threshold angle theta in degrees

    # for every vertex on the target mesh find the closest point on the source mesh and copy weights over
    var matched, skin_weights_interpolated = Utils.find_matches_closest_surface(v1,f1,n1,v2,f2,n2,w,distance_threshold_squared,angle_threshold_degrees)

    # visualize vertices for which we found a match
    add_scalar_quantity("TargetMesh", "Matched", matched)

    # Section 3.2 Skinning Weights Inpainting
    var inpainted_weights, success = Utils.inpaint(v2, f2, skin_weights_interpolated, matched)

    if (not success):
        print("[Error] Inpainting failed.")
        return

    # Visualize the weights for each bone
    add_scalar_quantity("TargetMesh", "Bone1", inpainted_weights[0])
    add_scalar_quantity("TargetMesh", "Bone2", inpainted_weights[1])

    # Optional smoothing
    var smoothed_inpainted_weights, vids_to_smooth = Utils.smooth(v2, f2, inpainted_weights, matched, distance_threshold, num_smooth_iter_steps=10, smooth_alpha=0.2)
    add_scalar_quantity("TargetMesh", "VIDs_to_smooth", vids_to_smooth)

    # Visualize the smoothed weights for each bone
    add_scalar_quantity("TargetMesh", "SmoothedBone1", smoothed_inpainted_weights[0])
    add_scalar_quantity("TargetMesh", "SmoothedBone2", smoothed_inpainted_weights[1])

## Given a number of points find their closest points on the surface of the V,F mesh
##
## Args:
##     P: #P by 3, where every row is a point coordinate
##     V: #V by 3 mesh vertices
##     F: #F by 3 mesh triangles indices
## Returns:
##     sqrD #P smallest squared distances
##     I #P primitive indices corresponding to smallest distances
##     C #P by 3 closest points
##     B #P by 3 of the barycentric coordinates of the closest point
func find_closest_point_on_surface(P: Vector3, V: Array, F: Array) -> Array:
    var result = igl.point_mesh_squared_distance(P, V, F)
    var sqrD = result[0]
    var I = result[1]
    var C = result[2]

    var F_closest = F[I]
    var V1 = V[F_closest[0]]
    var V2 = V[F_closest[1]]
    var V3 = V[F_closest[2]]

    var B = igl.barycentric_coordinates_tri(C, V1, V2, V3)

    return [sqrD, I, C, B]

##
## Interpolate per-vertex attributes A via barycentric coordinates B of the F[I] vertices
##
## Args:
##     A: #V by N per-vertex attributes
##     B  #B by 3 array of the barycentric coordinates of some points
##     I  #B primitive indices containing the closest point
##     F: #F by 3 mesh triangle indices
## Returns:
##     A_out #B interpolated attributes
##
func interpolate_attribute_from_bary(A, B, I, F):
    var F_closest = F[I]
    var a1 = A[F_closest[0]]
    var a2 = A[F_closest[1]]
    var a3 = A[F_closest[2]]

    var b1 = []
    var b2 = []
    var b3 = []

    for i in range(B.size()):
        b1.append([B[i][0]])
        b2.append([B[i][1]])
        b3.append([B[i][2]])

    var A_out = []
    for i in range(a1.size()):
        A_out.append(a1[i]*b1[i][0] + a2[i]*b2[i][0] + a3[i]*b3[i][0])

    return A_out

func normalize_vec(v):
    return v.normalized()


## For each vertex on the target mesh find a match on the source mesh.
##
## Args:
##     V1: #V1 by 3 source mesh vertices
##     F1: #F1 by 3 source mesh triangles indices
##     N1: #V1 by 3 source mesh normals
##
##     V2: #V2 by 3 target mesh vertices
##     F2: #F2 by 3 target mesh triangles indices
##     N2: #V2 by 3 target mesh normals
##
##     W1: #V1 by num_bones source mesh skin weights
##
##     dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
##     dANGLE_THRESHOLD_DEGREES: scalar normal threshold
##
## Returns:
##     Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
##     W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
func find_matches_closest_surface(V1, F1, N1, V2, F2, N2, W1, dDISTANCE_THRESHOLD_SQRD, dANGLE_THRESHOLD_DEGREES):
    var Matched = []
    for i in range(V2.size()):
        Matched.append(false)

    var sqrD, I, C, B = find_closest_point_on_surface(V2,V1,F1)

    # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals)
    # using the barycentric coordinates
    var W2 = interpolate_attribute_from_bary(W1,B,I,F1)
    var N1_match_interpolated = interpolate_attribute_from_bary(N1,B,I,F1)

    # check that the closest point passes our distance and normal thresholds
    for RowIdx in range(V2.size()):
        var n1 = normalize_vec(N1_match_interpolated[RowIdx])
        var n2 = normalize_vec(N2[RowIdx])
        var rad_angle = acos(n1.dot(n2))
        var deg_angle = rad2deg(rad_angle)
        if sqrD[RowIdx] <= dDISTANCE_THRESHOLD_SQRD and deg_angle <= dANGLE_THRESHOLD_DEGREES:
            Matched[RowIdx] = true

    return Matched, W2

func is_valid_array(sparse_matrix):
    var has_invalid_numbers = false
    for i in range(sparse_matrix.size()):
        if isnan(sparse_matrix[i]) or isinf(sparse_matrix[i]):
            has_invalid_numbers = true
            break
    return not has_invalid_numbers


func is_valid_array(sparse_matrix):
    var has_invalid_numbers = false
    for i in range(sparse_matrix.size()):
        if isnan(sparse_matrix[i]) or isinf(sparse_matrix[i]):
            has_invalid_numbers = true
            break
    return not has_invalid_numbers

## Inpaint weights for all the vertices on the target mesh for which we didnt
## find a good match on the source (i.e. Matched[i] == False).
##
## Args:
##     V2: #V2 by 3 target mesh vertices
##     F2: #F2 by 3 target mesh triangles indices
##     W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
##     Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
##
## Returns:
##     W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
##     success: true if inpainting succeeded, false otherwise
func inpaint(V2, F2, W2, Matched):
    print("Creating and preprocessing the input mesh...")
    var inputMesh = Mesh.new()
    inputMesh.add_vertices(V2)
    for i in range(0, F2.size(), 3):
        inputMesh.add_triangle(F2[i], F2[i+1], F2[i+2])
    print("Preprocessing the input mesh...")
    inputMesh.stripFacesWithDuplicateVertices()
    var oldToNewMap = inputMesh.stripUnusedVertices()
    inputMesh.triangulate()

    # Create a halfedge and geometry from the preprocessed mesh
    var mesh, geometry = makeGeneralHalfedgeAndGeometry(inputMesh.polygons, inputMesh.vertexCoordinates)

    print("Building tufted Laplacian...")
    var L, M = buildTuftedLaplacian(mesh, geometry, mollifyFactor)
    print("  ...done!")

    var Minv = sparse_diags(1.0 / M.diagonal())

    var is_valid = is_valid_array(L)
    if (not is_valid):
        print("[Error] Laplacian is invalid:")

    is_valid = is_valid_array(Minv)
    if (not is_valid):
        print("[Error] Mass matrix is invalid:")

    var Q = -L + L * Minv * L

    A = is_valid_array(Q)
    if (not is_valid):
        print("[Error] System matrix is invalid:")
    
    var Aeq = sparse_csc_matrix(0, 0)
    var Beq = []
    var B = zeros(L.shape[0], W2.shape[1])

    var b = range(0, V2.shape[0])
    b = b[Matched]
    var bc = W2[Matched,:]

    var results, W_inpainted = min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, true)

    return [W_inpainted, results]

func get_points_within_distance(V, VID, adj_list, VIDs_to_smooth, distance):
    var queue = []
    queue.append(VID)
    while queue.size() != 0:
        var vv = queue.pop_back()
        var neigh = adj_list[vv]
        for nn in neigh:
            if not VIDs_to_smooth[nn] and (V[VID]-V[nn]).length() < distance:
                VIDs_to_smooth[nn] = true
                if not queue.has(nn):
                    queue.append(nn)

## Smooth weights in the areas for which weights were inpainted and also their close neighbours.
##
## Args:
##     V2: #V2 by 3 target mesh vertices
##     F2: #F2 by 3 target mesh triangles indices
##     W2: #V2 by num_bones skinning weights
##     Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
##     dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
##     num_smooth_iter_steps: scalar number of smoothing steps
##     smooth_alpha: scalar the smoothing strength
##
## Returns:
##     W2_smoothed: #V2 by num_bones new smoothed weights
##     VIDs_to_smooth: 1D array of vertex IDs for which smoothing was applied
func smooth(V2, F2, W2, Matched, dDISTANCE_THRESHOLD, num_smooth_iter_steps=10, smooth_alpha=0.2):
    var NotMatched = !Matched
    var VIDs_to_smooth = NotMatched.duplicate()

    var adj_list = adjacency_list(F2)

    for i in range(0, V2.size()):
        if NotMatched[i]:
            get_points_within_distance(V2, i, adj_list, VIDs_to_smooth, dDISTANCE_THRESHOLD)

    var W2_smoothed = W2.duplicate()
    for step_idx in range(0, num_smooth_iter_steps):
        for i in range(0, V2.size()):
            if VIDs_to_smooth[i]:
                var neigh = adj_list[i]
                var num = neigh.size()
                var weight = W2_smoothed[i]

                var new_weight = (1-smooth_alpha)*weight
                for influence_idx in neigh:
                    var weight_connected = W2_smoothed[influence_idx]
                    new_weight += (weight_connected / num) * smooth_alpha

                W2_smoothed[i] = new_weight

    return [W2_smoothed, VIDs_to_smooth]
