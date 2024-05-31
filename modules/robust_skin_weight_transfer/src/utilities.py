import math

import igl
import numpy as np
import scipy as sp
from robust_laplacian import mesh_laplacian


def find_closest_point_on_surface(points, mesh_vertices, mesh_triangles):
    """
    Given a number of points find their closest points on the surface of the mesh_vertices, mesh_triangles mesh

    Args:
        points: #points by 3, where every row is a point coordinate
        mesh_vertices: #mesh_vertices by 3 mesh vertices
        mesh_triangles: #mesh_triangles by 3 mesh triangles indices
    Returns:
        smallest_squared_distances: #points smallest squared distances
        primitive_indices: #points primitive indices corresponding to smallest distances
        closest_points: #points by 3 closest points
        barycentric_coordinates: #points by 3 of the barycentric coordinates of the closest point
    """

    smallest_squared_distances, primitive_indices, closest_points = igl.point_mesh_squared_distance(
        points, mesh_vertices, mesh_triangles
    )

    closest_triangles = mesh_triangles[primitive_indices, :]
    vertex_1 = mesh_vertices[closest_triangles[:, 0], :]
    vertex_2 = mesh_vertices[closest_triangles[:, 1], :]
    vertex_3 = mesh_vertices[closest_triangles[:, 2], :]

    barycentric_coordinates = igl.barycentric_coordinates_tri(closest_points, vertex_1, vertex_2, vertex_3)

    return smallest_squared_distances, primitive_indices, closest_points, barycentric_coordinates


def interpolate_attribute_from_bary(vertex_attributes, barycentric_coordinates, primitive_indices, mesh_triangles):
    """
    Interpolate per-vertex attributes vertex_attributes via barycentric coordinates barycentric_coordinates of the mesh_triangles[primitive_indices,:] vertices

    Args:
        vertex_attributes: #mesh_vertices by N per-vertex attributes
        barycentric_coordinates: #barycentric_coordinates by 3 array of the barycentric coordinates of some points
        primitive_indices: #barycentric_coordinates primitive indices containing the closest point
        mesh_triangles: #mesh_triangles by 3 mesh triangle indices
    Returns:
        interpolated_attributes: #barycentric_coordinates interpolated attributes
    """
    closest_triangles = mesh_triangles[primitive_indices, :]
    attribute_1 = vertex_attributes[closest_triangles[:, 0], :]
    attribute_2 = vertex_attributes[closest_triangles[:, 1], :]
    attribute_3 = vertex_attributes[closest_triangles[:, 2], :]

    barycentric_coordinate_1 = barycentric_coordinates[:, 0]
    barycentric_coordinate_2 = barycentric_coordinates[:, 1]
    barycentric_coordinate_3 = barycentric_coordinates[:, 2]

    barycentric_coordinate_1 = barycentric_coordinate_1.reshape(-1, 1)
    barycentric_coordinate_2 = barycentric_coordinate_2.reshape(-1, 1)
    barycentric_coordinate_3 = barycentric_coordinate_3.reshape(-1, 1)

    interpolated_attributes = (
        attribute_1 * barycentric_coordinate_1
        + attribute_2 * barycentric_coordinate_2
        + attribute_3 * barycentric_coordinate_3
    )

    return interpolated_attributes


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def find_matches_closest_surface(
    source_vertices,
    source_triangles,
    source_normals,
    target_vertices,
    target_triangles,
    target_normals,
    source_weights,
    distance_threshold_squared,
    angle_threshold_degrees,
):
    """
    For each vertex on the target mesh find a match on the source mesh.

    Args:
        source_vertices: #source_vertices by 3 source mesh vertices
        source_triangles: #source_triangles by 3 source mesh triangles indices
        source_normals: #source_vertices by 3 source mesh normals

        target_vertices: #target_vertices by 3 target mesh vertices
        target_triangles: #target_triangles by 3 target mesh triangles indices
        target_normals: #target_vertices by 3 target mesh normals

        source_weights: #source_vertices by num_bones source mesh skin weights

        distance_threshold_squared: scalar distance threshold
        angle_threshold_degrees: scalar normal threshold

    Returns:
        matched: #target_vertices array of bools, where matched[i] is True if we found a good match for vertex i on the source mesh
        target_weights: #target_vertices by num_bones, where target_weights[i,:] are skinning weights copied directly from source using closest point method
    """

    matched = np.zeros(shape=(target_vertices.shape[0]), dtype=bool)
    squared_distance, closest_indices, closest_points, barycentric_coordinates = find_closest_point_on_surface(
        target_vertices, source_vertices, source_triangles
    )

    # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals)
    # using the barycentric coordinates
    target_weights = interpolate_attribute_from_bary(
        source_weights, barycentric_coordinates, closest_indices, source_triangles
    )
    source_normals_matched_interpolated = interpolate_attribute_from_bary(
        source_normals, barycentric_coordinates, closest_indices, source_triangles
    )

    # check that the closest point passes our distance and normal thresholds
    for row_index in range(0, target_vertices.shape[0]):
        normalized_source_normal = normalize_vector(source_normals_matched_interpolated[row_index, :])
        normalized_target_normal = normalize_vector(target_normals[row_index, :])
        radian_angle = np.arccos(np.dot(normalized_source_normal, normalized_target_normal))
        degree_angle = math.degrees(radian_angle)
        if squared_distance[row_index] <= distance_threshold_squared and degree_angle <= angle_threshold_degrees:
            matched[row_index] = True

    return matched, target_weights


def is_valid_array(sparse_matrix):
    has_invalid_numbers = np.isnan(sparse_matrix.data).any() or np.isinf(sparse_matrix.data).any()
    return not has_invalid_numbers


def inpaint(V2, F2, W2, Matched):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didn't
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
        success: true if inpainting succeeded, false otherwise
    """

    # Compute the laplacian
    L, M = mesh_laplacian(V2, F2)
    L = -L  # Flip the sign of the Laplacian
    Minv = sp.sparse.diags(1 / M.diagonal())

    Q = -L + L * Minv * L

    Aeq = sp.sparse.csc_matrix((0, 0))
    Beq = np.array([])
    B = np.zeros(shape=(L.shape[0], W2.shape[1]))

    b = np.array(range(0, int(V2.shape[0])), dtype=int)
    b = b[Matched]
    bc = W2[Matched, :]

    results, W_inpainted = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, True)

    return W_inpainted, results


def smooth(V2, F2, W2, Matched, dDISTANCE_THRESHOLD, num_smooth_iter_steps=10, smooth_alpha=0.2):
    """
    Smooth weights in the areas for which weights were inpainted and also their close neighbors.

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones skinning weights
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        num_smooth_iter_steps: scalar number of smoothing steps
        smooth_alpha: scalar the smoothing strength

    Returns:
        W2_smoothed: #V2 by num_bones new smoothed weights
        VIDs_to_smooth: 1D array of vertex IDs for which smoothing was applied
    """

    NotMatched = ~Matched
    VIDs_to_smooth = np.array(NotMatched, copy=True)

    adj_list = igl.adjacency_list(F2)

    def get_points_within_distance(V, VID, distance=dDISTANCE_THRESHOLD):
        """
        Get all neighbors of vertex VID within dDISTANCE_THRESHOLD
        """

        queue = []
        queue.append(VID)
        while len(queue) != 0:
            vv = queue.pop()
            neigh = adj_list[vv]
            for nn in neigh:
                if ~VIDs_to_smooth[nn] and np.linalg.norm(V[VID, :] - V[nn]) < distance:
                    VIDs_to_smooth[nn] = True
                    if nn not in queue:
                        queue.append(nn)

    for i in range(0, V2.shape[0]):
        if NotMatched[i]:
            get_points_within_distance(V2, i)

    W2_smoothed = np.array(W2, copy=True)
    for step_idx in range(0, num_smooth_iter_steps):
        for i in range(0, V2.shape[0]):
            if VIDs_to_smooth[i]:
                neigh = adj_list[i]
                num = len(neigh)
                weight = W2_smoothed[i, :]

                new_weight = (1 - smooth_alpha) * weight
                for influence_idx in neigh:
                    weight_connected = W2_smoothed[influence_idx, :]
                    new_weight += (weight_connected / num) * smooth_alpha

                W2_smoothed[i, :] = new_weight

    return W2_smoothed, VIDs_to_smooth
