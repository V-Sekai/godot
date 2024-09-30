import json
import math
import os
from typing import Tuple

import igl
import numpy as np
import scipy as sp
import robust_laplacian
import trimesh

def create_swept_volume(character_body_path, clothing_path, output_path, margin=0.5):
    """
    Creates a swept volume between a character body mesh and its clothing mesh with a specified margin.
    
    Args:
        character_body_path (str): Path to the character body mesh file (.obj).
        clothing_path (str): Path to the clothing mesh file (.obj).
        output_path (str): Path where the swept volume mesh will be saved (.obj).
        margin (float): Margin distance in millimeters to be added to the clothing mesh.
    """
    # Load meshes
    character_body = trimesh.load(character_body_path)
    clothing = trimesh.load(clothing_path)

    # Prepare data for closest point calculation
    test_points = np.array(character_body.vertices)
    mesh_vertices = np.array(clothing.vertices)
    mesh_faces = np.array(clothing.faces)

    # Find closest points on the clothing mesh for each vertex in the character body
    distances, face_indices, points, barycentric = find_closest_point_on_surface(
        test_points, mesh_vertices, mesh_faces
    )
    
    points_mesh = trimesh.Trimesh(vertices=points)

    # Create the convex hull mesh using trimesh
    convex_hull_mesh = points_mesh.convex_hull

    # Combine the extruded mesh with the original clothing mesh
    combined_mesh = character_body + convex_hull_mesh

    # Export the combined mesh to the specified output path
    combined_mesh.export(output_path)
    

def find_closest_point_on_surface(points, mesh_vertices, mesh_triangles):
    """
    Given a number of points, find their closest points on the surface of the mesh defined by mesh_vertices and mesh_triangles.

    Args:
        points (np.ndarray): An array of shape (#points, 3) where each row represents a point coordinate.
        mesh_vertices (np.ndarray): An array of shape (#mesh_vertices, 3) representing mesh vertices.
        mesh_triangles (np.ndarray): An array of shape (#mesh_triangles, 3) representing mesh triangle indices.

    Returns:
        tuple: A tuple containing:
            - smallest_squared_distances (np.ndarray): An array of shape (#points,) representing the smallest squared distances from each point to the mesh.
            - primitive_indices (np.ndarray): An array of shape (#points,) representing the indices of the triangles corresponding to the smallest distances.
            - closest_points (np.ndarray): An array of shape (#points, 3) representing the closest points on the mesh.
            - barycentric_coordinates (np.ndarray): An array of shape (#points, 3) representing the barycentric coordinates of the closest points.
    """
    points = np.asarray(points, dtype=np.float64)
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    mesh_triangles = np.asarray(mesh_triangles, dtype=np.int32)

    smallest_squared_distances, primitive_indices, closest_points = igl.point_mesh_squared_distance(
        points, mesh_vertices, mesh_triangles
    )
    closest_triangles = mesh_triangles[primitive_indices]
    vertex_1 = mesh_vertices[closest_triangles[:, 0]]
    vertex_2 = mesh_vertices[closest_triangles[:, 1]]
    vertex_3 = mesh_vertices[closest_triangles[:, 2]]
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

        # Take the absolute value of the dot product
        radian_angle = np.arccos(np.abs(np.dot(normalized_source_normal, normalized_target_normal)))

        degree_angle = math.degrees(radian_angle)
        if squared_distance[row_index] <= distance_threshold_squared and degree_angle <= angle_threshold_degrees:
            matched[row_index] = True

    return matched, target_weights


def is_valid_array(sparse_matrix):
    has_invalid_numbers = np.isnan(sparse_matrix.data).any() or np.isinf(sparse_matrix.data).any()
    return not has_invalid_numbers


def inpaint(V2, F2, W2, Matched):
    """
    Inpaint weights for all the vertices on the target mesh for which we didn't
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
    if not (V2.ndim == 2 and V2.shape[1] == 3):
        return None, False

    if not (F2.ndim == 2 and F2.shape[1] == 3 and np.max(F2) < len(V2)):
        return None, False

    if not (W2.ndim == 2 and W2.shape[0] == len(V2)):
        return None, False

    if not (Matched.ndim == 1 and len(Matched) == len(V2)):
        return None, False
    
    # Compute the laplacian
    L, M = robust_laplacian.mesh_laplacian(V2, F2)
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


def smooth(
    target_vertices,
    target_faces,
    skinning_weights,
    matched,
    distance_threshold,
    num_smooth_iter_steps=10,
    smooth_alpha=0.2,
):
    not_matched = ~matched
    vertices_ids_to_smooth = np.array(not_matched, copy=True)

    adjacency_list = igl.adjacency_list(target_faces)

    def get_points_within_distance(vertices, vertex_id, distance=distance_threshold):
        if vertex_id >= len(adjacency_list) or vertex_id < 0:
            return []

        queue = [vertex_id]
        visited = set(queue)

        while queue:
            current_vertex = queue.pop(0)
            if current_vertex >= len(adjacency_list):
                continue

            neighbors = adjacency_list[current_vertex]
            for neighbor in neighbors:
                if neighbor not in visited and np.linalg.norm(vertices[vertex_id] - vertices[neighbor]) < distance:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return list(visited)

    for i in range(target_vertices.shape[0]):
        if not_matched[i]:
            affected_vertices = get_points_within_distance(target_vertices, i)
            vertices_ids_to_smooth[affected_vertices] = True

    smoothed_weights = np.array(skinning_weights, copy=True)
    for step_idx in range(num_smooth_iter_steps):
        for i in range(target_vertices.shape[0]):
            if vertices_ids_to_smooth[i]:
                if i >= len(adjacency_list):
                    continue
                neighbors = adjacency_list[i]
                if not neighbors:
                    continue  # Skip this vertex if no neighbors

                num_neighbors = len(neighbors)
                weight = smoothed_weights[i, :]
                new_weight = (1 - smooth_alpha) * weight

                for neighbor in neighbors:
                    weight_connected = smoothed_weights[neighbor, :]
                    new_weight += (weight_connected / num_neighbors) * smooth_alpha

                smoothed_weights[i, :] = new_weight

    return smoothed_weights, vertices_ids_to_smooth

def load_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v, vt, vn, f, ft, fn = igl.read_obj(mesh_path)

    # Save original vertices for comparison
    vertices_original = v.copy()

    # Remove unreferenced vertices
    v, f, _, _ = igl.remove_unreferenced(v, f)

    if vertices_original.shape[0] != v.shape[0]:
        print("[Warning] Mesh has unreferenced vertices which were removed")

    # Calculate per vertex normals
    normals = igl.per_vertex_normals(v, f)

    return v, f, normals, vt


def main(source_mesh: str, target_mesh: str, output_file: str) -> None:
    # Get the directory of the current file
    current_folder: str = os.path.dirname(os.path.abspath(__file__))

    # Load the source mesh
    source_mesh_path: str = os.path.join(current_folder, source_mesh)
    vertices_1, faces_1, normals_1, uvs_1 = load_mesh(source_mesh_path)

    # Load the target mesh
    target_mesh_path: str = os.path.join(current_folder, target_mesh)
    vertices_2, faces_2, normals_2, uvs_2 = load_mesh(target_mesh_path)

    # You can setup your own skin weights matrix W \in R^(|V1| x num_bones) here
    # skin_weights = np.load("source_skinweights.npy")

    # For now, generate simple per-vertex data (can be skinning weights but can be any scalar data)
    skin_weights: np.ndarray = np.ones((vertices_1.shape[0], 2))  # our simple rig has only 2 bones
    skin_weights[:, 0] = 0.3  # first bone has an influence of 0.3 on all vertices
    skin_weights[:, 1] = 0.7  # second bone has an influence of 0.7 on all vertices

    # Section 3.1 Closest Point Matching
    distance_threshold: float = 0.05 * igl.bounding_box_diagonal(vertices_2)  # threshold distance D
    distance_threshold_squared: float = distance_threshold * distance_threshold
    angle_threshold_degrees: int = 30  # threshold angle theta in degrees

    # for every vertex on the target mesh find the closest point on the source mesh and copy weights over
    matched, interpolated_skin_weights = find_matches_closest_surface(
        vertices_1,
        faces_1,
        normals_1,
        vertices_2,
        faces_2,
        normals_2,
        skin_weights,
        distance_threshold_squared,
        angle_threshold_degrees,
    )

    # Section 3.2 Skinning Weights Inpainting
    inpainted_weights, success = inpaint(vertices_2, faces_2, interpolated_skin_weights, matched)

    if not success:
        print("[Error] Inpainting failed.")
        exit(0)

    # Optional smoothing
    smoothed_inpainted_weights, vertex_ids_to_smooth = smooth(
        vertices_2, faces_2, inpainted_weights, matched, distance_threshold, num_smooth_iter_steps=10, smooth_alpha=0.2
    )

    # Write the final mesh to the output file
    output_file_path: str = os.path.join(current_folder, output_file)

    skin_bones_target: np.ndarray = np.stack([np.eye(4) for _ in range(2)], axis=0)

    # Prepare the data to be written as JSON
    mesh_data = {
        "vertices": vertices_2.tolist(),
        "faces": faces_2.tolist(),
        "normals": normals_2.tolist(),
        "uvs": uvs_2.tolist(),
        "weights": smoothed_inpainted_weights.tolist(),
        "bones": skin_bones_target.tolist(),
    }

    # Write the final mesh to the output file
    output_file_path: str = os.path.join(current_folder, output_file)
    with open(output_file_path, "w") as f:
        json.dump(mesh_data, f)

import argparse
from argparse import ArgumentParser, Namespace

def parse_arguments():
    # TODO output mesh with weight transferred.
    parser = argparse.ArgumentParser(description="Generate a swept volume mesh between two given meshes.")
    parser.add_argument("--source_mesh", type=str, required=True, help="Path to the source mesh file")
    parser.add_argument("--target_mesh", type=str, required=True, help="Path to the target mesh file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    create_swept_volume(args.source_mesh, args.target_mesh, args.output_file)

# python src/main.py --source_mesh "meshes/sphere.obj" --target_mesh "meshes/grid.obj" --output_file "meshes/swept_cage.obj"
