import unittest
import numpy as np
from main import (find_closest_point_on_surface, interpolate_attribute_from_bary,
                  normalize_vector, find_matches_closest_surface, is_valid_array,
                  inpaint, smooth)


class TestMeshProcessing(unittest.TestCase):
    
    def test_find_closest_point_on_surface(self):
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1]
        ])
        triangles = np.array([
            [0, 1, 2]  # Only one triangle
        ])
        test_points = np.array([
            [0, 0, 0],  # Inside the projected area of the triangle
            [2, 2, 2],  # Outside near the plane of the triangle
            [0, 0, -2]  # Directly outside the triangle in the normal direction
        ])
        expected_distances = [1., 11., 1.]
        expected_indices = [0, 0, 0]
        expected_points = [[0., 0., -1.],
                        [1., 1., -1.],
                        [0., 0., -1.]]
        expected_barycentric = [[0.5, 0., 0.5],
                                [0., 0., 1.],
                                [0.5, 0., 0.5]]
        distances, indices, points, barycentric = find_closest_point_on_surface(
            test_points, vertices, triangles
        )
        np.testing.assert_array_almost_equal(distances, expected_distances)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_almost_equal(points, expected_points)
        np.testing.assert_array_almost_equal(barycentric, expected_barycentric)
        affine_matrix = np.array([
            [1, 0, 0, 1],  # Translation along x
            [0, 1, 0, 2],  # Translation along y
            [0, 0, 1, 3],  # Translation along z
            [0, 0, 0, 1]   # Homogeneous coordinate
        ])
        original_vertices = np.array([
            [-1, -1, -1, 1],
            [1, -1, -1, 1],
            [1, 1, -1, 1]
        ])
        transformed_vertices = original_vertices @ affine_matrix.T
        transformed_vertices = transformed_vertices[:, :3]  # Remove homogeneous coordinate
        test_points_transformed = np.array([
            [1, 2, 2],  # Inside the projected area of the triangle
            [3, 4, 5],  # Outside near the plane of the triangle
            [1, 2, 1]   # Directly outside the triangle in the normal direction
        ])
        distances_transformed, indices_transformed, points_transformed, barycentric_transformed = find_closest_point_on_surface(
            test_points_transformed, transformed_vertices, triangles
        )
        expected_distances = [ 0., 11.,  1.]
        expected_indices = [0, 0, 0]
        expected_points = [[1., 2., 2.],
            [2., 3., 2.],
            [1., 2., 2.]]
        np.testing.assert_array_almost_equal(distances_transformed, expected_distances)
        np.testing.assert_array_equal(indices_transformed, expected_indices)
        np.testing.assert_array_almost_equal(points_transformed, expected_points)
        np.testing.assert_array_almost_equal(barycentric_transformed, expected_barycentric)

    def test_interpolate_attribute_from_bary(self):
        vertex_attributes = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10]
        ])
        barycentric_coordinates = np.array([
            [0.2, 0.5, 0.3],
            [0.6, 0.3, 0.1]
        ])
        primitive_indices = np.array([0, 1])
        mesh_triangles = np.array([
            [0, 1, 2],
            [2, 3, 4]
        ])
        expected_output = np.array([
            [1*0.2 + 3*0.5 + 5*0.3, 2*0.2 + 4*0.5 + 6*0.3],
            [5*0.6 + 7*0.3 + 9*0.1, 6*0.6 + 8*0.3 + 10*0.1]
        ])
        result = interpolate_attribute_from_bary(vertex_attributes, barycentric_coordinates, primitive_indices, mesh_triangles)
        np.testing.assert_array_almost_equal(result, expected_output)

    def test_normalize_vector(self):
        vector = np.array([3, 4, 0])
        normalized = normalize_vector(vector)
        expected = np.array([0.6, 0.8, 0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_find_matches_closest_surface(self):
        # Mock data setup
        source_vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        source_triangles = np.array([
            [0, 1, 2]
        ])
        source_normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ])
        source_weights = np.array([
            [1, 0],
            [0, 1],
            [0.5, 0.5]
        ])

        target_vertices = np.array([
            [0.1, 0.1, 0],
            [2, 2, 2]  # This vertex should not match due to distance
        ])
        target_triangles = np.array([
            [0, 1]
        ])
        target_normals = np.array([
            [0, 0, 1],
            [1, 0, 0]  # This normal should not match due to angle
        ])

        distance_threshold_squared = 0.5
        angle_threshold_degrees = 10

        # Expected output
        expected_matched = np.array([True, False])
        expected_weights = np.array([[0.85, 0.15],
            [0.25, 0.75]]
        )

        # Running the function
        matched, target_weights = find_matches_closest_surface(
            source_vertices, source_triangles, source_normals,
            target_vertices, target_triangles, target_normals,
            source_weights, distance_threshold_squared, angle_threshold_degrees
        )

        # Asserting the results
        np.testing.assert_array_equal(matched, expected_matched)
        np.testing.assert_array_almost_equal(target_weights, expected_weights)

    def test_is_valid_array(self):
        valid_matrix = np.array([[1, 2], [3, 4]])
        invalid_matrix = np.array([[np.nan, 2], [np.inf, 4]])
        np.testing.assert_equal(is_valid_array(valid_matrix), True)
        np.testing.assert_equal(is_valid_array(invalid_matrix), False)

    def test_inpaint(self):
        V2 = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]  # This vertex needs inpainting
        ])
        F2 = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ])
        W2 = np.array([
            [1, 0],
            [0, 1],
            [0.5, 0.5],
            [0, 0]  # Initial weights for the vertex that needs inpainting
        ])
        Matched = np.array([True, True, True, False])
        expected_W_inpainted = np.array([[1.      , 0.      ],
            [0.      , 1.      ],
            [0.5     , 0.5     ],
            [0.117647, 0.882353] # Expected inpainted weights
        ])
        W_inpainted, success = inpaint(V2, F2, W2, Matched)
        np.testing.assert_equal(success, True)
        np.testing.assert_array_almost_equal(W_inpainted, expected_W_inpainted)
    
    def test_smooth(self):
        target_vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],  # This vertex needs smoothing
            [2, 1, 0]   # Additional vertex for distance check
        ])
        target_faces = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ])
        skinning_weights = np.array([
            [1, 0],
            [0, 1],
            [0.5, 0.5],
            [0.25, 0.75],  # Initial weights for the vertex that needs smoothing
            [0.1, 0.9]     # Additional vertex weight
        ])
        matched = np.array([True, True, True, False, False])
        distance_threshold = 1.5  # Distance threshold for smoothing

        smoothed_weights, vertices_ids_to_smooth = smooth(
            target_vertices,
            target_faces,
            skinning_weights,
            matched,
            distance_threshold,
            num_smooth_iter_steps=1,  # Single iteration for simplicity
            smooth_alpha=0.2
        )

        expected_smoothed_weights = np.array([
            [0.85,       0.15      ],
            [0.10666667, 0.89333333],
            [0.48044444, 0.51955556],
            [0.25871111, 0.74128889],
            [0.1,        0.9       ]
        ])
        expected_vertices_ids_to_smooth = np.array([True, True, True, True, True])

        np.testing.assert_array_almost_equal(smoothed_weights, expected_smoothed_weights)
        np.testing.assert_array_equal(vertices_ids_to_smooth, expected_vertices_ids_to_smooth)

if __name__ == '__main__':
    unittest.main()
