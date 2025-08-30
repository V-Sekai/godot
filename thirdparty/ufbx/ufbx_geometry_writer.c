#include "ufbx_geometry_writer.h"
#include <stdlib.h>

bool ufbx_ascii_write_geometry(ufbx_ascii_writer *writer, const ufbx_mesh *mesh) {
    if (!ufbx_ascii_write_node_begin(writer, "Geometry")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, mesh->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, mesh->element.name.data)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Mesh")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Vertices
    if (mesh->vertices.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "Vertices")) {
            return false;
        }
        
        // Convert vertices to double array
        double *vertex_data = (double*)malloc(mesh->vertices.count * 3 * sizeof(double));
        if (vertex_data) {
            for (size_t v = 0; v < mesh->vertices.count; v++) {
                vertex_data[v * 3 + 0] = mesh->vertices.data[v].x;
                vertex_data[v * 3 + 1] = mesh->vertices.data[v].y;
                vertex_data[v * 3 + 2] = mesh->vertices.data[v].z;
            }
            if (!ufbx_ascii_write_property_array_f64(writer, vertex_data, mesh->vertices.count * 3)) {
                free(vertex_data);
                return false;
            }
            free(vertex_data);
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }
    
    // PolygonVertexIndex
    if (mesh->vertex_indices.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "PolygonVertexIndex")) {
            return false;
        }
        
        // Convert indices to int32 array with FBX polygon encoding
        int32_t *index_data = (int32_t*)malloc(mesh->vertex_indices.count * sizeof(int32_t));
        if (index_data) {
            // Use actual face data to determine polygon boundaries
            if (mesh->num_faces > 0 && mesh->faces.data) {
                size_t current_index = 0;
                for (size_t f = 0; f < mesh->num_faces; f++) {
                    ufbx_face face = mesh->faces.data[f];
                    for (size_t fi = 0; fi < face.num_indices; fi++) {
                        if (face.index_begin + fi < mesh->vertex_indices.count) {
                            int32_t index = (int32_t)mesh->vertex_indices.data[face.index_begin + fi];
                            // Mark last index of each face as negative (FBX polygon end marker)
                            if (fi == face.num_indices - 1) {
                                index = -(index + 1);
                            }
                            index_data[current_index++] = index;
                        }
                    }
                }
            }
            if (!ufbx_ascii_write_property_array_i32(writer, index_data, mesh->vertex_indices.count)) {
                free(index_data);
                return false;
            }
            free(index_data);
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }

    // Edges (if available)
    if (mesh->edges.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "Edges")) {
            return false;
        }
        
        // Convert edges to int32 array - edges are vertex index pairs
        int32_t *edge_data = (int32_t*)malloc(mesh->edges.count * 2 * sizeof(int32_t));
        if (edge_data) {
            for (size_t e = 0; e < mesh->edges.count; e++) {
                edge_data[e * 2 + 0] = (int32_t)mesh->edges.data[e].a;
                edge_data[e * 2 + 1] = (int32_t)mesh->edges.data[e].b;
            }
            if (!ufbx_ascii_write_property_array_i32(writer, edge_data, mesh->edges.count * 2)) {
                free(edge_data);
                return false;
            }
            free(edge_data);
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }

    // GeometryVersion
    if (!ufbx_ascii_write_node_begin(writer, "GeometryVersion")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 124)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }

    // Write layer elements
    if (!ufbx_ascii_write_layer_element_normal(writer, mesh)) {
        return false;
    }
    
    if (!ufbx_ascii_write_layer_element_uv(writer, mesh)) {
        return false;
    }
    
    if (!ufbx_ascii_write_layer_element_material(writer, mesh)) {
        return false;
    }
    
    if (!ufbx_ascii_write_layer_structure(writer, mesh)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_layer_element_normal(ufbx_ascii_writer *writer, const ufbx_mesh *mesh) {
    // LayerElementNormal (CRITICAL FOR VALIDATION!)
    if (mesh->vertex_normal.exists && mesh->vertex_normal.values.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "LayerElementNormal")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "0 {\n")) {
            return false;
        }
        writer->indent_level++;

        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 102)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // Name
        if (!ufbx_ascii_write_node_begin(writer, "Name")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // MappingInformationType
        if (!ufbx_ascii_write_node_begin(writer, "MappingInformationType")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "ByPolygonVertex")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // ReferenceInformationType
        if (!ufbx_ascii_write_node_begin(writer, "ReferenceInformationType")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Direct")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // Normals
        if (!ufbx_ascii_write_node_begin(writer, "Normals")) {
            return false;
        }
        
        // Convert normals to double array (3 components per vertex)
        size_t normal_count = mesh->vertex_normal.values.count * 3;
        double *normal_data = (double*)malloc(normal_count * sizeof(double));
        if (normal_data) {
            for (size_t n = 0; n < mesh->vertex_normal.values.count; n++) {
                normal_data[n * 3 + 0] = mesh->vertex_normal.values.data[n].x;
                normal_data[n * 3 + 1] = mesh->vertex_normal.values.data[n].y;
                normal_data[n * 3 + 2] = mesh->vertex_normal.values.data[n].z;
            }
            if (!ufbx_ascii_write_property_array_f64(writer, normal_data, normal_count)) {
                free(normal_data);
                return false;
            }
            free(normal_data);
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // NormalsW (normal W components - usually all 1.0)
        if (!ufbx_ascii_write_node_begin(writer, "NormalsW")) {
            return false;
        }
        
        double *normalsw_data = (double*)malloc(mesh->vertex_normal.values.count * sizeof(double));
        if (normalsw_data) {
            for (size_t nw = 0; nw < mesh->vertex_normal.values.count; nw++) {
                normalsw_data[nw] = 1.0;
            }
            if (!ufbx_ascii_write_property_array_f64(writer, normalsw_data, mesh->vertex_normal.values.count)) {
                free(normalsw_data);
                return false;
            }
            free(normalsw_data);
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    return true;
}

bool ufbx_ascii_write_layer_element_uv(ufbx_ascii_writer *writer, const ufbx_mesh *mesh) {
    // LayerElementUV (CRITICAL FOR VALIDATION!)
    if (mesh->vertex_uv.exists && mesh->vertex_uv.values.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "LayerElementUV")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "0 {\n")) {
            return false;
        }
        writer->indent_level++;

        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 101)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // Name
        if (!ufbx_ascii_write_node_begin(writer, "Name")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "map1")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // MappingInformationType
        if (!ufbx_ascii_write_node_begin(writer, "MappingInformationType")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "ByPolygonVertex")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // ReferenceInformationType
        if (!ufbx_ascii_write_node_begin(writer, "ReferenceInformationType")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "IndexToDirect")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // UV coordinates
        if (!ufbx_ascii_write_node_begin(writer, "UV")) {
            return false;
        }
        
        // Convert UVs to double array (2 components per UV)
        size_t uv_count = mesh->vertex_uv.values.count * 2;
        double *uv_data = (double*)malloc(uv_count * sizeof(double));
        if (uv_data) {
            for (size_t u = 0; u < mesh->vertex_uv.values.count; u++) {
                uv_data[u * 2 + 0] = mesh->vertex_uv.values.data[u].x;
                uv_data[u * 2 + 1] = mesh->vertex_uv.values.data[u].y;
            }
            if (!ufbx_ascii_write_property_array_f64(writer, uv_data, uv_count)) {
                free(uv_data);
                return false;
            }
            free(uv_data);
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // UVIndex - CRITICAL for proper UV mapping!
        if (mesh->vertex_uv.indices.count > 0) {
            if (!ufbx_ascii_write_node_begin(writer, "UVIndex")) {
                return false;
            }
            
            int32_t *uv_index_data = (int32_t*)malloc(mesh->vertex_uv.indices.count * sizeof(int32_t));
            if (uv_index_data) {
                for (size_t ui = 0; ui < mesh->vertex_uv.indices.count; ui++) {
                    uv_index_data[ui] = (int32_t)mesh->vertex_uv.indices.data[ui];
                }
                if (!ufbx_ascii_write_property_array_i32(writer, uv_index_data, mesh->vertex_uv.indices.count)) {
                    free(uv_index_data);
                    return false;
                }
                free(uv_index_data);
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        } else if (mesh->vertex_uv.exists && mesh->vertex_uv.values.count > 0) {
            // Generate UV indices if missing but UVs exist
            if (!ufbx_ascii_write_node_begin(writer, "UVIndex")) {
                return false;
            }
            
            // Create sequential mapping for direct UV access
            int32_t *uv_index_data = (int32_t*)malloc(mesh->vertex_indices.count * sizeof(int32_t));
            if (uv_index_data) {
                for (size_t ui = 0; ui < mesh->vertex_indices.count; ui++) {
                    uv_index_data[ui] = (int32_t)mesh->vertex_indices.data[ui];
                }
                if (!ufbx_ascii_write_property_array_i32(writer, uv_index_data, mesh->vertex_indices.count)) {
                    free(uv_index_data);
                    return false;
                }
                free(uv_index_data);
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }

        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    return true;
}

bool ufbx_ascii_write_layer_element_material(ufbx_ascii_writer *writer, const ufbx_mesh *mesh) {
    // LayerElementMaterial (CRITICAL FOR VALIDATION!)
    if (mesh->materials.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "LayerElementMaterial")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "0 {\n")) {
            return false;
        }
        writer->indent_level++;

        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 101)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // Name
        if (!ufbx_ascii_write_node_begin(writer, "Name")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // MappingInformationType
        if (!ufbx_ascii_write_node_begin(writer, "MappingInformationType")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "AllSame")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // ReferenceInformationType
        if (!ufbx_ascii_write_node_begin(writer, "ReferenceInformationType")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "IndexToDirect")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        // Materials
        if (!ufbx_ascii_write_node_begin(writer, "Materials")) {
            return false;
        }
        
        int32_t material_indices[] = {0}; // First material
        if (!ufbx_ascii_write_property_array_i32(writer, material_indices, 1)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    return true;
}

bool ufbx_ascii_write_layer_structure(ufbx_ascii_writer *writer, const ufbx_mesh *mesh) {
    // Layer (CRITICAL STRUCTURE!)
    if (!ufbx_ascii_write_node_begin(writer, "Layer")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "0 {\n")) {
        return false;
    }
    writer->indent_level++;

    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 100)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }

    // LayerElement for Normals
    if (mesh->vertex_normal.exists && mesh->vertex_normal.values.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "LayerElement")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "{\n")) {
            return false;
        }
        writer->indent_level++;

        if (!ufbx_ascii_write_node_begin(writer, "Type")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "LayerElementNormal")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        if (!ufbx_ascii_write_node_begin(writer, "TypedIndex")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 0)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }

    // LayerElement for UVs
    if (mesh->vertex_uv.exists && mesh->vertex_uv.values.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "LayerElement")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "{\n")) {
            return false;
        }
        writer->indent_level++;

        if (!ufbx_ascii_write_node_begin(writer, "Type")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "LayerElementUV")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        if (!ufbx_ascii_write_node_begin(writer, "TypedIndex")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 0)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }

    // LayerElement for Materials
    if (mesh->materials.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "LayerElement")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "{\n")) {
            return false;
        }
        writer->indent_level++;

        if (!ufbx_ascii_write_node_begin(writer, "Type")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "LayerElementMaterial")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        if (!ufbx_ascii_write_node_begin(writer, "TypedIndex")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 0)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }

        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }

    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}
