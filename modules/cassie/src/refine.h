#pragma once

#include "thirdparty/eigen/Eigen/Core"

void refine_patch(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, float targetEdgeLength, //
		Eigen::MatrixXd &V_fine, Eigen::MatrixXi &F_fine);

#include <fstream>
#include <iterator>
#include <vector>

#include "core/variant/variant.h"

#if 0
// using Epick = CGAL::Exact_predicates_inexact_constructions_kernel;
// using Mesh = CGAL::Surface_mesh<Epick::Point_3>;
// using halfedge_descriptor = boost::graph_traits<Mesh>::halfedge_descriptor;
// using edge_descriptor = boost::graph_traits<Mesh>::edge_descriptor;
// namespace PMP = CGAL::Polygon_mesh_processing;

struct halfedge2edge {
    halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges) : m_mesh(m), m_edges(edges) {}
    void operator()(const halfedge_descriptor& h) const { m_edges.push_back(edge(h, m_mesh)); }
    const Mesh& m_mesh;
    std::vector<edge_descriptor>& m_edges;
};

void refine_patch(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, float targetEdgeLength, //
    Eigen::MatrixXd& V_fine, Eigen::MatrixXi& F_fine) {

    // construct the mesh
    Mesh mesh;
    std::map<int, Mesh::Vertex_index> vmap;
    for (int i = 0; i < V.rows(); ++i)
        vmap[i] = mesh.add_vertex(Epick::Point_3(V(i, 0), V(i, 1), V(i, 2)));

    for (int i = 0; i < F.rows(); ++i)
        mesh.add_face(vmap[F(i, 0)], vmap[F(i, 1)], vmap[F(i, 2)]);

    print_line("#V = " + String::num(mesh.number_of_vertices()));
    print_line("#F = " + String::num(mesh.number_of_faces()));

    double target_edge_length = targetEdgeLength;
    unsigned int nb_iter = 3;

    print_line("Split border ... ");
    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
    PMP::split_long_edges(border, target_edge_length, mesh);
    print_line("done.");

    print_line("Start remeshing (" + String::num(num_faces(mesh)) + " faces)...");
    PMP::isotropic_remeshing(
        faces(mesh), target_edge_length, mesh,
        PMP::parameters::number_of_iterations(nb_iter).protect_constraints(true) // i.e. protect border, here
    );
    print_line("Remeshing done.");

    print_line("#V = " + String::num(mesh.number_of_vertices()));
    print_line("#F = " + String::num(mesh.number_of_faces()));

    // Smoothing

    const unsigned int nb_iterations = 10;
    const double time = 0.001;
    std::set<Mesh::Vertex_index> constrained_vertices;
    for (Mesh::Vertex_index v : vertices(mesh))
    {
        if (is_border(v, mesh))
            constrained_vertices.insert(v);
    }
    print_line("Constraining: " + String::num(constrained_vertices.size()) + " border vertices");
    CGAL::Boolean_property_map<std::set<Mesh::Vertex_index> > vcmap(constrained_vertices);
    print_line("Smoothing shape... (" + String::num(nb_iterations) + " iterations)");
    PMP::smooth_shape(mesh, time, PMP::parameters::number_of_iterations(nb_iterations)
        .vertex_is_constrained_map(vcmap));

    print_line("save new matrices ... ");
    V_fine.setZero(mesh.number_of_vertices(), 3);
    F_fine.setZero(mesh.number_of_faces(), 3);

    std::map<Mesh::Vertex_index, int> vmap2;
    int vi = 0;
    for (auto vh : mesh.vertices()) {
        vmap2[vh] = vi;
        V_fine.row(vi) <<       //
            mesh.point(vh).x(), //
            mesh.point(vh).y(), //
            mesh.point(vh).z();
        vi++;
    }
     print_line("vertices ok");

    int fi = 0;
    for (auto fh : mesh.faces()) {
        int k = 0;
        for (auto vh : vertices_around_face(mesh.halfedge(fh), mesh))
            F_fine(fi, k++) = vmap2[vh];
        fi++;
    }
    print_line("faces ok");
}
#endif