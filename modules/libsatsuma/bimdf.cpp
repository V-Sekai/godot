#include "bimdf.h"
#include "libsatsuma/Extra/Highlevel.hh"
#include "core/variant/dictionary.h"

#include "src/libsatsuma/Problems/BiMCF.hh"

// int BIMDFSolver::add_node() {
//     return bimdf.add_node();
// }

// void BIMDFSolver::add_edge(const String &name, int u, int v, bool u_head, bool v_head, float target, float weight, int lower) {
//     using Abs = Satsuma::CostFunction::AbsDeviation;
//     lemon::ListGraphBase::Node node(v);
//     edges[name.utf8().get_data()] = bimdf.add_edge({
//         .u = u,
//         .v = v,
//         .u_head = u_head,
//         .v_head = v_head,
//         .cost_function = Abs{.target = target, .weight = weight},
//         .lower = static_cast<FlowScalar>(lower)
//     });
// }

Dictionary BIMDFSolver::solve() {
    auto config = Satsuma::BiMDFSolverConfig{
        .matching_solver = Satsuma::MatchingSolver::Lemon
    };
    auto result = Satsuma::solve_bimdf(bimdf, config);

    Dictionary solution;
    solution["total_cost"] = result.cost;

    for (const auto &[name, edge] : edges) {
        solution[name.utf8().get_data()] = (*result.solution)[edge];
    }

    return solution;
}

void BIMDFSolver::_bind_methods() {
}

BIMDFSolver::BIMDFSolver() {}
