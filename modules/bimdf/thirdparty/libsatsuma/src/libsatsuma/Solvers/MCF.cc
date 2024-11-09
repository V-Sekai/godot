//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#include <libsatsuma/Solvers/MCF.hh>
#include <libsatsuma/Exceptions.hh>
#include <lemon/network_simplex.h>

namespace Satsuma {
MCFResult solve_mcf_via_lemon_netsimp(const MCF &mcf)
{
    lemon::NetworkSimplex<MCF::GraphT, MCF::FlowScalar, MCF::CostScalar> solver{mcf.g};
    using LemonSolver = decltype(solver);
    solver.costMap(mcf.cost);
    solver.supplyMap(mcf.supply);
    solver.upperMap(mcf.upper);
    solver.lowerMap(mcf.lower);
    auto res = solver.run();
    MCFResult result;
    result.solution = nullptr;
    result.cost = 0;

    if (res == LemonSolver::OPTIMAL) {
        auto sol = std::make_unique<MCF::Solution>(mcf.g);
        for (const auto a: mcf.g.arcs()) {
            (*sol)[a] = solver.flow(a);
        }
        result.solution = std::move(sol);
        result.cost = solver.totalCost();
    }

    return result;
}

} // namespace Satsuma
