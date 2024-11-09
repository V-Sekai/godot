//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#include <libsatsuma/Solvers/Matching.hh>
#include <lemon/matching.h>

namespace Satsuma {

MatchingResult solve_matching_via_lemon(Matching const &mp)
{
    //lemon::TimeReport tr("MWPM-lemon: ");
    auto mwpm = lemon::MaxWeightedPerfectMatching(mp.g, mp.weight);
    auto success = mwpm.run();
    if (!success) {
        std::cerr << "Matching problem is infeasible" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    //std::cout << "mpwm cost: " << -mwpm.matchingWeight() << std::endl;
    auto sol = std::make_unique<Matching::Solution>(mp.g);
    for (auto e: mp.g.edges()) {
        (*sol)[e] = mwpm.matching(e);
    }
    return {.solution = std::move(sol), .weight = mwpm.matchingWeight()};
}

MatchingResult solve_matching_via_blossomV(Matching const &mp)
{
    std::cerr << "Satsuma was built without blossom-V support." << std::endl;
    std::exit(EXIT_FAILURE);
}

MatchingResult solve_matching(const Matching &mp, MatchingSolver solver)
{
    if (solver == MatchingSolver::BlossomV) {
        return solve_matching_via_blossomV(mp);
    } else if (solver == MatchingSolver::Lemon) {
        return solve_matching_via_lemon(mp);
    }
}


} // namespace Satsuma
