//  SPDX-FileCopyrightText: 2023 Martin Heistermann <martin.heistermann@unibe.ch>
//  SPDX-License-Identifier: MIT
#include <libsatsuma/Extra/Highlevel.hh>

#include <libsatsuma/Solvers/EvenBiMDF.hh>
#include <libsatsuma/Solvers/BiMDFRefinement.hh>
#include <libsatsuma/Reductions/BiMDF_to_BiMCF.hh>
#include <libsatsuma/Reductions/BiMCF_to_MCF.hh>
#include <libsatsuma/Reductions/BiMCF_to_BMatching.hh>
#include <libsatsuma/Reductions/BMatching_to_Matching.hh>
#include <libsatsuma/Reductions/BiMDF_Simplification.hh>
#include <libsatsuma/Solvers/Matching.hh>
#include <libsatsuma/Solvers/MCF.hh>
#include <libsatsuma/Exceptions.hh>

#include "lemon/maps.h"

#include <string>

#if SATSUMA_HAVE_GUROBI
#  include <libsatsuma/Solvers/BiMCFGurobi.hh> // just for testing
#endif

namespace Satsuma {


BiMDFMatchingResult solve_bimdf_matching(const BiMDF &bimdf, const BiMDFSolverConfig &_config)
{
    Timekeeper::HierarchicalStopWatch sw_root{"bimdf via matching"};
    Timekeeper::HierarchicalStopWatch sw_refinement{"refinement", sw_root};
    sw_root.resume();

#if 0 // this is pretty slow!
    BiMDF::CostScalar lower_bound = -std::numeric_limits<BiMDF::CostScalar>::infinity();
    {
        Timekeeper::HierarchicalStopWatch sw_lower_bound{"lower_bound", sw_root};
        Timekeeper::ScopedStopWatch _{sw_lower_bound};
        lower_bound = bimdf_lower_bound(bimdf, 2).cost;
        if (verbosity > 2) {
            std::cout << "lower bound: " << lower_bound << std::endl;
        }
    }
#endif

    auto dc_sol = approximate_bimdf_doublecover(bimdf, _config.double_cover);
    if (_config.verbosity > 2)
    {
        std::cout << "DC approx: cost = " << bimdf.cost(*dc_sol.solution)
                  << ", max dev " << dc_sol.info.max_deviation_solution << std::endl;
    }
    if (!_config.refine_with_matching) {
        auto cost = dc_sol.info.cost; // need to copy before moving dc_sol.info out
        return {.result = {
                .solution = std::move(dc_sol.solution),
                .cost = cost},
            .double_cover_info = std::move(dc_sol.info),
            .info = {
                .cost = cost,
                .cost_changes = {},
                .max_refinement_change = 0,
            },
            .stopwatch = std::move(dc_sol.stopwatch)};
    }
    auto sol = std::make_unique<BiMDF::Solution>(bimdf.g);
    lemon::mapCopy(bimdf.g, *dc_sol.solution, *sol);

    std::vector<double> cost_changes;

    for (int maxdev = _config.refinement_maxdev_min; maxdev <= _config.refinement_maxdev_max; ++maxdev) {
        if (_config.verbosity >= 1) {
            std::cout << "refinement max deviation = " << maxdev << ": cost ch. " << std::flush;
        }
        while(true) {
            sw_refinement.resume();
            auto res = refine_with_matching(bimdf, *sol, maxdev,
                    _config.deviation_limit,
                    _config.matching_solver);
            sw_refinement.stop();

            if (_config.verbosity >= 1)
            {
                std::cout << res.cost_change << " " << std::flush;
            }

            cost_changes.push_back(res.cost_change);
            if (res.cost_change > -1e-20) // TODO: use integer costs
                break;
            sol = std::move(res.sol);
        }
        if (_config.verbosity >= 1) {
            std::cout << std::endl;
        }
    }
    sw_root.stop();

    int max_change = 0;
    for (const auto e: bimdf.g.edges()) {
        max_change = std::max(max_change, std::abs((*sol)[e] - (*dc_sol.solution)[e]));
    }
    if (_config.verbosity > 2)
    {
        std::cout << "maximal edge change in refinement: " << max_change << std::endl;
    }

    if (!bimdf.is_valid(*sol)) {
        throw InternalError("refinement result infeasible.");
    }
    auto cost = bimdf.cost(*sol);

    auto sw_result = Timekeeper::HierarchicalStopWatchResult{sw_root};
    sw_result.add_child(std::move(dc_sol.stopwatch));

    return {.result = {
             .solution = std::move(sol),
             .cost = cost},
            .double_cover_info = std::move(dc_sol.info),
            .info = {
                .cost = cost,
                .cost_changes = std::move(cost_changes),
                .max_refinement_change = max_change,
            },
            .stopwatch = sw_result};
}

BiMDFFullResult solve_bimdf(const BiMDF &_bimdf, const BiMDFSolverConfig &_config) {
    Timekeeper::HierarchicalStopWatch sw("solve_bimdf");
    Timekeeper::HierarchicalStopWatch sw_cc("cc", sw);
    Timekeeper::HierarchicalStopWatch sw_simp("simplification", sw);
    sw.resume();

    sw_cc.resume();
    Satsuma::BiMDF_ConnectedComponents cc(_bimdf);
    sw_cc.stop();
    size_t n_cc = cc.num_components();

    std::vector<BiMDFResult> sols;
    std::vector<Timekeeper::HierarchicalStopWatchResult> sw_results;
    std::vector<BiMDFperConnectedComponentInfo> cc_info;

    sols.reserve(n_cc);
    sw_results.reserve(n_cc);
    cc_info.reserve(n_cc);

    // Iterate over each connected component
	for (BiMDF *it = cc.bimdfs().first; it != cc.bimdfs().second; ++it) {
		const BiMDF &sub_bimdf = *it;
		sw_simp.resume();

        // Simplify and solve the matching problem for each sub-BiMDF
        Satsuma::BiMDF_Simplification simp(sub_bimdf);
        sw_simp.stop();
        auto simp_sol = Satsuma::solve_bimdf_matching(simp.bimdf(), _config);
        sw_results.push_back(std::move(simp_sol.stopwatch));
        cc_info.push_back({
            .n_nodes = sub_bimdf.n_nodes(),
            .n_edges = sub_bimdf.n_edges(),
            .double_cover = std::move(simp_sol.double_cover_info),
            .matching = std::move(simp_sol.info)
        });

        if (_config.verbosity >= 3) {
            std::cout << "\tSimplified sub-BiMDF, cost = " << simp.bimdf().cost(*simp_sol.result.solution) << std::endl;
        }

        sw_simp.resume();
		BiMDFResult bimdf_result = simp.translate_solution(simp_sol.result);
		sw_simp.stop();

        sols.push_back(std::move(bimdf_result));
	}

	sw_cc.resume();
	BiMDFResult bimdf_sol = cc.translate_solutions(sols);
	sw_cc.stop();

    if (_config.verbosity >= 1) {
        std::cout << "Solved BiMDF, total cost = " << _bimdf.cost(*bimdf_sol.solution) << std::endl;
    }

    sw.stop();
	Timekeeper::HierarchicalStopWatchResult sw_result = Timekeeper::HierarchicalStopWatchResult(sw);
	size_t sub_id = 0;
	for (Timekeeper::HierarchicalStopWatchResult &sub_sw : sw_results) {
		auto tmp = sub_sw.name + " " + std::to_string(sub_id++);
		sub_sw.name = tmp;
        sw_result.add_child(sub_sw);
	}

	return {
        .solution = std::move(bimdf_sol.solution),
        .cost = bimdf_sol.cost,
        .cc_info = cc_info,
        .stopwatch = sw_result
    };
}

} // namespace Satsuma
