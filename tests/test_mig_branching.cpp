// Tests for MIG-disjunction branching (Karamanov §2) and DA-dyn(k) cut
// selection (Karamanov §3). Correctness only — performance is studied
// separately via experiments/branching_study.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "core/problem.h"
#include "model/model.h"
#include "sep/cut.h"
#include "sep/cut_selector.h"

using Catch::Matchers::WithinAbs;

namespace {

cptp::SolverOptions quiet_with_branching(const std::string& mode) {
  return {
      {"time_limit", "30"},
      {"output_flag", "false"},
      {"branch_hyper", mode},
      {"branch_hyper_mig_k", "10"},
      {"branch_hyper_sb_max_depth", "3"},
  };
}

cptp::SolverOptions quiet_with_cut_fraction(double fraction) {
  return {
      {"time_limit", "30"},
      {"output_flag", "false"},
      {"cut_selector_fraction", std::to_string(fraction)},
  };
}

cptp::Problem make_5node_tour() {
  cptp::Problem prob;
  std::vector<cptp::Edge> edges = {
      {0, 1}, {0, 2}, {0, 3}, {0, 4},
      {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
  };
  std::vector<double> costs = {3, 4, 5, 6, 2, 3, 4, 2, 3, 2};
  std::vector<double> profits = {0, 8, 6, 5, 4};
  std::vector<double> demands = {0, 2, 2, 2, 2};
  prob.build(5, edges, costs, profits, demands, /*capacity=*/6, 0, 0);
  return prob;
}

}  // namespace

// ---------------------------------------------------------------------------
// MIG branching: equivalence with default branching on small instances.
// ---------------------------------------------------------------------------

TEST_CASE("MIG branching reaches the same optimum as default", "[mig]") {
  auto prob = make_5node_tour();
  cptp::Model model_off;
  model_off.set_problem(prob);
  auto r_off = model_off.solve({{"time_limit", "30"}, {"output_flag", "false"}});

  cptp::Model model_mig;
  model_mig.set_problem(prob);
  auto r_mig = model_mig.solve(quiet_with_branching("mig"));

  REQUIRE(r_off.has_solution());
  REQUIRE(r_mig.has_solution());
  REQUIRE_THAT(r_mig.objective, WithinAbs(r_off.objective, 1e-6));
}

TEST_CASE("'all' mode (pairs+clusters+demand+cardinality+mig) finds optimum",
          "[mig]") {
  auto prob = make_5node_tour();
  cptp::Model m_off;
  m_off.set_problem(prob);
  auto r_off = m_off.solve({{"time_limit", "30"}, {"output_flag", "false"}});

  cptp::Model m_all;
  m_all.set_problem(prob);
  auto r_all = m_all.solve(quiet_with_branching("all"));

  REQUIRE(r_off.has_solution());
  REQUIRE(r_all.has_solution());
  REQUIRE_THAT(r_all.objective, WithinAbs(r_off.objective, 1e-6));
}

TEST_CASE("Unknown branch_hyper mode falls back to off (warning)", "[mig]") {
  auto prob = make_5node_tour();
  cptp::Model m;
  m.set_problem(prob);
  cptp::SolverOptions opts = {
      {"time_limit", "30"},
      {"output_flag", "false"},
      {"branch_hyper", "this_is_not_a_real_mode"},
  };
  auto r = m.solve(opts);
  REQUIRE(r.has_solution());  // should not crash; falls back to "off"
}

// ---------------------------------------------------------------------------
// CutSelector: angle/depth selection correctness.
// ---------------------------------------------------------------------------

namespace {

cptp::sep::Cut make_cut(std::vector<int32_t> idx, std::vector<double> val,
                        double rhs, double violation) {
  cptp::sep::Cut c;
  c.indices = std::move(idx);
  c.values = std::move(val);
  c.rhs = rhs;
  c.violation = violation;
  return c;
}

}  // namespace

TEST_CASE("CutSelector::cosine_similarity on identical cuts is 1.0",
          "[cut_selector]") {
  auto a = make_cut({0, 1, 2}, {1.0, 2.0, 3.0}, 5.0, 0.5);
  auto b = make_cut({0, 1, 2}, {1.0, 2.0, 3.0}, 5.0, 0.5);
  REQUIRE_THAT(cptp::sep::CutSelector::cosine_similarity(a, b),
               WithinAbs(1.0, 1e-9));
}

TEST_CASE("CutSelector::cosine_similarity on orthogonal cuts is 0.0",
          "[cut_selector]") {
  auto a = make_cut({0, 1}, {1.0, 0.0}, 1.0, 0.5);
  auto b = make_cut({0, 1}, {0.0, 1.0}, 1.0, 0.5);
  REQUIRE_THAT(cptp::sep::CutSelector::cosine_similarity(a, b),
               WithinAbs(0.0, 1e-9));
}

TEST_CASE("CutSelector::cosine_similarity on disjoint supports is 0.0",
          "[cut_selector]") {
  auto a = make_cut({0, 1}, {1.0, 1.0}, 1.0, 0.5);
  auto b = make_cut({2, 3}, {1.0, 1.0}, 1.0, 0.5);
  REQUIRE_THAT(cptp::sep::CutSelector::cosine_similarity(a, b),
               WithinAbs(0.0, 1e-9));
}

TEST_CASE("CutSelector::select_da_dyn fraction=1.0 keeps everything",
          "[cut_selector]") {
  std::vector<cptp::sep::Cut> cuts;
  cuts.push_back(make_cut({0, 1}, {1.0, 1.0}, 1.0, 0.9));
  cuts.push_back(make_cut({2, 3}, {1.0, 1.0}, 1.0, 0.5));
  cuts.push_back(make_cut({0, 2}, {1.0, 1.0}, 1.0, 0.3));

  auto picked = cptp::sep::CutSelector::select_da_dyn(cuts, 1.0);
  REQUIRE(picked.size() == 3);
}

TEST_CASE(
    "CutSelector::select_da_dyn drops near-parallel cuts when fraction<1",
    "[cut_selector]") {
  // Cut 1 and Cut 2 are identical (cos = 1), so the selector should drop one.
  // Cut 3 is orthogonal to them.
  std::vector<cptp::sep::Cut> cuts;
  cuts.push_back(make_cut({0, 1}, {1.0, 1.0}, 1.0, 0.9));   // dominant
  cuts.push_back(make_cut({0, 1}, {1.0, 1.0}, 1.0, 0.85));  // duplicate
  cuts.push_back(make_cut({2, 3}, {1.0, 1.0}, 1.0, 0.5));   // orthogonal

  auto picked = cptp::sep::CutSelector::select_da_dyn(cuts, 0.67);
  // Should keep two: the dominant one + the orthogonal one (drop the duplicate).
  REQUIRE(picked.size() == 2);
  // The dominant cut (highest violation) must be present.
  bool has_dominant = false;
  for (const auto& c : picked) {
    if (c.indices == std::vector<int32_t>{0, 1} && c.violation > 0.86) {
      has_dominant = true;
    }
  }
  REQUIRE(has_dominant);
}

TEST_CASE("CutSelector::select_da_dyn empty input returns empty",
          "[cut_selector]") {
  auto picked = cptp::sep::CutSelector::select_da_dyn({}, 0.5);
  REQUIRE(picked.empty());
}

TEST_CASE("CutSelector::select_da_dyn fraction=0 returns empty",
          "[cut_selector]") {
  std::vector<cptp::sep::Cut> cuts;
  cuts.push_back(make_cut({0, 1}, {1.0, 1.0}, 1.0, 0.9));
  auto picked = cptp::sep::CutSelector::select_da_dyn(std::move(cuts), 0.0);
  REQUIRE(picked.empty());
}

// ---------------------------------------------------------------------------
// CutSelector integrated into solve(): solution correctness preserved.
// ---------------------------------------------------------------------------

TEST_CASE("Aggressive cut filtering still finds the optimum",
          "[cut_selector][model]") {
  auto prob = make_5node_tour();
  cptp::Model m_full;
  m_full.set_problem(prob);
  auto r_full = m_full.solve(quiet_with_cut_fraction(1.0));

  cptp::Model m_filtered;
  m_filtered.set_problem(prob);
  auto r_filtered = m_filtered.solve(quiet_with_cut_fraction(0.25));

  REQUIRE(r_full.has_solution());
  REQUIRE(r_filtered.has_solution());
  REQUIRE_THAT(r_filtered.objective, WithinAbs(r_full.objective, 1e-6));
}
