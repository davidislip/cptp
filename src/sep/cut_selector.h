#pragma once

#include <cstdint>
#include <vector>

#include "sep/cut.h"

namespace cptp::sep {

/// Angle + depth based cut selection.
///
/// Implements the DA-dyn(k) algorithm from Karamanov (2006) §3.2.3:
///   1. Sort candidate cuts by depth (violation) descending.
///   2. Greedily accept cuts whose angle with all already-accepted cuts
///      exceeds a threshold (cosine below it).
///   3. The cosine threshold is chosen by binary search so that the number
///      of selected cuts is approximately fraction_k * total.
///
/// Motivation (Karamanov §3.2.2 + §3.3): adding many near-parallel cuts
/// "flattens" the polyhedron near the LP optimum, hurting subsequent rounds.
/// Filtering by pairwise angle preserves cutting power while cutting the
/// pool size by ~10x at comparable gap-closing performance.
class CutSelector {
 public:
  /// Select a subset of cuts using the DA-dyn(fraction_k) algorithm.
  ///
  /// @param cuts        Candidate cuts (consumed and returned filtered).
  /// @param fraction_k  Fraction of cuts to keep, in (0, 1]. 1.0 = keep all.
  /// @return            Selected cuts, ordered by depth descending.
  static std::vector<Cut> select_da_dyn(std::vector<Cut> cuts,
                                        double fraction_k);

  /// Same as select_da_dyn but returns indices into the *input* vector
  /// (in selection order). Useful when the caller wants to keep auxiliary
  /// metadata aligned with the cuts. The input must already be sorted by
  /// violation descending.
  static std::vector<int32_t> select_da_dyn_indices(
      const std::vector<Cut>& cuts, double fraction_k);

  /// Cosine of the angle between two cuts in their full coefficient vectors.
  /// Returns 0.0 if either cut has zero norm.
  static double cosine_similarity(const Cut& a, const Cut& b);

 private:
  /// Try to select cuts with the given cosine threshold.
  /// A cut is accepted if its cosine with all previously-accepted cuts is
  /// at most threshold. Returns the indices of accepted cuts (in input order).
  /// Cuts must already be sorted by violation descending.
  static std::vector<int32_t> select_with_threshold(
      const std::vector<Cut>& cuts, double cosine_threshold);
};

}  // namespace cptp::sep
