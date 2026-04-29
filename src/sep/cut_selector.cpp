#include "sep/cut_selector.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace cptp::sep {

double CutSelector::cosine_similarity(const Cut& a, const Cut& b) {
  // Sparse dot product via index lookup on the smaller cut.
  const Cut* small = (a.indices.size() <= b.indices.size()) ? &a : &b;
  const Cut* large = (small == &a) ? &b : &a;

  std::unordered_map<int32_t, double> map_large;
  map_large.reserve(large->indices.size() * 2);
  for (size_t i = 0; i < large->indices.size(); ++i) {
    map_large.emplace(large->indices[i], large->values[i]);
  }

  double dot = 0.0;
  for (size_t i = 0; i < small->indices.size(); ++i) {
    auto it = map_large.find(small->indices[i]);
    if (it != map_large.end()) {
      dot += small->values[i] * it->second;
    }
  }

  double norm_a_sq = 0.0;
  for (double v : a.values) norm_a_sq += v * v;
  double norm_b_sq = 0.0;
  for (double v : b.values) norm_b_sq += v * v;

  if (norm_a_sq < 1e-20 || norm_b_sq < 1e-20) return 0.0;
  return dot / std::sqrt(norm_a_sq * norm_b_sq);
}

std::vector<int32_t> CutSelector::select_with_threshold(
    const std::vector<Cut>& cuts, double cosine_threshold) {
  std::vector<int32_t> accepted;
  accepted.reserve(cuts.size());
  for (size_t i = 0; i < cuts.size(); ++i) {
    bool ok = true;
    for (int32_t j : accepted) {
      if (std::abs(cosine_similarity(cuts[i], cuts[j])) > cosine_threshold) {
        ok = false;
        break;
      }
    }
    if (ok) accepted.push_back(static_cast<int32_t>(i));
  }
  return accepted;
}

std::vector<int32_t> CutSelector::select_da_dyn_indices(
    const std::vector<Cut>& cuts, double fraction_k) {
  if (cuts.empty()) return {};
  if (fraction_k >= 1.0 - 1e-12) {
    std::vector<int32_t> all(cuts.size());
    for (size_t i = 0; i < cuts.size(); ++i) all[i] = static_cast<int32_t>(i);
    return all;
  }
  if (fraction_k <= 0.0) return {};

  const int32_t target =
      std::max(1, static_cast<int32_t>(std::ceil(fraction_k * cuts.size())));

  // First try threshold = 1.0 (no filtering).
  std::vector<int32_t> best = select_with_threshold(cuts, 1.0);
  if (static_cast<int32_t>(best.size()) <= target) return best;

  // Binary search: find threshold producing ~target cuts.
  // Smaller threshold → stricter filter → fewer selected.
  double lo = 0.0, hi = 1.0;
  for (int iter = 0; iter < 30; ++iter) {
    double mid = 0.5 * (lo + hi);
    auto sel = select_with_threshold(cuts, mid);
    if (static_cast<int32_t>(sel.size()) > target) {
      hi = mid;
    } else {
      lo = mid;
      best = std::move(sel);
    }
    if (hi - lo < 1e-4) break;
  }
  return best;
}

std::vector<Cut> CutSelector::select_da_dyn(std::vector<Cut> cuts,
                                            double fraction_k) {
  if (cuts.empty()) return {};
  if (fraction_k >= 1.0 - 1e-12) return cuts;
  if (fraction_k <= 0.0) return {};

  std::sort(cuts.begin(), cuts.end(), [](const Cut& a, const Cut& b) {
    return a.violation > b.violation;
  });

  auto picked = select_da_dyn_indices(cuts, fraction_k);
  std::vector<Cut> result;
  result.reserve(picked.size());
  for (int32_t i : picked) result.push_back(std::move(cuts[i]));
  return result;
}

}  // namespace cptp::sep
