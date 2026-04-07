// lib/ndarray/stats.hpp
#pragma once

#include "ndarray/ndarray.hpp"

namespace ds_tn
{

struct NDArrayElementSummary
{
    f64 min{};
    f64 max{};
    f64 sum{};
};

[[nodiscard]] auto l1_norm(const NDArray& array) -> f64;
[[nodiscard]] auto l2_norm(const NDArray& array) -> f64;
[[nodiscard]] auto lp_norm(const NDArray& array, f64 p) -> f64;
[[nodiscard]] auto infinity_norm(const NDArray& array) -> f64;
[[nodiscard]] auto element_summary(const NDArray& array) -> NDArrayElementSummary;

}  // namespace ds_tn
