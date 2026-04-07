// lib/tensor_stats.hpp
#pragma once

#include "tensor.hpp"

namespace ds_tn {

struct TensorElementSummary {
    f64 min{};
    f64 max{};
    f64 sum{};
};

[[nodiscard]] auto l1_norm(const Tensor &tensor) -> f64;
[[nodiscard]] auto l2_norm(const Tensor &tensor) -> f64;
[[nodiscard]] auto lp_norm(const Tensor &tensor, f64 p) -> f64;
[[nodiscard]] auto infinity_norm(const Tensor &tensor) -> f64;
[[nodiscard]] auto element_summary(const Tensor &tensor) -> TensorElementSummary;

} // namespace ds_tn
