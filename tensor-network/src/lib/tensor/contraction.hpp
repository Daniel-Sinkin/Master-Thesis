// lib/tensor/contraction.hpp
#pragma once

#include "tensor/tensor.hpp"

#include <span>
#include <string>
#include <vector>

namespace ds_tn
{

using IndexNames = std::span<const std::string>;

struct IndexPartition
{
    std::vector<std::string> left{};    // preserves original ordering
    std::vector<std::string> right{};   // preserves original ordering
    std::vector<std::string> shared{};  // sorted lexicographically
};

[[nodiscard]] auto partition_indices(IndexNames left, IndexNames right) -> IndexPartition;
[[nodiscard]] auto partition_indices(const Tensor& left, const Tensor& right) -> IndexPartition;
[[nodiscard]] auto contraction_output_legs(const Tensor& left, const Tensor& right)
    -> std::vector<std::string>;
[[nodiscard]] auto contraction_output_shape(const Tensor& left, const Tensor& right)
    -> std::vector<usize>;
[[nodiscard]] auto contraction_output_tensor(const Tensor& left, const Tensor& right) -> Tensor;

}  // namespace ds_tn
