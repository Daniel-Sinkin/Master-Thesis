// lib/tensor/contraction.cpp
#include "tensor/contraction.hpp"

#include <algorithm>
#include <stdexcept>

namespace ds_tn
{
namespace
{

auto require_valid_tensor(const Tensor& tensor, const char* argument_name) -> void
{
    if (tensor.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument(
            std::string{"contraction helpers require "} + argument_name + " to be a valid Tensor."
        );
    }
}

[[nodiscard]] auto remove_shared(IndexNames names, std::span<const std::string> shared)
    -> std::vector<std::string>
{
    auto out = std::vector<std::string>{};
    out.reserve(names.size());

    for (const auto& name : names)
    {
        if (std::ranges::contains(shared, name))
        {
            continue;
        }
        out.push_back(name);
    }

    return out;
}

[[nodiscard]] auto leg_extent(const Tensor& tensor, std::string_view leg_name) -> usize
{
    for (auto axis = 0zu; axis < tensor.rank(); ++axis)
    {
        if (tensor.leg_name(axis) == leg_name)
        {
            return tensor.shape(axis);
        }
    }

    throw std::invalid_argument("Requested leg name is not present on the tensor.");
}

auto require_matching_shared_extents(
    const Tensor& left, const Tensor& right, std::span<const std::string> shared
) -> void
{
    for (const auto& leg_name : shared)
    {
        if (leg_extent(left, leg_name) != leg_extent(right, leg_name))
        {
            throw std::invalid_argument(
                "contraction_output_shape requires shared legs to have matching extents."
            );
        }
    }
}

}  // namespace

auto partition_indices(IndexNames left, IndexNames right) -> IndexPartition
{
    auto shared = std::vector<std::string>{};
    shared.reserve(std::min(left.size(), right.size()));

    for (const auto& name : left)
    {
        if (std::ranges::contains(right, name))
        {
            shared.push_back(name);
        }
    }

    std::ranges::sort(shared);

    return {
        .left = remove_shared(left, shared),
        .right = remove_shared(right, shared),
        .shared = std::move(shared),
    };
}

auto partition_indices(const Tensor& left, const Tensor& right) -> IndexPartition
{
    require_valid_tensor(left, "left");
    require_valid_tensor(right, "right");
    return partition_indices(left.leg_names(), right.leg_names());
}

auto contraction_output_legs(const Tensor& left, const Tensor& right) -> std::vector<std::string>
{
    const auto partition = partition_indices(left, right);
    auto legs = partition.left;
    legs.insert(legs.end(), partition.right.begin(), partition.right.end());
    return legs;
}

auto contraction_output_shape(const Tensor& left, const Tensor& right) -> std::vector<usize>
{
    const auto partition = partition_indices(left, right);
    require_matching_shared_extents(left, right, partition.shared);

    auto shape = std::vector<usize>{};
    shape.reserve(partition.left.size() + partition.right.size());

    for (const auto& leg_name : partition.left)
    {
        shape.push_back(leg_extent(left, leg_name));
    }
    for (const auto& leg_name : partition.right)
    {
        shape.push_back(leg_extent(right, leg_name));
    }

    return shape;
}

auto contraction_output_tensor(const Tensor& left, const Tensor& right) -> Tensor
{
    const auto legs = contraction_output_legs(left, right);
    auto shape = contraction_output_shape(left, right);
    return Tensor(std::move(shape), std::span<const std::string>{legs.begin(), legs.end()});
}

}  // namespace ds_tn
