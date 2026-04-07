// lib/tensor/contraction.cpp
#include "tensor/contraction.hpp"

#include <algorithm>
#include <stdexcept>

namespace ds_tn
{
namespace
{

struct SharedLeg
{
    std::string name{};
    usize left_index{0};
    usize right_index{0};
};

auto require_valid_tensor(const Tensor& tensor, const char* argument_name) -> void
{
    if (tensor.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument(
            std::string{"contraction helpers require "} + argument_name + " to be a valid Tensor."
        );
    }
}

[[nodiscard]] auto
is_shared(IndexNames names, IndexNames other_names, usize index, std::vector<SharedLeg>& shared_legs)
    -> bool
{
    const auto& name = names[index];
    for (auto other_index = 0zu; other_index < other_names.size(); ++other_index)
    {
        if (other_names[other_index] == name)
        {
            shared_legs.push_back({
                .name = std::string{name},
                .left_index = index,
                .right_index = other_index,
            });
            return true;
        }
    }
    return false;
}

auto require_matching_shared_extents(
    const Tensor& left,
    const Tensor& right,
    std::span<const usize> left_shared,
    std::span<const usize> right_shared
) -> void
{
    for (auto i = 0zu; i < left_shared.size(); ++i)
    {
        if (left.shape(left_shared[i]) != right.shape(right_shared[i]))
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
    auto shared_legs = std::vector<SharedLeg>{};
    shared_legs.reserve(std::min(left.size(), right.size()));
    auto left_not_shared = std::vector<usize>{};
    left_not_shared.reserve(left.size());
    auto right_not_shared = std::vector<usize>{};
    right_not_shared.reserve(right.size());

    for (auto left_index = 0zu; left_index < left.size(); ++left_index)
    {
        if (!is_shared(left, right, left_index, shared_legs))
        {
            left_not_shared.push_back(left_index);
        }
    }

    std::ranges::sort(shared_legs, std::less{}, &SharedLeg::name);

    auto left_shared = std::vector<usize>{};
    auto right_shared = std::vector<usize>{};
    left_shared.reserve(shared_legs.size());
    right_shared.reserve(shared_legs.size());
    for (const auto& shared_leg : shared_legs)
    {
        left_shared.push_back(shared_leg.left_index);
        right_shared.push_back(shared_leg.right_index);
    }

    for (auto right_index = 0zu; right_index < right.size(); ++right_index)
    {
        if (!std::ranges::contains(right_shared, right_index))
        {
            right_not_shared.push_back(right_index);
        }
    }

    return {
        .left_not_shared = std::move(left_not_shared),
        .left_shared = std::move(left_shared),
        .right_shared = std::move(right_shared),
        .right_not_shared = std::move(right_not_shared),
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
    auto legs = std::vector<std::string>{};
    legs.reserve(partition.left_not_shared.size() + partition.right_not_shared.size());

    for (const auto axis : partition.left_not_shared)
    {
        legs.push_back(left.leg_name(axis));
    }
    for (const auto axis : partition.right_not_shared)
    {
        legs.push_back(right.leg_name(axis));
    }

    return legs;
}

auto contraction_output_shape(const Tensor& left, const Tensor& right) -> std::vector<usize>
{
    const auto partition = partition_indices(left, right);
    require_matching_shared_extents(left, right, partition.left_shared, partition.right_shared);

    auto shape = std::vector<usize>{};
    shape.reserve(partition.left_not_shared.size() + partition.right_not_shared.size());

    for (const auto axis : partition.left_not_shared)
    {
        shape.push_back(left.shape(axis));
    }
    for (const auto axis : partition.right_not_shared)
    {
        shape.push_back(right.shape(axis));
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
