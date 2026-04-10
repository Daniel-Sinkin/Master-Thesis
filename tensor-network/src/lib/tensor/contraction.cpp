// lib/tensor/contraction.cpp
#include "tensor/contraction.hpp"

#include "ndarray/blas.hpp"
#include "permutation/permutation.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto concat_indices(std::span<const usize> lhs, std::span<const usize> rhs)
    -> std::vector<usize>
{
    auto out = std::vector<usize>{};
    out.reserve(lhs.size() + rhs.size());
    out.insert(out.end(), lhs.begin(), lhs.end());
    out.insert(out.end(), rhs.begin(), rhs.end());
    return out;
}

[[nodiscard]] auto permutation_from_axis_order(std::span<const usize> axis_order) -> Permutation
{
    auto mapping = std::vector<usize>(axis_order.size());
    for (auto destination = 0zu; destination < axis_order.size(); ++destination)
    {
        mapping[axis_order[destination]] = destination;
    }
    return Permutation{std::move(mapping)};
}

[[nodiscard]] auto
product_over_selected_axes(std::span<const usize> shape, std::span<const usize> indices) -> usize
{
    auto out = usize{1};
    for (const auto axis : indices)
    {
        out *= shape[axis];
    }
    return out;
}

auto require_valid_tensor(const Tensor& tensor, const char* argument_name) -> void
{
    if (tensor.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument(
            std::string{"contraction helpers require "} + argument_name + " to be a valid Tensor."
        );
    }
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

    const auto not_shared_indices = [&shared](IndexNames names) -> std::vector<usize>
    {
        const auto num_remaining = names.size() - shared.size();
        if (num_remaining == 0)
        {
            return {};
        }

        auto out = std::vector<usize>{};
        out.reserve(num_remaining);
        for (auto i = 0zu; i < names.size(); ++i)
        {
            if (std::ranges::contains(shared, names[i]))
            {
                continue;
            }
            out.push_back(i);
        }
        return out;
    };

    const auto shared_indices = [&shared](IndexNames names) -> std::vector<usize>
    {
        auto out = std::vector<usize>{};
        out.reserve(shared.size());
        for (const auto& name : shared)
        {
            const auto it = std::ranges::find(names, name);
            out.push_back(static_cast<usize>(std::distance(names.begin(), it)));
        }
        return out;
    };

    return {
        .left_not_shared = not_shared_indices(left),
        .left_shared = shared_indices(left),
        .right_shared = shared_indices(right),
        .right_not_shared = not_shared_indices(right),
    };
}

auto partition_indices(const Tensor& left, const Tensor& right) -> IndexPartition
{
    {  // Expects
        require_valid_tensor(left, "left");
        require_valid_tensor(right, "right");
    }
    return partition_indices(left.leg_names(), right.leg_names());
}

auto contraction_output_legs(const Tensor& left, const Tensor& right) -> std::vector<std::string>
{
    {  // Expects
        require_valid_tensor(left, "left");
        require_valid_tensor(right, "right");
    }

    const auto partition = partition_indices(left.leg_names(), right.leg_names());
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
    {  // Expects
        require_valid_tensor(left, "left");
        require_valid_tensor(right, "right");
    }

    const auto partition = partition_indices(left.leg_names(), right.leg_names());
    {  // Expects
        require_matching_shared_extents(left, right, partition.left_shared, partition.right_shared);
    }

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

auto contract(const Tensor& left, const Tensor& right) -> Tensor
{
    {  // Expects
        require_valid_tensor(left, "left");
        require_valid_tensor(right, "right");
    }

    const auto [left_not_shared, left_shared, right_shared, right_not_shared] =
        partition_indices(left.leg_names(), right.leg_names());
    {  // Expects
        require_matching_shared_extents(left, right, left_shared, right_shared);
    }
    const auto make_matrix = [](const Tensor& tensor,
                                std::span<const usize> leading_axes,
                                std::span<const usize> trailing_axes) -> NDArray
    {
        const auto transposed = apply_permutation(
            tensor,
            permutation_from_axis_order(concat_indices(leading_axes, trailing_axes))
        );
        const auto matrix = NDArray::reshape(
            transposed.array(),
            std::array{
                product_over_selected_axes(tensor.shape(), leading_axes),
                product_over_selected_axes(tensor.shape(), trailing_axes),
            }
        );
        assert(matrix.shape()[0] * matrix.shape()[1] == tensor.size());
        return matrix;
    };

    const auto left_matrix = make_matrix(left, left_not_shared, left_shared);
    const auto right_matrix = make_matrix(right, right_shared, right_not_shared);

    const auto flattened = matrix_matrix_product(left_matrix, right_matrix);
    return {
        NDArray::reshape(flattened, contraction_output_shape(left, right)),
        contraction_output_legs(left, right)
    };
}

}  // namespace ds_tn
