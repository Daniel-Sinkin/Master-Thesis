// lib/permutation/permutation.cpp
#include "permutation/permutation.hpp"

#include "tensor/tensor.hpp"

#include <utility>

namespace ds_tn
{

Permutation::Permutation(std::vector<usize> mapping) : mapping_(std::move(mapping))
{
    if (!is_valid_mapping(mapping_))
    {
        throw std::invalid_argument(
            "Permutation requires each index in [0, size) to appear exactly once."
        );
    }
}

Permutation::Permutation(std::initializer_list<usize> mapping)
    : Permutation(std::vector<usize>{mapping})
{
}

auto Permutation::at(usize index) const -> usize
{
    return mapping_.at(index);
}

auto Permutation::operator[](usize index) const noexcept -> usize
{
    return mapping_[index];
}

auto Permutation::size() const noexcept -> usize
{
    return mapping_.size();
}

auto Permutation::is_valid_mapping(std::span<const usize> mapping) noexcept -> bool
{
    auto seen = std::vector<bool>(mapping.size(), false);
    for (const auto destination : mapping)
    {
        if (destination >= mapping.size() || seen[destination])
        {
            return false;
        }
        seen[destination] = true;
    }
    return true;
}

auto apply_permutation(const NDArray& array, const Permutation& permutation) -> NDArray
{
    if (array.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("apply_permutation requires a valid NDArray.");
    }
    if (array.rank() != permutation.size())
    {
        throw std::invalid_argument(
            "apply_permutation requires permutation size to match NDArray rank."
        );
    }

    auto out = NDArray(permutation.apply(array.shape()));
    for (auto linear_index = 0zu; linear_index < array.size(); ++linear_index)
    {
        const auto src_indices = array.indices_from_linear(linear_index);
        const auto dst_indices = permutation.apply(src_indices);
        out(dst_indices) = array(src_indices);
    }

    return out;
}

auto apply_permutation(const Tensor& tensor, const Permutation& permutation) -> Tensor
{
    if (tensor.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument("apply_permutation requires a valid Tensor.");
    }
    if (tensor.rank() != permutation.size())
    {
        throw std::invalid_argument(
            "apply_permutation requires permutation size to match Tensor rank."
        );
    }

    const auto permuted_array = apply_permutation(tensor.array(), permutation);
    auto permuted_leg_names = permutation.apply(tensor.leg_names());
    return Tensor{std::move(permuted_array), std::move(permuted_leg_names)};
}

}  // namespace ds_tn
