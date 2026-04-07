// lib/tensor/compare.cpp
#include "tensor/compare.hpp"

#include "ndarray/compare.hpp"

#include <cmath>
#include <ranges>

namespace ds_tn
{

auto close_per_element(const Tensor& lhs, const Tensor& rhs, f64 tolerance) -> bool
{
    if (lhs.validity() != TensorValidity::valid or rhs.validity() != TensorValidity::valid)
    {
        return false;
    }

    return close_per_element(lhs.array(), rhs.array(), tolerance);
}

auto close_accumulated(const Tensor& lhs, const Tensor& rhs, f64 tolerance) -> bool
{
    if (lhs.validity() != TensorValidity::valid or rhs.validity() != TensorValidity::valid)
    {
        return false;
    }

    return close_accumulated(lhs.array(), rhs.array(), tolerance);
}

auto is_zero(const Tensor& tensor, f64 tolerance) -> bool
{
    if (tolerance < 0.0 or tensor.validity() != TensorValidity::valid)
    {
        return false;
    }

    return std::ranges::all_of(
        iota_n(tensor.size()),
        [&](usize index) { return std::abs(tensor.data()[index]) <= tolerance; }
    );
}

}  // namespace ds_tn
