// app/main.cpp
#include "ndarray/ndarray.hpp"
#include "tensor/contraction.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <expected>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <print>
#include <sstream>
#include <vector>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto format_indices(std::span<const usize> indices) -> std::string
{
    auto out = std::ostringstream{};
    out << '{';
    for (auto i = 0zu; i < indices.size(); ++i)
    {
        if (i != 0)
        {
            out << ", ";
        }
        out << indices[i];
    }
    out << '}';
    return out.str();
}

[[nodiscard]] auto format_names(std::span<const std::string> names) -> std::string
{
    auto out = std::ostringstream{};
    out << '{';
    for (auto i = 0zu; i < names.size(); ++i)
    {
        if (i != 0)
        {
            out << ", ";
        }
        out << '"' << names[i] << '"';
    }
    out << '}';
    return out.str();
}

[[nodiscard]] auto format_shape(std::span<const usize> shape) -> std::string
{
    auto out = std::ostringstream{};
    out << '(';
    for (auto axis = 0zu; axis < shape.size(); ++axis)
    {
        if (axis != 0)
        {
            out << ", ";
        }
        out << shape[axis];
    }
    out << ')';
    return out.str();
}

}  // namespace

enum class ReshapeError
{
    wrong_total,
    allocation_failed,
    empty_shape,
    invalid_array
};

template <typename T>
    requires requires(T t, T s) {
        { t * s } -> std::convertible_to<T>;
        T{1};
    }
auto product(std::span<const T> values) -> T
{
    return std::accumulate(values.begin(), values.end(), T{1}, std::multiplies<>{});
}

auto reshape(const NDArray& array, std::initializer_list<usize> new_shape) noexcept
    -> std::expected<NDArray, ReshapeError>
{
    if (new_shape.begin() == new_shape.end())
    {
        return std::unexpected{ReshapeError::empty_shape};
    }
    if (array.validity() != NDArrayValidity::valid)
    {
        return std::unexpected{ReshapeError::invalid_array};
    }
    if (product<usize>(new_shape) != array.size())
    {
        return std::unexpected{ReshapeError::wrong_total};
    }
    try
    {
        NDArray out{new_shape};
        std::ranges::copy(array.data(), array.data() + array.size(), out.data());
        return out;
    }
    catch (...)
    {
        return std::unexpected{ReshapeError::allocation_failed};
    }
}

}  // namespace ds_tn

int main()
{
    using namespace ds_tn;
    using namespace std::string_literals;

    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});

    {  // Partition
        const auto [l_rem, l_con, r_con, r_rem] = partition_indices(left, right);
        assert(l_rem.size() == 2 and (l_rem[0] == 2zu and l_rem[1] == 3zu));
        assert(l_con.size() == 2 and (l_con[0] == 1zu and l_con[1] == 0zu));
        assert(r_con.size() == 2 and (r_con[0] == 2zu and r_con[1] == 3zu));
        assert(r_rem.size() == 2 and (r_rem[0] == 0zu and r_rem[1] == 1zu));

#define FORMAT_INDEX(idx) std::println(#idx " = {}", format_indices((idx)));
        FORMAT_INDEX(l_rem);
        FORMAT_INDEX(l_con);
        FORMAT_INDEX(r_con);
        FORMAT_INDEX(r_rem);
    }

    {  // Leg contraction
        const auto contracted = contraction_output_tensor(left, right);

        const std::array expected_legs{"a"s, "b"s, "c"s, "d"s};
        assert(std::ranges::equal(contracted.leg_names(), expected_legs));

        std::vector<usize> expected_shape{5, 7, 11, 13};
        assert(std::ranges::equal(contracted.shape(), expected_shape));

        std::println("contracted legs  = {}", format_names(contracted.leg_names()));
        std::println("contracted shape = {}", format_shape(contracted.shape()));
        std::println("contracted       = {}", contracted.format_metadata());
    }
}
