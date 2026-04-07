// app/main.cpp
#include "tensor/contraction.hpp"
#include "tensor/tensor.hpp"

#include <array>
#include <cassert>
#include <print>
#include <ranges>
#include <sstream>
#include <vector>

namespace ds_tn
{
namespace
{

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
}  // namespace ds_tn

int main()
{
    using namespace ds_tn;

    const auto ti = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto tj = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});
    const auto expected_left = std::vector<std::string>{"a", "b"};
    const auto expected_right = std::vector<std::string>{"c", "d"};
    const auto expected_shared = std::vector<std::string>{"i", "j"};

    const auto partition = partition_indices(ti, tj);
    assert(partition.left == expected_left);
    assert(partition.right == expected_right);
    assert(partition.shared == expected_shared);

    std::println("partition.left   = {}", format_names(partition.left));
    std::println("partition.right  = {}", format_names(partition.right));
    std::println("partition.shared = {}", format_names(partition.shared));

    const auto contracted = contraction_output_tensor(ti, tj);
    const auto expected_legs = std::array<std::string, 4>{"a", "b", "c", "d"};
    assert(std::ranges::equal(contracted.leg_names(), std::span<const std::string>{expected_legs}));
    assert(contracted.shape(0) == 5zu);
    assert(contracted.shape(1) == 7zu);
    assert(contracted.shape(2) == 11zu);
    assert(contracted.shape(3) == 13zu);

    std::println("contracted legs  = {}", format_names(contracted.leg_names()));
    std::println("contracted shape = {}", format_shape(contracted.shape()));
    std::println("contracted       = {}", contracted.format_metadata());
}
