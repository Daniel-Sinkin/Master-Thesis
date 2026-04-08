// app/main.cpp
#include "tensor/contraction.hpp"
#include "tensor/tensor.hpp"

#include <print>

int main()
{
    using namespace ds_tn;
    auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    std::println("left = {}", left.format_metadata());

    auto right = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});
    std::println("right = {}", right.format_metadata());

    const auto result = contract(left, right);
    std::println("contraction(left, right) = {}", result.format_metadata());
}
