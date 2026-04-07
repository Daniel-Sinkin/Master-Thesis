#include "tensor/contraction.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ranges>
#include <vector>

namespace ds_tn
{

TEST_CASE("partition_indices separates shared and unshared legs with stable ordering", "[tensor]")
{
    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});

    const auto partition = partition_indices(left, right);

    REQUIRE(partition.left_not_shared == std::vector<usize>{2, 3});
    REQUIRE(partition.left_shared == std::vector<usize>{1, 0});
    REQUIRE(partition.right_shared == std::vector<usize>{2, 3});
    REQUIRE(partition.right_not_shared == std::vector<usize>{0, 1});
}

TEST_CASE("contraction_output_tensor preserves left then right leg ordering", "[tensor]")
{
    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});

    const auto output = contraction_output_tensor(left, right);
    const std::array<std::string, 4> expected_legs{"a", "b", "c", "d"};

    REQUIRE(output.shape(0) == 5zu);
    REQUIRE(output.shape(1) == 7zu);
    REQUIRE(output.shape(2) == 11zu);
    REQUIRE(output.shape(3) == 13zu);
    REQUIRE(std::ranges::equal(output.leg_names(), std::span<const std::string>{expected_legs}));

    for (auto index = 0zu; index < output.size(); ++index)
    {
        REQUIRE(output.data()[index] == Catch::Approx(0.0));
    }
}

TEST_CASE("contraction_output_tensor rejects mismatched shared extents", "[tensor]")
{
    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 4, 2}, {"c", "d", "i", "j"});

    REQUIRE_THROWS_AS(contraction_output_tensor(left, right), std::invalid_argument);
}

}  // namespace ds_tn
