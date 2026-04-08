#include "permutation/permutation.hpp"
#include "tensor/tensor.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace ds_tn
{

TEST_CASE("Permutation validates mappings and supports forward and inverse application", "[permutation]")
{
    const auto permutation = Permutation{1, 2, 0};

    REQUIRE(permutation.size() == 3zu);
    REQUIRE(permutation.at(0) == 1zu);
    REQUIRE(permutation[1] == 2zu);
    REQUIRE(permutation.apply(std::vector<usize>{10, 20, 30}) == std::vector<usize>{30, 10, 20});
    REQUIRE(
        permutation.apply_inverse(std::vector<usize>{30, 10, 20}) == std::vector<usize>{10, 20, 30}
    );

    REQUIRE_THROWS_AS((Permutation{1, 1, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS((Permutation{0, 2}), std::invalid_argument);
}

TEST_CASE("apply_permutation permutes NDArray shape and values", "[permutation][ndarray]")
{
    const auto base = NDArray::rank3({
        {
            {0.0, 1.0},
            {2.0, 3.0},
            {4.0, 5.0},
        },
        {
            {6.0, 7.0},
            {8.0, 9.0},
            {10.0, 11.0},
        },
    });

    const auto permuted = apply_permutation(base, Permutation{1, 2, 0});

    REQUIRE(permuted.shape(0) == 2zu);
    REQUIRE(permuted.shape(1) == 2zu);
    REQUIRE(permuted.shape(2) == 3zu);
    REQUIRE(permuted(0, 0, 1) == Catch::Approx(2.0));
    REQUIRE(permuted(1, 1, 2) == Catch::Approx(11.0));

    REQUIRE_THROWS_AS(apply_permutation(base, Permutation{0, 1}), std::invalid_argument);
}

TEST_CASE("apply_permutation permutes Tensor shape, values, and leg names", "[permutation][tensor]")
{
    const auto base = Tensor(
        NDArray::rank3({
            {
                {0.0, 1.0},
                {2.0, 3.0},
                {4.0, 5.0},
            },
            {
                {6.0, 7.0},
                {8.0, 9.0},
                {10.0, 11.0},
            },
        }),
        {"j", "i", "a"}
    );

    const auto permuted = apply_permutation(base, Permutation{1, 2, 0});
    const std::array<std::string, 3> expected_legs{"a", "j", "i"};

    REQUIRE(permuted.shape(0) == 2zu);
    REQUIRE(permuted.shape(1) == 2zu);
    REQUIRE(permuted.shape(2) == 3zu);
    REQUIRE(std::ranges::equal(permuted.leg_names(), std::span<const std::string>{expected_legs}));
    REQUIRE(permuted(0, 0, 1) == Catch::Approx(2.0));
    REQUIRE(permuted(1, 1, 2) == Catch::Approx(11.0));

    REQUIRE_THROWS_AS(apply_permutation(base, Permutation{0, 1}), std::invalid_argument);
}

}  // namespace ds_tn
