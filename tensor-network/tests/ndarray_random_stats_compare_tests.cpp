#include "ndarray/compare.hpp"
#include "ndarray/generator.hpp"
#include "ndarray/ndarray.hpp"
#include "ndarray/stats.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ds_tn {

TEST_CASE("NDArray generators are reproducible and validate their parameters", "[ndarray][random]") {
    auto generator_a = NDArrayGenerator(1234);
    auto generator_b = NDArrayGenerator(1234);

    const auto uniform_a = generator_a.uniform({3, 4}, -2.0, 3.0);
    const auto uniform_b = generator_b.uniform({3, 4}, -2.0, 3.0);
    REQUIRE(close_per_element(uniform_a, uniform_b, 0.0));

    for (usize index = 0; index < uniform_a.size(); ++index) {
        REQUIRE(uniform_a.data()[index] >= -2.0);
        REQUIRE(uniform_a.data()[index] <= 3.0);
    }

    const auto uniform_options_seed_a = NDArray::random_uniform(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        },
        77);
    const auto uniform_options_seed_b = NDArray::random_uniform(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        },
        77);
    REQUIRE(close_per_element(uniform_options_seed_a, uniform_options_seed_b, 0.0));

    const auto uniform_default_seed_a = NDArray::random_uniform(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        });
    const auto uniform_default_seed_b = NDArray::random_uniform(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        });
    REQUIRE(not close_per_element(uniform_default_seed_a, uniform_default_seed_b, 0.0));

    const auto normal_options_seed_a = NDArray::random_normal(
        {2, 3},
        RandomNormalOptions{
            .mu = 0.0,
            .sigma = 1.0,
        },
        99);
    const auto normal_options_seed_b = NDArray::random_normal(
        {2, 3},
        RandomNormalOptions{
            .mu = 0.0,
            .sigma = 1.0,
        },
        99);
    REQUIRE(close_per_element(normal_options_seed_a, normal_options_seed_b, 0.0));

    const auto normal_default_seed_a = NDArray::random_normal(
        {2, 3},
        RandomNormalOptions{
            .mu = 0.0,
            .sigma = 1.0,
        });
    const auto normal_default_seed_b = NDArray::random_normal(
        {2, 3},
        RandomNormalOptions{
            .mu = 0.0,
            .sigma = 1.0,
        });
    REQUIRE(not close_per_element(normal_default_seed_a, normal_default_seed_b, 0.0));

    REQUIRE(close_per_element(generator_a.uniform({4}, 2.5, 2.5), NDArray::vector(2.5, 2.5, 2.5, 2.5), 0.0));
    REQUIRE(close_per_element(generator_a.normal({4}, -3.0, 0.0), NDArray::vector(-3.0, -3.0, -3.0, -3.0), 0.0));

    REQUIRE_THROWS_AS(generator_a.uniform({2}, 2.0, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(generator_a.uniform({2}, 0.0, std::numeric_limits<f64>::infinity()), std::invalid_argument);
    REQUIRE_THROWS_AS(generator_a.normal({2}, 0.0, -1.0), std::invalid_argument);
}

TEST_CASE("NDArray norms and element summaries flatten over all entries", "[ndarray][stats]") {
    const auto matrix = NDArray::matrix({
        {3.0, 4.0},
        {-12.0, 0.0},
    });

    REQUIRE(l1_norm(matrix) == Catch::Approx(19.0));
    REQUIRE(l2_norm(matrix) == Catch::Approx(13.0));
    REQUIRE(lp_norm(matrix, 3.0) == Catch::Approx(std::pow(1819.0, 1.0 / 3.0)));
    REQUIRE(infinity_norm(matrix) == Catch::Approx(12.0));

    const auto summary = element_summary(matrix);
    REQUIRE(summary.min == Catch::Approx(-12.0));
    REQUIRE(summary.max == Catch::Approx(4.0));
    REQUIRE(summary.sum == Catch::Approx(-5.0));

    const auto empty = NDArray({0});
    REQUIRE(l1_norm(empty) == Catch::Approx(0.0));
    REQUIRE(l2_norm(empty) == Catch::Approx(0.0));
    REQUIRE(lp_norm(empty, 3.0) == Catch::Approx(0.0));
    REQUIRE_THROWS_AS(infinity_norm(empty), std::invalid_argument);
    REQUIRE_THROWS_AS(element_summary(empty), std::invalid_argument);
    REQUIRE_THROWS_AS(lp_norm(matrix, 0.0), std::invalid_argument);
}

TEST_CASE("NDArray comparison helpers are pure shape and data comparisons", "[ndarray][compare]") {
    const auto lhs = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });
    const auto rhs = NDArray::matrix({
        {1.0, 2.0 + 1e-6},
        {3.0, 4.0},
    });

    REQUIRE(close_per_element(lhs, rhs, 1e-5));
    REQUIRE(not close_per_element(lhs, rhs, 1e-7));
    REQUIRE(close_accumulated(lhs, rhs, 1e-5));
    REQUIRE(not close_accumulated(lhs, rhs, 1e-7));
    REQUIRE(not close_per_element(lhs, NDArray::vector(1.0, 2.0, 3.0, 4.0), 1e-5));
    REQUIRE(not close_accumulated(lhs, rhs, -1.0));
}

} // namespace ds_tn
