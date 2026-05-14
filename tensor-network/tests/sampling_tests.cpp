#include "tensor/sampling.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <random>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto product_peps_2x2(bool fully_padded = false) -> Peps
{
    auto peps = Peps{Peps::Config{
        .n_rows = 2,
        .n_cols = 2,
        .bond_dim = 2,
        .physical_dim = 2,
        .fully_padded = fully_padded,
    }};

    for (auto& tensor : peps.tensors())
    {
        std::ranges::fill(tensor.data(), tensor.data() + tensor.size(), 0.0);
        for (auto linear = 0zu; linear < tensor.size(); ++linear)
        {
            const auto indices = tensor.indices_from_linear(linear);
            const auto virtual_indices_are_zero =
                indices[Peps::k_leg_right] == 0zu and indices[Peps::k_leg_top] == 0zu
                and indices[Peps::k_leg_left] == 0zu and indices[Peps::k_leg_bottom] == 0zu;
            if (virtual_indices_are_zero)
            {
                const auto spin = indices[Peps::k_leg_physical];
                tensor.data()[linear] = spin == 0zu ? 2.0 : 1.0;
            }
        }
    }

    return peps;
}

[[nodiscard]] auto probability_sum(std::span<const RowProbability> options) -> f64
{
    return std::accumulate(
        options.begin(),
        options.end(),
        f64{0.0},
        [](f64 sum, const RowProbability& option) { return sum + option.probability; }
    );
}

}  // namespace

TEST_CASE("base encoding helpers round-trip spin configurations", "[tensor][sampling]")
{
    const auto spins = std::array<usize, 3>{1, 0, 1};

    REQUIRE(encode_base(spins, 2) == 5zu);
    REQUIRE(std::ranges::equal(decode_base(5, 3, 2), spins));
    REQUIRE(spin_configuration_to_string(spins) == "101");
    REQUIRE(spin_configuration_to_string(std::array<usize, 3>{0, 1, 12}) == "01[12]");
}

TEST_CASE(
    "exact PEPS amplitudes and probabilities are enumerable on a small PEPS", "[tensor][sampling]"
)
{
    const auto peps = product_peps_2x2();
    const auto spins = std::array<usize, 4>{0, 1, 1, 0};

    REQUIRE(peps_amplitude(peps, spins) == Catch::Approx(4.0));

    const auto distribution = exact_peps_distribution(peps);
    const auto encoded = encode_base(spins, 2);

    REQUIRE(distribution.states.size() == 16zu);
    REQUIRE(distribution.norm_squared == Catch::Approx(625.0));
    REQUIRE(distribution.states[encoded].amplitude == Catch::Approx(4.0));
    REQUIRE(distribution.states[encoded].weight == Catch::Approx(16.0));
    REQUIRE(distribution.states[encoded].probability == Catch::Approx(16.0 / 625.0));
}

TEST_CASE("exact PEPS amplitudes cap fully padded outer boundary legs", "[tensor][sampling]")
{
    auto peps = product_peps_2x2(true);
    peps(0, 0)(0, 1, 0, 0, 0) = 1000.0;
    peps(0, 1)(1, 1, 0, 0, 0) = 1000.0;
    peps(1, 0)(0, 0, 0, 1, 0) = 1000.0;
    peps(1, 1)(1, 0, 0, 1, 0) = 1000.0;

    const auto spins = std::array<usize, 4>{0, 1, 1, 0};

    REQUIRE(std::ranges::equal(peps(0, 0).shape(), std::array<usize, 5>{2, 2, 2, 2, 2}));
    REQUIRE(peps_amplitude(peps, spins) == Catch::Approx(4.0));
    REQUIRE(exact_peps_distribution(peps).norm_squared == Catch::Approx(625.0));
}

TEST_CASE(
    "conditional row probabilities implement the direct-sampling chain rule", "[tensor][sampling]"
)
{
    const auto distribution = exact_peps_distribution(product_peps_2x2());

    const auto first_row = conditional_row_probabilities(distribution, 0, {});
    REQUIRE(first_row.size() == 4zu);
    REQUIRE(probability_sum(first_row) == Catch::Approx(1.0));
    REQUIRE(first_row[0].probability == Catch::Approx(16.0 / 25.0));
    REQUIRE(first_row[1].probability == Catch::Approx(4.0 / 25.0));
    REQUIRE(first_row[2].probability == Catch::Approx(4.0 / 25.0));
    REQUIRE(first_row[3].probability == Catch::Approx(1.0 / 25.0));

    const auto prefix = std::array<usize, 2>{0, 1};
    const auto second_row = conditional_row_probabilities(distribution, 1, prefix);
    REQUIRE(second_row.size() == 4zu);
    REQUIRE(probability_sum(second_row) == Catch::Approx(1.0));
    REQUIRE(second_row[0].probability == Catch::Approx(16.0 / 25.0));
    REQUIRE(second_row[1].probability == Catch::Approx(4.0 / 25.0));
    REQUIRE(second_row[2].probability == Catch::Approx(4.0 / 25.0));
    REQUIRE(second_row[3].probability == Catch::Approx(1.0 / 25.0));
}

TEST_CASE(
    "direct sampling returns a configuration with the exact chain probability", "[tensor][sampling]"
)
{
    const auto distribution = exact_peps_distribution(product_peps_2x2());
    auto rng = std::mt19937_64{17};

    const auto sample = sample_direct_exact(distribution, rng);
    const auto encoded_sample = encode_base(sample.spins, distribution.physical_dim);

    REQUIRE(sample.spins.size() == 4zu);
    REQUIRE(sample.steps.size() == 2zu);
    REQUIRE(sample.steps[0].options.size() == 4zu);
    REQUIRE(sample.steps[1].options.size() == 4zu);
    REQUIRE(sample.probability == Catch::Approx(distribution.states[encoded_sample].probability));
}

}  // namespace ds_tn
