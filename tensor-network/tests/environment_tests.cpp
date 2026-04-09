#include "models/transverse_ising.hpp"
#include "tensor/compare.hpp"
#include "tensor/contraction.hpp"
#include "tensor/environment.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <format>

namespace ds_tn
{
namespace
{

template <typename TensorLike>
[[nodiscard]] auto contract_all(TensorLike tensors) -> Tensor
{
    if (tensors.size() == 0)
    {
        throw std::invalid_argument("contract_all requires a non-empty tensor chain.");
    }

    auto out = tensors[0];
    for (auto site = 1zu; site < tensors.size(); ++site)
    {
        out = contract(out, tensors[site]);
    }
    return out;
}

[[nodiscard]] auto squeeze_tensor(const Tensor& tensor) -> Tensor
{
    auto leg_names = std::vector<std::string>{};
    leg_names.reserve(tensor.rank());
    for (auto axis = 0zu; axis < tensor.rank(); ++axis)
    {
        if (tensor.shape(axis) == 1)
        {
            continue;
        }
        leg_names.push_back(tensor.leg_name(axis));
    }
    return Tensor{tensor.array().squeeze(), std::move(leg_names)};
}

[[nodiscard]] auto physical_state_tensor(const MPS& mps) -> Tensor
{
    return squeeze_tensor(contract_all(mps.tensors()));
}

[[nodiscard]] auto physical_mpo_tensor(std::span<const Tensor> mpo) -> Tensor
{
    return squeeze_tensor(contract_all(mpo));
}

[[nodiscard]] auto full_expectation_value(const MPS& mps, std::span<const Tensor> mpo) -> Tensor
{
    auto ket = physical_state_tensor(mps);
    for (auto site = 0zu; site < mps.size(); ++site)
    {
        ket.rename_leg(std::format("physical_{}", site), std::format("physical_in_{}", site));
    }

    auto bra = physical_state_tensor(mps);
    for (auto site = 0zu; site < mps.size(); ++site)
    {
        bra.rename_leg(std::format("physical_{}", site), std::format("physical_out_{}", site));
    }

    return contract(bra, contract(physical_mpo_tensor(mpo), ket));
}

[[nodiscard]] auto left_boundary_environment() -> Tensor
{
    auto out = Tensor(
        std::vector<usize>{1, 1, 1},
        std::array<std::string, 3>{"bra_edge_left", "edge_left", "ket_edge_left"}
    );
    out.data()[0] = 1.0;
    return out;
}

}  // namespace

TEST_CASE("right_boundary_environment creates the trivial right boundary tensor", "[tensor][environment]")
{
    const auto mpo = transverse_ising_mpo(4, 1.0, 1.0);
    const auto mps = random_mps(
        4,
        {
            .physical_dim = 2,
            .max_bond_dim = 5,
            .seed = 7,
        }
    );

    const auto env = right_boundary_environment(mps[3], mpo[3]);

    REQUIRE(std::ranges::equal(env.shape(), std::array<usize, 3>{1, 1, 1}));
    REQUIRE(
        std::ranges::equal(
            env.leg_names(),
            std::array<std::string, 3>{"bra_edge_right", "edge_right", "ket_edge_right"}
        )
    );
    REQUIRE(env.data()[0] == 1.0);
}

TEST_CASE("right_environments carry the expected open bond triple at each site", "[tensor][environment]")
{
    auto mpo = transverse_ising_mpo(4, 1.0, 1.0);
    auto mps = random_mps(
        4,
        {
            .physical_dim = 2,
            .max_bond_dim = 5,
            .seed = 13,
        }
    );
    mps.right_orthogonalize();

    const auto envs = right_environments(mps, mpo);

    REQUIRE(envs.size() == 4zu);

    REQUIRE(std::ranges::equal(envs[3].shape(), std::array<usize, 3>{1, 1, 1}));
    REQUIRE(
        std::ranges::equal(
            envs[3].leg_names(),
            std::array<std::string, 3>{"bra_edge_right", "edge_right", "ket_edge_right"}
        )
    );

    REQUIRE(
        std::ranges::equal(
            envs[2].shape(),
            std::array<usize, 3>{mps[2].shape(2), mpo[2].shape(3), mps[2].shape(2)}
        )
    );
    REQUIRE(
        std::ranges::equal(
            envs[2].leg_names(),
            std::array<std::string, 3>{"bra_bond_23", "bond_23", "ket_bond_23"}
        )
    );

    REQUIRE(
        std::ranges::equal(
            envs[1].shape(),
            std::array<usize, 3>{mps[1].shape(2), mpo[1].shape(3), mps[1].shape(2)}
        )
    );
    REQUIRE(
        std::ranges::equal(
            envs[1].leg_names(),
            std::array<std::string, 3>{"bra_bond_12", "bond_12", "ket_bond_12"}
        )
    );

    REQUIRE(
        std::ranges::equal(
            envs[0].shape(),
            std::array<usize, 3>{mps[0].shape(2), mpo[0].shape(3), mps[0].shape(2)}
        )
    );
    REQUIRE(
        std::ranges::equal(
            envs[0].leg_names(),
            std::array<std::string, 3>{"bra_bond_01", "bond_01", "ket_bond_01"}
        )
    );
}

TEST_CASE("right_environments reproduce the full expectation value when closed on the left", "[tensor][environment]")
{
    const auto mpo = transverse_ising_mpo(4, 1.0, 0.7);
    auto mps = random_mps(
        4,
        {
            .physical_dim = 2,
            .max_bond_dim = 4,
            .seed = 23,
        }
    );
    mps.right_orthogonalize();

    const auto envs = right_environments(mps, mpo);
    const auto via_env = contract(left_boundary_environment(), update_right_environment(envs[0], mps[0], mpo[0]));
    const auto expected = full_expectation_value(mps, mpo);

    REQUIRE(via_env.is_scalar());
    REQUIRE(expected.is_scalar());
    REQUIRE(close_accumulated(via_env, expected, 1.e-10));
}

TEST_CASE("right_environments validates MPO bond compatibility", "[tensor][environment]")
{
    auto mpo = transverse_ising_mpo(3, 1.0, 1.0);
    const auto mps = random_mps(
        3,
        {
            .physical_dim = 2,
            .max_bond_dim = 4,
            .seed = 29,
        }
    );
    mpo[1].rename_leg(mpo[1].leg_name(0), "broken_bond");

    REQUIRE_THROWS_AS(right_environments(mps, mpo), std::invalid_argument);
}

}  // namespace ds_tn
