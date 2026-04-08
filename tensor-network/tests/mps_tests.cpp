#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "tensor/compare.hpp"
#include "tensor/contraction.hpp"
#include "tensor/mps.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto contract_all(const MPS& mps) -> Tensor
{
    if (mps.size() == 0)
    {
        throw std::invalid_argument("contract_all requires a non-empty MPS.");
    }

    auto out = mps[0];
    for (auto site = 1zu; site < mps.size(); ++site)
    {
        out = contract(out, mps[site]);
    }
    return out;
}

[[nodiscard]] auto identity_matrix(usize size) -> NDArray
{
    auto out = NDArray({size, size});
    for (auto i = 0zu; i < size; ++i)
    {
        out(i, i) = 1.0;
    }
    return out;
}

[[nodiscard]] auto is_left_orthogonal(const Tensor& tensor, f64 tolerance = 1.e-10) -> bool
{
    const auto reshaped = tensor.array().reshape({tensor.shape(0) * tensor.shape(1), tensor.shape(2)});
    return close_accumulated(gram_matrix(reshaped), identity_matrix(tensor.shape(2)), tolerance);
}

[[nodiscard]] auto is_right_orthogonal(const Tensor& tensor, f64 tolerance = 1.e-10) -> bool
{
    const auto reshaped = tensor.array().reshape({tensor.shape(0), tensor.shape(1) * tensor.shape(2)});
    return close_accumulated(
        gram_matrix(transpose_matrix(reshaped)), identity_matrix(tensor.shape(0)), tolerance
    );
}

}  // namespace

TEST_CASE("to_mps exactly reconstructs a rank-3 NDArray without truncation", "[tensor][mps]")
{
    const auto data = NDArray::iota(8).reshape({2, 2, 2});

    const auto mps = to_mps(data);
    const auto reconstructed = contract_all(mps).array().squeeze();

    REQUIRE(mps.size() == 3zu);
    REQUIRE(mps[0].leg_name(0) == "edge_left");
    REQUIRE(mps[0].leg_name(2) == "bond_01");
    REQUIRE(mps[1].leg_name(0) == "bond_01");
    REQUIRE(mps[1].leg_name(2) == "bond_12");
    REQUIRE(mps[2].leg_name(0) == "bond_12");
    REQUIRE(mps[2].leg_name(2) == "edge_right");
    REQUIRE(close_per_element(reconstructed, data, 1e-10));
}

TEST_CASE("to_mps respects max bond dimension on a separable tensor", "[tensor][mps]")
{
    auto data = NDArray({2, 3, 2});
    for (auto i = 0zu; i < 2; ++i)
    {
        for (auto j = 0zu; j < 3; ++j)
        {
            for (auto k = 0zu; k < 2; ++k)
            {
                data(i, j, k) = static_cast<f64>((i + 1) * (j + 2) * (k + 3));
            }
        }
    }

    const auto mps = to_mps(data, 1);
    const auto reconstructed = contract_all(mps).array().squeeze();

    REQUIRE(mps.size() == 3zu);
    REQUIRE(mps[0].shape(2) == 1zu);
    REQUIRE(mps[1].shape(0) == 1zu);
    REQUIRE(mps[1].shape(2) == 1zu);
    REQUIRE(mps[2].shape(0) == 1zu);
    REQUIRE(close_per_element(reconstructed, data, 1e-10));
}

TEST_CASE("to_mps validates its inputs", "[tensor][mps]")
{
    REQUIRE_THROWS_AS(to_mps(NDArray::scalar(1.0)), std::invalid_argument);
    REQUIRE_THROWS_AS(to_mps(NDArray::vector(1.0, 2.0, 3.0), 0), std::invalid_argument);
}

TEST_CASE("random_mps creates expected tensor shapes and shared bond labels", "[tensor][mps]")
{
    const auto mps = random_mps(
        4,
        {
            .physical_dim = 3,
            .max_bond_dim = 5,
        }
    );

    REQUIRE(mps.size() == 4zu);
    REQUIRE(std::ranges::equal(mps[0].shape(), std::array<usize, 3>{1, 3, 5}));
    REQUIRE(std::ranges::equal(mps[1].shape(), std::array<usize, 3>{5, 3, 5}));
    REQUIRE(std::ranges::equal(mps[2].shape(), std::array<usize, 3>{5, 3, 5}));
    REQUIRE(std::ranges::equal(mps[3].shape(), std::array<usize, 3>{5, 3, 1}));

    REQUIRE(mps[0].leg_name(0) == "edge_left");
    REQUIRE(mps[0].leg_name(2) == "bond_01");
    REQUIRE(mps[1].leg_name(0) == "bond_01");
    REQUIRE(mps[1].leg_name(2) == "bond_12");
    REQUIRE(mps[2].leg_name(0) == "bond_12");
    REQUIRE(mps[2].leg_name(2) == "bond_23");
    REQUIRE(mps[3].leg_name(0) == "bond_23");
    REQUIRE(mps[3].leg_name(2) == "edge_right");
}

TEST_CASE("random_mps is deterministic when seeded", "[tensor][mps]")
{
    const auto lhs = random_mps(
        3,
        {
            .physical_dim = 2,
            .max_bond_dim = 4,
            .seed = 17,
        }
    );
    const auto rhs = random_mps(
        3,
        {
            .physical_dim = 2,
            .max_bond_dim = 4,
            .seed = 17,
        }
    );

    REQUIRE(lhs.size() == rhs.size());
    for (auto site = 0zu; site < lhs.size(); ++site)
    {
        REQUIRE(close_per_element(lhs[site], rhs[site], 0.0));
        REQUIRE(std::ranges::equal(lhs[site].leg_names(), rhs[site].leg_names()));
    }
}

TEST_CASE("random_mps validates its inputs", "[tensor][mps]")
{
    REQUIRE_THROWS_AS(random_mps(0), std::invalid_argument);
    REQUIRE_THROWS_AS(random_mps(2, {.physical_dim = 0, .max_bond_dim = 2}), std::invalid_argument);
    REQUIRE_THROWS_AS(random_mps(2, {.physical_dim = 2, .max_bond_dim = 0}), std::invalid_argument);
}

TEST_CASE("MPS call operator aliases the stored tensor", "[tensor][mps]")
{
    auto mps = random_mps(
        3,
        {
            .physical_dim = 2,
            .max_bond_dim = 4,
            .seed = 11,
        }
    );

    const auto before = mps[1].data()[0];
    auto& via_call = mps(1);
    via_call.data()[0] = before + 1.0;

    REQUIRE(&via_call == &mps[1]);
    REQUIRE(mps[1].data()[0] == Catch::Approx(before + 1.0));

    const auto& const_mps = mps;
    REQUIRE(&const_mps(1) == &const_mps[1]);
}

TEST_CASE("MPS left_orthogonalize preserves the represented tensor", "[tensor][mps]")
{
    auto mps = random_mps(
        4,
        {
            .physical_dim = 2,
            .max_bond_dim = 5,
            .seed = 7,
        }
    );
    const auto original = contract_all(mps);

    mps.left_orthogonalize();

    REQUIRE(close_accumulated(contract_all(mps), original, 1.e-10));
    for (auto site = 0zu; site + 1 < mps.size(); ++site)
    {
        REQUIRE(is_left_orthogonal(mps(site)));
    }
}

TEST_CASE("MPS right_orthogonalize preserves the represented tensor", "[tensor][mps]")
{
    auto mps = random_mps(
        4,
        {
            .physical_dim = 2,
            .max_bond_dim = 5,
            .seed = 13,
        }
    );
    const auto original = contract_all(mps);

    mps.right_orthogonalize();

    REQUIRE(close_accumulated(contract_all(mps), original, 1.e-10));
    for (auto site = 1zu; site < mps.size(); ++site)
    {
        REQUIRE(is_right_orthogonal(mps(site)));
    }
}

TEST_CASE("MPS orthogonalization validates adjacent bond labels", "[tensor][mps]")
{
    auto mps = random_mps(
        3,
        {
            .physical_dim = 2,
            .max_bond_dim = 4,
            .seed = 19,
        }
    );
    mps(1).rename_leg(mps(1).leg_name(0), "broken_bond");

    REQUIRE_THROWS_AS(mps.left_orthogonalize(), std::invalid_argument);
    REQUIRE_THROWS_AS(mps.right_orthogonalize(), std::invalid_argument);
}

}  // namespace ds_tn
