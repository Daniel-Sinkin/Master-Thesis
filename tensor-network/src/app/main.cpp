// app/main.cpp
#include "models/transverse_ising.hpp"
#include "ndarray/lapack.hpp"
#include "tensor/contraction.hpp"
#include "tensor/mps.hpp"

#include <algorithm>
#include <print>

int main()
{
    using namespace ds_tn;

    const auto mpo = transverse_ising_mpo(4, 1.0, 1.0);
    (void) mpo;

    const auto num_sites = mpo.size();
    const auto max_bond_dim = 5;
    auto mps = random_mps(
        num_sites,
        RandomMPSConfig{
            .physical_dim = 2,
            .max_bond_dim = max_bond_dim,
            .seed = 0zu,
        }
    );

    for (auto k = num_sites - 1; k >= 1; --k)
    {
        auto& curr = mps(k);
        auto& next = mps(k - 1);

        const auto bond_left = curr.shape(0);
        const auto d = curr.shape(1);
        const auto bond_right = curr.shape(2);

        const auto reshaped = NDArray::reshape(curr.array(), {bond_left, d * bond_right});
        const auto [Q, R] = qr(reshaped, MatrixTransform::transpose);

        curr.array() = NDArray::reshape(Q, {bond_left, d, bond_right});

        next = contract(next, Tensor{R, {curr.leg_name(0), next.leg_name(2)}});
    }
}
