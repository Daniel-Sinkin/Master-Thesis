// app/main.cpp
#include "models/transverse_ising.hpp"
#include "tensor/mps.hpp"

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
    mps.right_orthogonalize();
}
