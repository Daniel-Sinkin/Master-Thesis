// app/main.cpp
#include "models/transverse_ising.hpp"
#include "tensor/environment.hpp"
#include "tensor/mps.hpp"
#include "tensor/tensor.hpp"

#include <format>

int main()
{
    using namespace ds_tn;

    const auto lhs = Tensor::random_normal({10, 20, 30, 40, 50, 60});
    lhs.print_metadata();

    std::println("Hello");

    if constexpr (false)
    {
        const auto mpo = transverse_ising_mpo(4, 1.0, 1.0);
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
        const auto envs = right_environments(mps, mpo);

        for (auto site = 0zu; site < envs.size(); ++site)
        {
            envs[site].print_metadata(std::format("right_env_{}", site));
        }
    }
}
