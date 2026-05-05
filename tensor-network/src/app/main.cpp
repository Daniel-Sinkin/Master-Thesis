// app/main.cpp
#include "permutation/permutation.hpp"
#include "tensor/contraction.hpp"
#include "tensor/mps.hpp"
#include "tensor/peps.hpp"

#include <format>
#include <print>
#include <string>
#include <vector>

int main()
{
    using namespace ds_tn;

    const auto peps = random_peps(
        3,
        5,
        RandomPepsConfig{
            .random_options = RandomNormalOptions{.mu = 0.0, .sigma = 0.1},
            .seed = 7,
        }
    );

    std::println("PEPS metadata:");
    peps.print_metadata({.include_memory = true});

    const auto left_boundary_name = [](usize col) -> std::string {
        if (col == 0zu)
        {
            return "boundary_row0_edge_left";
        }
        return std::format("boundary_row0_bond_{}_{}", col - 1zu, col);
    };
    const auto right_boundary_name = [](usize col, usize n_cols) -> std::string {
        if (col + 1zu == n_cols)
        {
            return "boundary_row0_edge_right";
        }
        return std::format("boundary_row0_bond_{}_{}", col, col + 1zu);
    };

    auto boundary_sites = std::vector<Tensor>{};
    boundary_sites.reserve(peps.n_cols());

    for (auto col = 0zu; col < peps.n_cols(); ++col)
    {
        const auto& top_row_site = peps(0zu, col);

        auto dummy_top = Tensor{
            std::vector<usize>{1zu, top_row_site.shape(Peps::k_leg_top), 1zu},
            {
                "dummy_left",
                top_row_site.leg_name(Peps::k_leg_top),
                "dummy_right",
            }
        };
        dummy_top(0zu, 0zu, 0zu) = 1.0;

        const auto contracted = contract(dummy_top, top_row_site);
        const auto grouped = apply_permutation(contracted, Permutation{0, 4, 5, 1, 3, 2});

        if (col == 0zu)
        {
            grouped.print_metadata("first_row_site_0_grouped");
        }

        const auto shape = grouped.shape();
        boundary_sites.emplace_back(
            grouped.array().reshape({shape[0] * shape[1], shape[2] * shape[3], shape[4] * shape[5]}),
            std::vector<std::string>{
                left_boundary_name(col),
                std::format("boundary_row0_middle_{}", col),
                right_boundary_name(col, peps.n_cols()),
            }
        );
    }

    const auto first_boundary = MPS{std::move(boundary_sites)};
    first_boundary(0zu).print_metadata("first_boundary_site_0_reduced");
}
