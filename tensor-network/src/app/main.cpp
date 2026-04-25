// app/main.cpp
#include "tensor/contraction.hpp"
#include "tensor/peps.hpp"

#include <print>

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

    auto top_left = peps(0, 0);
    auto top_left_neighbor = peps(0, 1);

    std::println("\nCopied tensors before contraction:");
    top_left.print_metadata("top_left_copy");
    top_left_neighbor.print_metadata("top_left_neighbor_copy");

    top_left.rename_leg(
        top_left.leg_name(Peps::k_leg_right),
        top_left_neighbor.leg_name(Peps::k_leg_left)
    );

    std::println("\nCopied tensors after aligning the shared bond:");
    top_left.print_metadata("top_left_copy_aligned");
    top_left_neighbor.print_metadata("top_left_neighbor_copy");

    const auto contracted = contract(top_left, top_left_neighbor);

    std::println("\nContracted tensor metadata:");
    contracted.print_metadata("contracted_top_left_pair");

    std::println("\nPEPS metadata after the local contraction demo (network unchanged):");
    peps.print_metadata({.include_memory = true});
}
