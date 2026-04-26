// app/main.cpp
#include "permutation/permutation.hpp"
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

    // Copying
    auto _0_0 = peps(0, 0);
    auto _1_0 = peps(1, 0);

    _0_0.rename_leg("b0,0", "t1,0");
    const auto top_left = contract(_0_0, _1_0);
    top_left.print_metadata({.name = "top_left after contraction"});
    // got r0,0; t0,0; l0,0; p0,0; r1,0; l1,0; b1,0; p1,0
    const auto grouped = apply_permutation(top_left, Permutation{6, 2, 0, 3, 7, 1, 4, 5});
    grouped.print_metadata({.name = "top_left grouped for boundary MPS"});
}
