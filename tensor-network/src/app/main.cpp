// app/main.cpp
#include "ndarray/ndarray.hpp"
#include "permutation/permutation.hpp"

#include <cassert>

int main()
{
    using namespace ds_tn;

    const auto base = NDArray::tensor3({
        {
            {0.0, 1.0},
            {2.0, 3.0},
            {4.0, 5.0},
        },
        {
            {6.0, 7.0},
            {8.0, 9.0},
            {10.0, 11.0},
        },
    });
    const auto permutation = Permutation{1, 2, 0};
    assert(permutation.size() == base.rank());
    assert(permutation.at(0) == 1);
    assert(permutation[1] == 2);

    const auto permuted = apply_permutation(base, permutation);
    assert(permuted.shape(0) == 2);
    assert(permuted.shape(1) == 2);
    assert(permuted.shape(2) == 3);
    assert(permuted(0, 0, 1) == 2.0);
    assert(permuted(1, 1, 2) == 11.0);
}
