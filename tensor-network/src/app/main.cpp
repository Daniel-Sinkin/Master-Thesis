// app/main.cpp
#include "ndarray/blas.hpp"
#include "ndarray/lapack.hpp"
#include "ndarray/ndarray.hpp"

#include <print>  // IWYU pragma: keep

#define NAMED_PRINT(x)                                                                             \
    std::print(#x " = ");                                                                          \
    (x).print();

int main()
{
    using namespace ds_tn;

    auto arr = NDArray::reshape(NDArray::iota(8), {2, 2, 2});  // [2 x 2 x 2]
    {
        arr = NDArray::reshape(arr, {2, 4});  // [2 x 4]
        const auto [u, s, vt] = svd(arr);     // [2 x 2], [2], [2 x 4]

        const auto A1 = NDArray::reshape(u, {1, 2, 2});
        NAMED_PRINT(s);
        NAMED_PRINT(vt);
        const auto rem = matrix_matrix_product(s.diag(), vt);
        rem.print();
    }
}
