// app/main.cpp
#include "ndarray/blas.hpp"    // IWYU pragma: keep
#include "ndarray/ndarray.hpp" // IWYU pragma: keep
#include "ndarray/stats.hpp"   // IWYU pragma: keep
#include "tensor/tensor.hpp"   // IWYU pragma: keep

#include <iostream>

int main() {
    using namespace ds_tn;

    constexpr usize n{3};
    const auto A = NDArray::random_normal({n, n}, 0.0, 1.0, 7);
    const auto x = NDArray::random_normal({n}, 0.0, 1.0, 11);
    const auto y = matrix_vector_product(A, x);
    auto tensor = Tensor{y, {"result"}};

    tensor.print();
    std::cout << "l2(y) = " << l2_norm(y) << '\n';

    return 0;
}
