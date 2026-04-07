// app/main.cpp
#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "tensor/tensor.hpp"

#include <iostream>

int main() {
    using namespace ds_tn;

    const auto matrix = NDArray::matrix({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    });
    const auto vector = NDArray::vector(7.0, 8.0, 9.0);
    const auto product = matrix_vector_product(matrix, vector);

    auto named_product = Tensor(product, {"result"});

    if (not close_per_element(product, NDArray::vector(50.0, 122.0))) {
        return 1;
    }

    named_product.print();
    std::cout << "dot(product, product) = " << dot_product(product, product) << '\n';

    return 0;
}
