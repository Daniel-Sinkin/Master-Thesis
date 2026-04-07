// app/main.cpp
#include "ndarray/blas.hpp"     // IWYU pragma: keep
#include "ndarray/lanczos.hpp"  // IWYU pragma: keep
#include "ndarray/ndarray.hpp"  // IWYU pragma: keep

#include <print>

int main()
{
    using namespace ds_tn;

    const auto A = NDArray::random({10, 10}, RandomNormalOptions{}, 7);
    auto symmetric_A = gram_matrix(A);
    symmetric_A *= 0.5;

    if (const auto res = lanczos(symmetric_A, {.num_iterations = 30, .verbose = true}); res)
    {
        std::println("Lanczos smallest Ritz value: {:.8f}", res->ritz_value);
        res->ritz_vector.print(6);
        std::println("");
    }
    else
    {
        std::println(
            "Failed to perform lanczos for {}, got error code {}.",
            symmetric_A.format_metadata(),
            to_string(res.error())
        );
    }
}
