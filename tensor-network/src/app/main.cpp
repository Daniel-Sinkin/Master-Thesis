// app/main.cpp
#include "ndarray/blas.hpp"     // IWYU pragma: keep
#include "ndarray/lanczos.hpp"  // IWYU pragma: keep
#include "ndarray/ndarray.hpp"  // IWYU pragma: keep

#include <print>

namespace ds_tn
{
auto contract_simple(const NDArray& A, const NDArray& B, usize axis_a, usize axis_b) -> NDArray
{

    return {};
}
}  // namespace ds_tn

int main()
{
    using namespace ds_tn;

    const auto A = []
    {
        auto out = NDArray::random({10, 10}, RandomNormalOptions{}, 7);
        out = gram_matrix(out);
        out *= 0.5;
        return out;
    }();

    if (const auto res = lanczos(A, {.num_iterations = 30, .verbose = false}); res)
    {
        auto [ritz_vec, ritz_val] = *res;
        std::println("Lanczos smallest Ritz value: {:.8f}", ritz_val);
        ritz_vec.print(6);
        auto Av = matrix_vector_product(A, ritz_vec);
        axpy(-ritz_val, ritz_vec, Av);
        Av.print(6);
    }
    else
    {
        std::println(
            "Failed to perform lanczos for {}, got error code {}.",
            A.format_metadata(),
            to_string(res.error())
        );
    }
}
