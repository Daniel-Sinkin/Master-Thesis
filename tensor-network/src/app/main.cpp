// app/main.cpp
#include "ndarray/blas.hpp"    // IWYU pragma: keep
#include "ndarray/ndarray.hpp" // IWYU pragma: keep
#include "ndarray/stats.hpp"   // IWYU pragma: keep
#include "tensor/tensor.hpp"   // IWYU pragma: keep

namespace ds_tn {

struct LanczosConfig {
    bool precondition_checks{true};
    bool iteration_checks{false}; // expensive
    RandomNormalOptions random_options{.mu = 0.0, .sigma = 0.1};
    std::optional<NDArraySeed> seed{};
};
struct LanczosResult {
    NDArray x;
    f64 lambda;
};
auto lanczos(NDArray A, usize m = 20, LanczosConfig cfg = {}) -> LanczosResult {
    return {NDArray::scalar(0.0), lambda{}};
}
} // namespace ds_tn

int main() {
    using namespace ds_tn;

    constexpr usize n{3};
    constexpr usize m{20};

    const auto A = NDArray::random_normal({n, n}, {.sigma = 0.1});

    auto v_prev = NDArray({n});
    auto v_curr = NDArray::random_normal({n}, {.sigma = 0.1});
    v_curr.normalize();

    NDArray Av{{n}};
    auto update_Av = [&A, &v_curr, &Av] { matrix_vector_product(A, v_curr, Av); };

    std::vector<f64> alphas;
    std::vector<f64> betas;
    std::vector<NDArray> vs;

    alphas.reserve(m);
    betas.reserve(m);
    vs.reserve(m);

    vs.push_back(v_curr);

    for (auto iter = 0zu; iter < m; ++iter) {
        if constexpr (false) {
            assert(alphas.size() == iter);
            assert(betas.size() == iter);
            assert(vs.size() == iter + 1);
        }
        update_Av();
        alphas.push_back(dot_product(Av, v_curr));

        axpy(-alphas.back(), v_curr, Av);
        if (iter > 0) {
            axpy(betas.back(), v_prev, Av);
        }

        for (auto j = 0zu; j < iter; ++j) {
            axpy(-dot_product(Av, vs[j]), vs[j], Av);
        }
    }
}
