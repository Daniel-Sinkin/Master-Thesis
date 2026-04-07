// app/main.cpp
#include "ndarray/blas.hpp"     // IWYU pragma: keep
#include "ndarray/compare.hpp"  // IWYU pragma: keep
#include "ndarray/lapack.hpp"   // IWYU pragma: keep
#include "ndarray/ndarray.hpp"  // IWYU pragma: keep
#include "ndarray/stats.hpp"    // IWYU pragma: keep
#include "tensor/tensor.hpp"    // IWYU pragma: keep

#include <cassert>
#include <expected>
#include <functional>
#include <print>
#include <string_view>
#include <utility>

namespace ds_tn
{
constexpr f64 k_lanczos_termination_eps{1.e-12};

// Takes in a vector v and returns A * v, A must be symmetrical
using LanczosApplyAOperator = std::function<NDArray(const NDArray&)>;

struct LanczosResult
{
    NDArray ritz_vector{};
    f64 ritz_value{};
};
enum class LanczosError : u8
{
    not_matrix,
    matrix_not_square,
    matrix_not_symmetric,
    invalid_dimension,
    invalid_iteration_count,
    invalid_operator_output,
};
[[nodiscard]] constexpr auto to_string(LanczosError error) noexcept -> std::string_view
{
    switch (error)
    {
        case LanczosError::not_matrix:
            return "not_matrix";
        case LanczosError::matrix_not_square:
            return "matrix_not_square";
        case LanczosError::matrix_not_symmetric:
            return "matrix_not_symmetric";
        case LanczosError::invalid_dimension:
            return "invalid_dimension";
        case LanczosError::invalid_iteration_count:
            return "invalid_iteration_count";
        case LanczosError::invalid_operator_output:
            return "invalid_operator_output";
    }

    return "unknown_lanczos_error";
}
struct LanczosConfig
{
    usize num_iterations{25};

    RandomOptions random_options{RandomNormalOptions{.mu = 0.0, .sigma = 0.1}};
    std::optional<NDArraySeed> seed{};

    bool do_reorthogonalization{false};

    bool check_symmetric{false};
    f64 symmetry_tolerance{1e-12};

    bool verbose{false};
};

auto lanczos(usize dimension, const LanczosApplyAOperator& apply_A_operator, LanczosConfig cfg = {})
    -> std::expected<LanczosResult, LanczosError>
{
    const auto n = dimension;
    const auto m = cfg.num_iterations;

    if (n == 0)
    {
        return std::unexpected{LanczosError::invalid_dimension};
    }
    if (m == 0)
    {
        return std::unexpected{LanczosError::invalid_iteration_count};
    }

    if (cfg.verbose)
    {
        std::println(
            "Lanczos start: dimension={}, iterations={}, reorthogonalization={}",
            n,
            m,
            cfg.do_reorthogonalization
        );
    }

    std::vector<f64> alphas;
    std::vector<f64> betas;
    std::vector<NDArray> vs;

    alphas.reserve(m);
    betas.reserve(m);
    vs.reserve(m);

    auto v_prev = NDArray({n});
    auto v_curr = NDArray::random({n}, cfg.random_options, cfg.seed);
    v_curr.normalize();
    vs.push_back(v_curr);

    if (cfg.verbose)
    {
        std::println("Initial normalized Lanczos vector:");
        v_curr.print(6);
        std::println("");
    }

    for (auto iter = 0zu; iter < m; ++iter)
    {
        // Av <- A * v_curr
        auto Av = apply_A_operator(v_curr);
        if (!Av.is_vector() || Av.shape(0) != n)
        {
            return std::unexpected{LanczosError::invalid_operator_output};
        }
        {  // Three-term recurrence
            // Av <- Av - (Av.v_curr) * v_curr
            alphas.push_back(dot_product(Av, v_curr));
            axpy(-alphas.back(), v_curr, Av);
            if (iter > 0)
            {
                // Av <- Av - beta[k - 1] * v_prev
                axpy(-betas.back(), v_prev, Av);
            }
        }

        if (cfg.verbose)
        {
            std::println("iter {}: alpha = {:.12f}", iter, alphas.back());
        }
        if (cfg.do_reorthogonalization)
        {
            for (auto j = 0zu; j < iter; ++j)
            {
                axpy(-dot_product(Av, vs[j]), vs[j], Av);
            }
        }
        const auto norm = Av.l2_norm();
        if (cfg.verbose)
        {
            std::println("iter {}: residual norm = {:.12e}", iter, norm);
        }
        if (norm < k_lanczos_termination_eps)
        {
            if (cfg.verbose)
            {
                std::println(
                    "Lanczos terminated early at iter {} with residual norm {:.12e}.", iter, norm
                );
            }
            break;
        }
        betas.push_back(norm);
        v_prev = v_curr;
        Av /= norm;
        v_curr = std::move(Av);
        vs.push_back(v_curr);
    }

    const auto krylov_dimension = alphas.size();
    betas.resize(krylov_dimension > 0 ? krylov_dimension - 1 : 0);
    vs.resize(krylov_dimension);

    if (cfg.verbose)
    {
        std::println("Lanczos Krylov dimension: {}", krylov_dimension);
    }

    const auto [evals, evecs] = sym_tri_eigendecomp({.diagonal = alphas, .off_diagonal = betas});

    if (cfg.verbose)
    {
        std::println("Smallest Ritz value: {:.12f}", evals(0));
    }

    const auto compute_ritz_vector = [&]
    {
        auto out = NDArray({n});
        for (auto k = 0zu; k < krylov_dimension; ++k)
        {
            axpy(evecs(k, 0), vs[k], out);
        }
        out.normalize();
        return out;
    };
    auto result = LanczosResult{.ritz_vector = compute_ritz_vector(), .ritz_value = evals(0)};

    if (cfg.verbose)
    {
        std::println("Normalized Ritz vector:");
        result.ritz_vector.print(6);
        std::println("");
    }

    return result;
}
auto lanczos(const NDArray& A, LanczosConfig cfg = {}) -> std::expected<LanczosResult, LanczosError>
{
    if (!A.is_matrix())
    {
        return std::unexpected{LanczosError::not_matrix};
    }
    if (A.shape(0) != A.shape(1))
    {
        return std::unexpected{LanczosError::matrix_not_square};
    }
    if (cfg.check_symmetric && !is_symmetric(A, cfg.symmetry_tolerance))
    {
        return std::unexpected{LanczosError::matrix_not_symmetric};
    }
    const auto apply_A = [&A](const NDArray& v) -> NDArray { return matrix_vector_product(A, v); };
    return lanczos(A.shape(0), apply_A, cfg);
}
}  // namespace ds_tn

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
