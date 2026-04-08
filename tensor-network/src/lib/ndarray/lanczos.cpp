// lib/ndarray/lanczos.cpp
#include "ndarray/lanczos.hpp"

#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "ndarray/lapack.hpp"

#include <print>
#include <utility>
#include <vector>

namespace ds_tn
{
namespace
{

constexpr f64 k_lanczos_termination_eps{1.e-12};

}  // namespace

auto lanczos(usize dimension, const LanczosApplyAOperator& apply_A_operator, LanczosConfig cfg)
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
        auto Av = apply_A_operator(v_curr);
        if (!Av.is_vector() || Av.shape(0) != n)
        {
            return std::unexpected{LanczosError::invalid_operator_output};
        }

        alphas.push_back(dot_product(Av, v_curr));
        axpy(-alphas.back(), v_curr, Av);
        if (iter > 0)
        {
            axpy(-betas.back(), v_prev, Av);
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

    auto eig = sym_tri_eigendecomp({.diagonal = alphas, .off_diagonal = betas});
    if (!eig)
    {
        return std::unexpected{LanczosError::tridiagonal_eigendecomp_failed};
    }
    const auto& [evals, evecs] = *eig;

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

auto lanczos(const NDArray& A, LanczosConfig cfg) -> std::expected<LanczosResult, LanczosError>
{
    if (!A.is_matrix())
    {
        return std::unexpected{LanczosError::not_matrix};
    }
    if (A.shape(0) != A.shape(1))
    {
        return std::unexpected{LanczosError::matrix_not_square};
    }
    if (cfg.check_symmetric and !is_symmetric(A, cfg.symmetry_tolerance))
    {
        return std::unexpected{LanczosError::matrix_not_symmetric};
    }

    const auto apply_A = [&A](const NDArray& v) -> NDArray { return matrix_vector_product(A, v); };
    return lanczos(A.shape(0), apply_A, cfg);
}

}  // namespace ds_tn
