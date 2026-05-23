using LinearAlgebra
using Random
using Test

using ITensors
using QuantumNaturalGradient
using QuantumNaturalfPEPS
using TimerOutputs

const QNG = QuantumNaturalGradient
const QNPEPS = QuantumNaturalfPEPS

if !isdefined(QNPEPS, Symbol("deprecate_make_inds_match!"))
    @eval QNPEPS begin
        function deprecate_make_inds_match!(f, m1, m2, loginner;
                                            make_inds_match=true)
            return m1, m2
        end
    end
end

if !isdefined(QNPEPS, :check_hascommoninds)
    @eval QNPEPS begin
        check_hascommoninds(args...) = nothing
    end
end

if !isdefined(ITensors, Symbol("_log_or_not_dot"))
    @eval ITensors begin
        function _log_or_not_dot(args...; kwargs...)
            error("compatibility placeholder should be passed through only")
        end
    end
end

function heisenberg_opsum(lx::Int, ly::Int; j1=1.0)
    ham = OpSum()
    for i in 1:lx, j in 1:ly, op in ("X", "Y", "Z")
        if j < ly
            ham .+= (j1, op, (i, j), op, (i, j + 1))
        end
        if i < lx
            ham .+= (j1, op, (i, j), op, (i + 1, j))
        end
    end
    return ham
end

function make_peps(::Type{T}, lx::Int, ly::Int, bond_dim::Int; seed::Int) where {T}
    Random.seed!(seed)
    hilbert = siteinds("S=1/2", lx, ly)
    peps = PEPS(T, hilbert;
        bond_dim=bond_dim,
        sample_dim=32,
        sample_cutoff=1e-8,
        double_contract_dim=bond_dim,
        contract_dim=32,
        contract_cutoff=1e-8,
        show_warning=false,
    )
    QNPEPS.multiply_algebraic_spectrum!(peps, 3.0)
    return hilbert, peps
end

function relative_error(a, b)
    return norm(a - b) / max(norm(a), norm(b), eps(Float64))
end

function same_samples(a, b)
    length(a) == length(b) || return false
    return all(a[i] == b[i] for i in eachindex(a))
end

function check_compact_scatter(::Type{T}; seed::Int) where {T}
    lx, ly = 3, 2
    hilbert, peps = make_peps(T, lx, ly, 2; seed)
    ham_op = QNG.TensorOperatorSum(heisenberg_opsum(lx, ly), hilbert)
    timer = TimerOutput()
    QNPEPS.update_double_layer_envs!(peps)
    Random.seed!(seed + 1000)

    sample, _, env_top = QNPEPS.get_sample(peps; mode=:full, timer)
    logpsi, env_top, env_down, _ = QNPEPS.get_logψ_and_envs(peps, sample, env_top)
    h_envs_r, h_envs_l = QNPEPS.get_all_horizontal_envs(peps, env_top, env_down, sample)

    dense = QNPEPS.get_Ok(peps, env_top, env_down, sample, logpsi; h_envs_r, h_envs_l)
    layout = QNPEPS.compact_parameter_layout(peps)
    compact = QNPEPS.get_compact_Ok(peps, env_top, env_down, sample, logpsi;
                                    h_envs_r, h_envs_l, layout)
    scattered = zeros(T, length(peps))
    QNPEPS.scatter_compact_to_dense!(scattered, compact, sample, layout)

    @test length(compact) * 2 == length(dense)
    @test relative_error(scattered, dense) < (T <: Complex ? 1e-10 : 1e-11)
    return (; dense_len=length(dense), compact_len=length(compact))
end

function dense_minsr(::Type{T}; seed::Int, sample_seed::Int, sample_nr::Int) where {T}
    lx, ly = 3, 2
    hilbert, peps = make_peps(T, lx, ly, 2; seed)
    ham_op = QNG.TensorOperatorSum(heisenberg_opsum(lx, ly), hilbert)
    timer = TimerOutput()
    solver = QNG.EigenSolver(1e-5, 0.0; verbose=false)
    QNPEPS.update_double_layer_envs!(peps)
    Random.seed!(sample_seed)
    out = QNPEPS.Oks_and_Eks_singlethread(peps, ham_op, sample_nr; timer)
    ng = QNG.NaturalGradient(out[:Oks], out[:Eks], out[:logψs], out[:samples];
        importance_weights=copy(out[:weights]),
        solver,
        timer,
        verbose=false,
    )
    return (; theta=QNG.get_θdot(ng; θtype=T), Tmat=QNG.dense_T(ng.J),
            samples=out[:samples], weights=out[:weights], Eks=out[:Eks],
            timer, nparams=length(peps))
end

function direct_minsr(::Type{T}; seed::Int, sample_seed::Int, sample_nr::Int) where {T}
    lx, ly = 3, 2
    hilbert, peps = make_peps(T, lx, ly, 2; seed)
    ham_op = QNG.TensorOperatorSum(heisenberg_opsum(lx, ly), hilbert)
    timer = TimerOutput()
    solver = QNG.EigenSolver(1e-5, 0.0; verbose=false)
    Random.seed!(sample_seed)
    out = QNPEPS.direct_gram_minsr_singlethread(peps, ham_op, sample_nr;
        solver,
        timer,
        return_rows=true,
    )
    return (; theta=T.(out[:theta_dot]), Tmat=out[:T], samples=out[:samples],
            weights=out[:weights], Eks=out[:Eks], timer,
            nparams=out[:dense_parameter_count], compact=out[:compact_parameter_count])
end

function check_end_to_end(::Type{T}; seed::Int, sample_seed::Int, sample_nr::Int) where {T}
    dense = dense_minsr(T; seed, sample_seed, sample_nr)
    direct = direct_minsr(T; seed, sample_seed, sample_nr)

    @test same_samples(dense.samples, direct.samples)
    @test relative_error(dense.weights, direct.weights) < 1e-12
    @test relative_error(dense.Eks, direct.Eks) < (T <: Complex ? 1e-10 : 1e-11)
    @test relative_error(dense.Tmat, direct.Tmat) < (T <: Complex ? 1e-9 : 1e-10)
    @test relative_error(dense.theta, direct.theta) < (T <: Complex ? 1e-8 : 1e-9)
    @test direct.compact * 2 == direct.nparams
    return (; theta_err=relative_error(dense.theta, direct.theta),
            T_err=relative_error(dense.Tmat, direct.Tmat),
            compact=direct.compact,
            nparams=direct.nparams)
end

@testset "compact sampled-sector Ok" begin
    real_stats = check_compact_scatter(Float64; seed=101)
    complex_stats = check_compact_scatter(ComplexF64; seed=102)
    @info "compact scatter ok" real_stats complex_stats
end

@testset "direct Gram minSR equivalence" begin
    real_stats = check_end_to_end(Float64; seed=201, sample_seed=301, sample_nr=8)
    complex_stats = check_end_to_end(ComplexF64; seed=202, sample_seed=302, sample_nr=8)
    @info "direct minSR ok" real_stats complex_stats
end
