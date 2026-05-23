using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

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

function arg_value(name::String, default::String)
    prefix = "--" * name * "="
    for arg in ARGS
        if startswith(arg, prefix)
            return arg[length(prefix)+1:end]
        end
    end
    return default
end

function has_flag(name::String)
    return any(==("--" * name), ARGS)
end

function json_string(x)
    s = string(x)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    s = replace(s, "\n" => "\\n")
    return "\"" * s * "\""
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

function make_peps(::Type{T}, lx::Int, ly::Int, bond_dim::Int;
                   contract_dim::Int, sample_dim::Int=contract_dim,
                   double_contract_dim::Int=bond_dim,
                   contract_cutoff=1e-4, sample_cutoff=1e-3,
                   seed::Int=1, alpha_init=3.0) where {T}
    Random.seed!(seed)
    hilbert = siteinds("S=1/2", lx, ly)
    peps = PEPS(T, hilbert;
        bond_dim,
        sample_dim,
        sample_cutoff,
        double_contract_dim,
        contract_dim,
        contract_cutoff,
        show_warning=false,
    )
    QNPEPS.multiply_algebraic_spectrum!(peps, alpha_init)
    return hilbert, peps
end

function make_matched_real_case(::Type{Tlo};
                                lx::Int=4, ly::Int=4, bond_dim::Int=3,
                                contract_dim::Int=72, sample_dim::Int=72,
                                double_contract_dim::Int=3,
                                contract_cutoff=1e-4, sample_cutoff=1e-3,
                                seed::Int=31) where {Tlo}
    hilbert64, peps64 = make_peps(Float64, lx, ly, bond_dim;
        contract_dim, sample_dim, double_contract_dim,
        contract_cutoff, sample_cutoff, seed)

    Random.seed!(seed + 1)
    hilbert_lo = siteinds("S=1/2", lx, ly)
    peps_lo = PEPS(Tlo, hilbert_lo;
        bond_dim,
        sample_dim,
        sample_cutoff,
        double_contract_dim,
        contract_dim,
        contract_cutoff,
        show_warning=false,
    )
    write!(peps_lo, Tlo.(vec(peps64)))

    ham = heisenberg_opsum(lx, ly)
    ham64 = QNG.TensorOperatorSum(ham, hilbert64)
    ham_lo = QNG.TensorOperatorSum(ham, hilbert_lo)
    return (; peps64, peps_lo, ham64, ham_lo, name="real_$(lx)x$(ly)_heisenberg_D$(bond_dim)")
end

function relative_error(a, b)
    return norm(a - b) / max(norm(a), norm(b), eps(Float64))
end

function max_abs_error(a, b)
    return maximum(abs.(a .- b))
end

function append_summary(path::String, fields)
    open(path, "a") do io
        parts = String[]
        for (k, v) in fields
            if v isa AbstractString
                push!(parts, json_string(k) * ":" * json_string(v))
            elseif v isa Bool
                push!(parts, json_string(k) * ":" * string(v))
            else
                push!(parts, json_string(k) * ":" * string(v))
            end
        end
        println(io, "{" * join(parts, ",") * "}")
    end
end

function write_timer(path::String, timer)
    open(path, "w") do io
        show(io, timer)
        println(io)
    end
end

function generate_samples(peps, sample_nr::Int; seed::Int, timer=TimerOutput())
    Random.seed!(seed)
    QNPEPS.update_double_layer_envs!(peps)
    samples = Vector{Matrix{Int}}(undef, sample_nr)
    logpcs = Vector{Float64}(undef, sample_nr)
    for s in 1:sample_nr
        sample, logpc, _ = @timeit timer "fixed_sample_generation" QNPEPS.get_sample(peps; mode=:full, timer)
        samples[s] = sample
        logpcs[s] = Float64(logpc)
    end
    return samples, logpcs
end

function compact_fixed_samples_minsr(peps, ham_op, samples::Vector{Matrix{Int}}, logpcs::Vector{Float64};
                                     solver=QNG.EigenSolver(1e-4, 0.0; verbose=false),
                                     timer=TimerOutput())
    sample_nr = length(samples)
    layout = QNPEPS.compact_parameter_layout(peps)
    compact_count = QNPEPS.compact_parameter_count(layout)
    T = eltype(peps)
    realT = real(T)
    rows = Vector{Vector{T}}(undef, sample_nr)
    Eks = Vector{T}(undef, sample_nr)
    logpsis = Vector{Complex{realT}}(undef, sample_nr)
    contract_dims = Vector{Int}(undef, sample_nr)

    @timeit timer "fixed_compact_Oks_and_Eks" begin
        for s in 1:sample_nr
            S = samples[s]
            logpsi, env_top, env_down, max_bond =
                @timeit timer "vertical_envs" QNPEPS.get_logψ_and_envs(peps, S)
            h_envs_r, h_envs_l =
                @timeit timer "horizontal_envs" QNPEPS.get_all_horizontal_envs(peps, env_top, env_down, S)
            logpsi_flipped = Dict{Any, Number}()
            Ek_terms =
                @timeit timer "precomp_sHψ_elems" QNG.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
            E_loc =
                @timeit timer "energy" QNPEPS.get_Ek(peps, ham_op, env_top, env_down, S, logpsi;
                                                     h_envs_r, h_envs_l,
                                                     logψ_flipped=logpsi_flipped,
                                                     Ek_terms, timer)
            row = Vector{T}(undef, compact_count)
            rows[s] =
                @timeit timer "compact_log_gradients" QNPEPS.get_compact_Ok(peps, env_top, env_down, S, logpsi;
                                                                            h_envs_r, h_envs_l,
                                                                            Ok=row, layout)
            if T <: Real
                Eks[s] = T(real(E_loc))
            else
                Eks[s] = T(E_loc)
            end
            logpsis[s] = Complex{realT}(logpsi)
            contract_dims[s] = max_bond
        end
    end

    raw_weights = QNPEPS.compute_importance_weights(logpsis, logpcs)
    weights = copy(raw_weights)
    weights ./= mean(weights)
    mean_dense =
        @timeit timer "direct_mean" QNPEPS._compact_weighted_mean_dense(rows, samples, weights, length(peps), layout)
    Tmat =
        @timeit timer "direct_T" QNPEPS._direct_centered_T(rows, samples, mean_dense, weights, layout)
    energy_centered, energy_mean = QNPEPS._center_energy(Eks, weights)
    raw =
        @timeit timer "direct_solve" -solver(Tmat, energy_centered)
    theta_dot =
        @timeit timer "direct_theta_dot" QNPEPS._direct_theta_dot(rows, samples, mean_dense, weights, raw, layout)

    return (; theta_dot, Tmat, Eks, energy_mean, logpsis, weights=raw_weights,
            normalised_weights=weights, contract_dims,
            compact_parameter_count=compact_count,
            dense_parameter_count=length(peps),
            timer)
end

function timed_run(f)
    GC.gc()
    live_before = Base.gc_live_bytes()
    timed = @timed f()
    GC.gc()
    live_after = Base.gc_live_bytes()
    return (; value=timed.value, elapsed_s=timed.time, allocated_bytes=timed.bytes,
            gc_time_s=timed.gctime, live_before, live_after)
end

function compare_precision(; sample_nr::Int, sample_seed::Int, outdir::String,
                           lx::Int=4, ly::Int=4, bond_dim::Int=3,
                           tag::String="main")
    mkpath(outdir)
    case = make_matched_real_case(Float32; lx, ly, bond_dim)
    sample_timer = TimerOutput()
    samples, logpcs = generate_samples(case.peps64, sample_nr; seed=sample_seed, timer=sample_timer)

    fp64 = timed_run(() -> compact_fixed_samples_minsr(case.peps64, case.ham64, samples, logpcs;
        solver=QNG.EigenSolver(1e-4, 0.0; verbose=false)))
    fp32 = timed_run(() -> compact_fixed_samples_minsr(case.peps_lo, case.ham_lo, samples, logpcs;
        solver=QNG.EigenSolver(1e-4, 0.0; verbose=false)))

    out64 = fp64.value
    out32 = fp32.value
    write_timer(joinpath(outdir, "$(tag)_fixed_sample_generation.timer.txt"), sample_timer)
    write_timer(joinpath(outdir, "$(tag)_fp64_fixed_compact.timer.txt"), out64.timer)
    write_timer(joinpath(outdir, "$(tag)_fp32_fixed_compact.timer.txt"), out32.timer)

    logpsi64 = ComplexF64.(out64.logpsis)
    logpsi32 = ComplexF64.(out32.logpsis)
    E64 = Float64.(real.(out64.Eks))
    E32 = Float64.(real.(out32.Eks))
    w64 = Float64.(out64.normalised_weights)
    w32 = Float64.(out32.normalised_weights)
    T64 = Matrix{Float64}(real.(out64.Tmat))
    T32 = Matrix{Float64}(real.(out32.Tmat))
    theta64 = Float64.(real.(out64.theta_dot))
    theta32 = Float64.(real.(out32.theta_dot))

    elem64 = sizeof(Float64)
    elem32 = sizeof(Float32)
    dense_params = out64.dense_parameter_count
    compact_params = out64.compact_parameter_count
    fp64_rows = sample_nr * compact_params * elem64
    fp32_rows = sample_nr * compact_params * elem32
    fp64_T = sample_nr * sample_nr * elem64
    fp32_T = sample_nr * sample_nr * elem32

    fields = [
        "timestamp" => string(now()),
        "tag" => tag,
        "scenario" => case.name,
        "sample_nr" => sample_nr,
        "sample_seed" => sample_seed,
        "dense_parameter_count" => dense_params,
        "compact_parameter_count" => compact_params,
        "compact_fraction" => compact_params / dense_params,
        "fp64_elapsed_s" => @sprintf("%0.6f", fp64.elapsed_s),
        "fp32_elapsed_s" => @sprintf("%0.6f", fp32.elapsed_s),
        "fp64_allocated_bytes" => fp64.allocated_bytes,
        "fp32_allocated_bytes" => fp32.allocated_bytes,
        "fp64_live_before" => fp64.live_before,
        "fp64_live_after" => fp64.live_after,
        "fp32_live_before" => fp32.live_before,
        "fp32_live_after" => fp32.live_after,
        "logpsi_rel_err" => @sprintf("%0.17g", relative_error(logpsi64, logpsi32)),
        "logpsi_max_abs_err" => @sprintf("%0.17g", max_abs_error(logpsi64, logpsi32)),
        "Eks_rel_err" => @sprintf("%0.17g", relative_error(E64, E32)),
        "Eks_max_abs_err" => @sprintf("%0.17g", max_abs_error(E64, E32)),
        "weights_rel_err" => @sprintf("%0.17g", relative_error(w64, w32)),
        "weights_max_abs_err" => @sprintf("%0.17g", max_abs_error(w64, w32)),
        "T_rel_err" => @sprintf("%0.17g", relative_error(T64, T32)),
        "T_max_abs_err" => @sprintf("%0.17g", max_abs_error(T64, T32)),
        "theta_rel_err" => @sprintf("%0.17g", relative_error(theta64, theta32)),
        "theta_max_abs_err" => @sprintf("%0.17g", max_abs_error(theta64, theta32)),
        "energy_mean_abs_err" => @sprintf("%0.17g", abs(Float64(real(out64.energy_mean)) - Float64(real(out32.energy_mean)))),
        "fp64_compact_row_bytes" => fp64_rows,
        "fp32_compact_row_bytes" => fp32_rows,
        "fp64_T_bytes" => fp64_T,
        "fp32_T_bytes" => fp32_T,
    ]
    append_summary(joinpath(outdir, "precision_summary.jsonl"), fields)
    @info "precision compared" tag scenario=case.name sample_nr fp64_elapsed_s=fp64.elapsed_s fp32_elapsed_s=fp32.elapsed_s theta_rel_err=relative_error(theta64, theta32) T_rel_err=relative_error(T64, T32) Eks_rel_err=relative_error(E64, E32)
end

function main()
    BLAS.set_num_threads(parse(Int, arg_value("blas-threads", "1")))
    sample_nr = parse(Int, arg_value("samples", "512"))
    sample_seed = parse(Int, arg_value("sample-seed", "9001"))
    lx = parse(Int, arg_value("lx", "4"))
    ly = parse(Int, arg_value("ly", "4"))
    bond_dim = parse(Int, arg_value("bond-dim", "3"))
    default_out = joinpath("research", "peps_cuda", "profiles", "julia_cpu",
        "precision_fp32_vs_fp64_" * Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    outdir = arg_value("out", default_out)

    open(joinpath(mkpath(outdir), "metadata.txt"), "w") do io
        println(io, "timestamp=", now())
        println(io, "julia_version=", VERSION)
        println(io, "threads=", Threads.nthreads())
        println(io, "blas_threads=", BLAS.get_num_threads())
        println(io, "samples=", sample_nr)
        println(io, "sample_seed=", sample_seed)
        println(io, "lx=", lx)
        println(io, "ly=", ly)
        println(io, "bond_dim=", bond_dim)
        println(io, "comparison=fixed FP64-generated sample set; FP32 PEPS parameters are rounded from FP64")
        println(io, "regularization=multiply_algebraic_spectrum!(alpha=3.0)")
    end

    if has_flag("warmup")
        compare_precision(; sample_nr=min(sample_nr, 16), sample_seed=sample_seed + 1,
                          outdir, lx, ly, bond_dim, tag="warmup")
        GC.gc()
    end

    compare_precision(; sample_nr, sample_seed, outdir, lx, ly, bond_dim, tag="main")
end

main()
