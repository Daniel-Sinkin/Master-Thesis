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

function safe_name(name::String)
    return replace(name, r"[^A-Za-z0-9_.-]" => "_")
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

function tfi_opsum(lx::Int, ly::Int; jz=1.0, hx=0.7)
    ham = OpSum()
    for i in 1:lx, j in 1:ly
        ham .+= (-hx, "X", (i, j))
        if j < ly
            ham .+= (jz, "Z", (i, j), "Z", (i, j + 1))
        end
        if i < lx
            ham .+= (jz, "Z", (i, j), "Z", (i + 1, j))
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

function make_case(name::String, sample_nr::Int)
    if name == "real_4x4"
        lx, ly = 4, 4
        hilbert, peps = make_peps(Float64, lx, ly, 2;
            contract_dim=64, sample_dim=64, double_contract_dim=2,
            contract_cutoff=1e-4, sample_cutoff=1e-3, seed=11)
        ham = heisenberg_opsum(lx, ly)
        return (; name="real_4x4_heisenberg_D2", peps, ham, eigencut=1e-4, sample_nr)
    elseif name == "real_4x4_d3"
        lx, ly = 4, 4
        hilbert, peps = make_peps(Float64, lx, ly, 3;
            contract_dim=72, sample_dim=72, double_contract_dim=3,
            contract_cutoff=1e-4, sample_cutoff=1e-3, seed=31)
        ham = heisenberg_opsum(lx, ly)
        return (; name="real_4x4_heisenberg_D3", peps, ham, eigencut=1e-4, sample_nr)
    elseif name == "real_5x5"
        lx, ly = 5, 5
        hilbert, peps = make_peps(Float64, lx, ly, 2;
            contract_dim=64, sample_dim=64, double_contract_dim=2,
            contract_cutoff=1e-4, sample_cutoff=1e-3, seed=41)
        ham = heisenberg_opsum(lx, ly)
        return (; name="real_5x5_heisenberg_D2", peps, ham, eigencut=1e-4, sample_nr)
    elseif name == "complex_4x4"
        lx, ly = 4, 4
        j1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
        j2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
        lambda = 2 * sin(0.06 * pi)
        hilbert, peps = make_peps(ComplexF64, lx, ly, 2;
            contract_dim=48, sample_dim=48, double_contract_dim=2,
            contract_cutoff=1e-4, sample_cutoff=1e-3, seed=12)
        ham = QNPEPS.hamiltonain_CSL(hilbert, j1, j2, lambda)
        return (; name="complex_4x4_csl_D2", peps, ham, eigencut=1e-4, sample_nr)
    elseif name == "real_3x3"
        lx, ly = 3, 3
        hilbert, peps = make_peps(Float64, lx, ly, 2;
            contract_dim=32, sample_dim=32, double_contract_dim=2,
            contract_cutoff=1e-8, sample_cutoff=1e-8, seed=23)
        ham = tfi_opsum(lx, ly)
        return (; name="real_3x3_tfi_D2", peps, ham, eigencut=1e-5, sample_nr)
    else
        error("unknown scenario: $(name)")
    end
end

function relative_error(a, b)
    return norm(a - b) / max(norm(a), norm(b), eps(Float64))
end

function same_samples(a, b)
    length(a) == length(b) || return false
    return all(a[i] == b[i] for i in eachindex(a))
end

function write_timer(path::String, timer)
    open(path, "w") do io
        show(io, timer)
        println(io)
    end
end

as_ham_op(ham::QNG.TensorOperatorSum, hilbert) = ham
as_ham_op(ham, hilbert) = QNG.TensorOperatorSum(ham, hilbert)

function run_dense(case, sample_seed::Int)
    timer = TimerOutput()
    solver = QNG.EigenSolver(case.eigencut, 0.0; verbose=false)
    ham_op = as_ham_op(case.ham, siteinds(case.peps))
    QNPEPS.update_double_layer_envs!(case.peps)
    Random.seed!(sample_seed)
    out = QNPEPS.Oks_and_Eks_singlethread(case.peps, ham_op, case.sample_nr; timer)
    ng = QNG.NaturalGradient(out[:Oks], out[:Eks], out[:logψs], out[:samples];
        importance_weights=copy(out[:weights]),
        solver,
        timer,
        verbose=false,
    )
    return (; theta=QNG.get_θdot(ng; θtype=eltype(case.peps)),
            Tmat=QNG.dense_T(ng.J),
            samples=out[:samples],
            weights=out[:weights],
            Eks=out[:Eks],
            timer,
            nparams=length(case.peps),
            compact=QNPEPS.compact_parameter_count(QNPEPS.compact_parameter_layout(case.peps)))
end

function run_direct(case, sample_seed::Int)
    timer = TimerOutput()
    solver = QNG.EigenSolver(case.eigencut, 0.0; verbose=false)
    ham_op = as_ham_op(case.ham, siteinds(case.peps))
    Random.seed!(sample_seed)
    out = QNPEPS.direct_gram_minsr_singlethread(case.peps, ham_op, case.sample_nr;
        solver,
        timer,
    )
    return (; theta=eltype(case.peps).(out[:theta_dot]),
            Tmat=out[:T],
            samples=out[:samples],
            weights=out[:weights],
            Eks=out[:Eks],
            timer,
            nparams=out[:dense_parameter_count],
            compact=out[:compact_parameter_count])
end

function timed_run(f)
    GC.gc()
    live_before = Base.gc_live_bytes()
    rss_before = maxrss_bytes()
    timed = @timed f()
    GC.gc()
    live_after = Base.gc_live_bytes()
    rss_after = maxrss_bytes()
    return (; value=timed.value, elapsed_s=timed.time, allocated_bytes=timed.bytes,
            gc_time_s=timed.gctime, live_before, live_after, rss_before, rss_after)
end

function maxrss_bytes()
    if isdefined(Sys, :maxrss)
        return Sys.maxrss()
    end
    return -1
end

function memory_model(::Type{T}, sample_nr::Int, nparams::Int, compact::Int) where {T}
    elem = sizeof(T)
    fp32_elem = T <: Complex ? 8 : 4
    dense_rows = sample_nr * nparams * elem
    compact_rows = sample_nr * compact * elem
    gram = sample_nr * sample_nr * elem
    dense_mean = nparams * elem
    fp32_dense_rows = sample_nr * nparams * fp32_elem
    fp32_compact_rows = sample_nr * compact * fp32_elem
    fp32_gram = sample_nr * sample_nr * fp32_elem
    return (; elem, dense_rows, compact_rows, gram, dense_mean,
            fp32_elem, fp32_dense_rows, fp32_compact_rows, fp32_gram)
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

function run_compare(; scenario::String, sample_nr::Int, sample_seed::Int, outdir::String,
                     repeat_idx::Int=1)
    dense_case = make_case(scenario, sample_nr)
    direct_case = make_case(scenario, sample_nr)
    name = dense_case.name

    dense = timed_run(() -> run_dense(dense_case, sample_seed))
    direct = timed_run(() -> run_direct(direct_case, sample_seed))

    dense_out = dense.value
    direct_out = direct.value
    theta_err = relative_error(dense_out.theta, direct_out.theta)
    T_err = relative_error(dense_out.Tmat, direct_out.Tmat)
    weights_err = relative_error(dense_out.weights, direct_out.weights)
    Eks_err = relative_error(dense_out.Eks, direct_out.Eks)
    samples_match = same_samples(dense_out.samples, direct_out.samples)
    model = memory_model(eltype(dense_case.peps), sample_nr, dense_out.nparams, direct_out.compact)

    label = safe_name(name * "_Ns$(sample_nr)_R$(repeat_idx)")
    write_timer(joinpath(outdir, label * "_dense.timer.txt"), dense_out.timer)
    write_timer(joinpath(outdir, label * "_direct.timer.txt"), direct_out.timer)

    append_summary(joinpath(outdir, "direct_minsr_summary.jsonl"), [
        "timestamp" => string(now()),
        "scenario" => name,
        "repeat_idx" => repeat_idx,
        "sample_nr" => sample_nr,
        "sample_seed" => sample_seed,
        "element_type" => string(eltype(dense_case.peps)),
        "nparams_dense" => dense_out.nparams,
        "nparams_compact" => direct_out.compact,
        "compact_fraction" => direct_out.compact / dense_out.nparams,
        "dense_elapsed_s" => @sprintf("%0.6f", dense.elapsed_s),
        "direct_elapsed_s" => @sprintf("%0.6f", direct.elapsed_s),
        "speedup_dense_over_direct" => @sprintf("%0.6f", dense.elapsed_s / direct.elapsed_s),
        "dense_allocated_bytes" => dense.allocated_bytes,
        "direct_allocated_bytes" => direct.allocated_bytes,
        "dense_live_before" => dense.live_before,
        "dense_live_after" => dense.live_after,
        "dense_maxrss_before" => dense.rss_before,
        "dense_maxrss_after" => dense.rss_after,
        "direct_live_before" => direct.live_before,
        "direct_live_after" => direct.live_after,
        "direct_maxrss_before" => direct.rss_before,
        "direct_maxrss_after" => direct.rss_after,
        "theta_rel_err" => @sprintf("%0.17g", theta_err),
        "T_rel_err" => @sprintf("%0.17g", T_err),
        "weights_rel_err" => @sprintf("%0.17g", weights_err),
        "Eks_rel_err" => @sprintf("%0.17g", Eks_err),
        "samples_match" => samples_match,
        "dense_row_bytes_fp64_or_complex" => model.dense_rows,
        "compact_row_bytes_fp64_or_complex" => model.compact_rows,
        "gram_bytes_fp64_or_complex" => model.gram,
        "dense_mean_bytes_fp64_or_complex" => model.dense_mean,
        "dense_row_bytes_fp32_or_complex32" => model.fp32_dense_rows,
        "compact_row_bytes_fp32_or_complex32" => model.fp32_compact_rows,
        "gram_bytes_fp32_or_complex32" => model.fp32_gram,
    ])

    @info "compared" scenario=name sample_nr dense_elapsed_s=dense.elapsed_s direct_elapsed_s=direct.elapsed_s theta_err T_err samples_match
end

function main()
    BLAS.set_num_threads(parse(Int, arg_value("blas-threads", "1")))
    scenario = arg_value("scenario", "real_4x4")
    sample_nr = parse(Int, arg_value("samples", "128"))
    sample_seed = parse(Int, arg_value("sample-seed", "777"))
    repeats = parse(Int, arg_value("repeats", "1"))
    default_out = joinpath("research", "peps_cuda", "profiles", "julia_cpu",
        "direct_minsr_" * Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    outdir = arg_value("out", default_out)
    mkpath(outdir)

    open(joinpath(outdir, "metadata.txt"), "w") do io
        println(io, "timestamp=", now())
        println(io, "julia_version=", VERSION)
        println(io, "threads=", Threads.nthreads())
        println(io, "blas_threads=", BLAS.get_num_threads())
        println(io, "scenario=", scenario)
        println(io, "samples=", sample_nr)
        println(io, "sample_seed=", sample_seed)
        println(io, "repeats=", repeats)
        println(io, "regularization=multiply_algebraic_spectrum!(alpha=3.0)")
        println(io, "compiled_modules_note=run with --compiled-modules=no for this reference harness")
    end

    if has_flag("warmup")
        run_compare(; scenario, sample_nr=min(sample_nr, 8),
                    sample_seed=sample_seed + 1, outdir, repeat_idx=0)
        GC.gc()
    end

    for repeat_idx in 1:repeats
        run_compare(; scenario, sample_nr,
                    sample_seed=sample_seed + 1000 * (repeat_idx - 1),
                    outdir, repeat_idx)
        GC.gc()
    end
end

main()
