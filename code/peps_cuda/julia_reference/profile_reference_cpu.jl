using Dates
using Distributed
using LinearAlgebra
using Printf
using Profile
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

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
    return path
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

function tfi_opsum(lx::Int, ly::Int; jz=1.0, hx=0.5)
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
        bond_dim=bond_dim,
        sample_dim=sample_dim,
        sample_cutoff=sample_cutoff,
        double_contract_dim=double_contract_dim,
        contract_dim=contract_dim,
        contract_cutoff=contract_cutoff,
        show_warning=false,
    )
    QNPEPS.multiply_algebraic_spectrum!(peps, alpha_init)
    return hilbert, peps
end

function make_heisenberg_example(; sample_nr::Int, maxiter::Int)
    lx = 4
    ly = 4
    hilbert, peps = make_peps(Float64, lx, ly, 2;
        contract_dim=64, sample_dim=64, double_contract_dim=2,
        contract_cutoff=1e-4, sample_cutoff=1e-3, seed=11)
    ham = heisenberg_opsum(lx, ly; j1=1.0)
    return (; name="example_heisenberg_4x4_d2",
            peps, ham, sample_nr, maxiter, lr=0.05, eigencut=1e-4)
end

function make_csl_example(; sample_nr::Int, maxiter::Int)
    lx = 4
    ly = 4
    j1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
    j2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
    lambda = 2 * sin(0.06 * pi)
    hilbert, peps = make_peps(ComplexF64, lx, ly, 2;
        contract_dim=48, sample_dim=48, double_contract_dim=2,
        contract_cutoff=1e-4, sample_cutoff=1e-3, seed=12)
    ham = QNPEPS.hamiltonain_CSL(hilbert, j1, j2, lambda)
    return (; name="example_csl_4x4_d2",
            peps, ham, sample_nr, maxiter, lr=0.05, eigencut=1e-4)
end

function make_synthetic_real(; sample_nr::Int, maxiter::Int)
    lx = 3
    ly = 2
    hilbert, peps = make_peps(Float64, lx, ly, 2;
        contract_dim=32, sample_dim=32, double_contract_dim=2,
        contract_cutoff=1e-8, sample_cutoff=1e-8, seed=21)
    ham = heisenberg_opsum(lx, ly; j1=1.0)
    return (; name="synthetic_real_3x2_d2_heisenberg",
            peps, ham, sample_nr, maxiter, lr=0.03, eigencut=1e-5)
end

function make_synthetic_complex(; sample_nr::Int, maxiter::Int)
    lx = 3
    ly = 2
    hilbert, peps = make_peps(ComplexF64, lx, ly, 2;
        contract_dim=32, sample_dim=32, double_contract_dim=2,
        contract_cutoff=1e-8, sample_cutoff=1e-8, seed=22)
    ham = heisenberg_opsum(lx, ly; j1=1.0)
    return (; name="synthetic_complex_3x2_d2_heisenberg",
            peps, ham, sample_nr, maxiter, lr=0.03, eigencut=1e-5)
end

function make_synthetic_tfi(; sample_nr::Int, maxiter::Int)
    lx = 3
    ly = 3
    hilbert, peps = make_peps(Float64, lx, ly, 2;
        contract_dim=32, sample_dim=32, double_contract_dim=2,
        contract_cutoff=1e-8, sample_cutoff=1e-8, seed=23)
    ham = tfi_opsum(lx, ly; jz=1.0, hx=0.7)
    return (; name="synthetic_real_3x3_d2_tfi",
            peps, ham, sample_nr, maxiter, lr=0.03, eigencut=1e-5)
end

function maxrss_bytes()
    if isdefined(Sys, :maxrss)
        return Sys.maxrss()
    end
    return -1
end

function run_scenario(case; threaded::Bool, multiproc::Bool=false)
    timer = TimerOutput()
    oks_eks = QNPEPS.generate_Oks_and_Eks(case.peps, case.ham;
        threaded=threaded, multiproc=multiproc, timer)
    integrator = QNG.Euler(lr=case.lr)
    solver = QNG.EigenSolver(case.eigencut, verbose=false)
    theta = QNG.Parameters(case.peps)
    callback = (; kwargs...) -> nothing
    loss, trained_theta, misc = QNG.evolve(oks_eks, theta;
        integrator,
        verbosity=0,
        solver,
        sample_nr=case.sample_nr,
        maxiter=case.maxiter,
        callback,
        timer,
    )
    return (; timer, loss, misc, trained_theta)
end

function write_profile(path::String)
    open(path, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, maxdepth=60,
                      mincount=2)
    end
end

function write_timer(path::String, timer)
    open(path, "w") do io
        show(io, timer)
        println(io)
    end
end

function run_profiled(factory, outdir::String;
                      sample_nr::Int, maxiter::Int, threaded::Bool,
                      multiproc::Bool, do_warmup::Bool)
    if do_warmup
        warm_case = factory(; sample_nr=sample_nr, maxiter=maxiter)
        try
            run_scenario(warm_case; threaded=threaded, multiproc=multiproc)
        catch err
            @warn "warmup failed" case=warm_case.name exception=(err, catch_backtrace())
        end
        GC.gc()
    end

    case = factory(; sample_nr=sample_nr, maxiter=maxiter)
    mode = multiproc ? "_multiproc" : (threaded ? "_threaded" : "_singlethread")
    label = safe_name(case.name * mode)
    profile_path = joinpath(outdir, label * ".profile.txt")
    timer_path = joinpath(outdir, label * ".timer.txt")
    summary_path = joinpath(outdir, "summary.jsonl")

    Profile.clear()
    Profile.init(; delay=0.001)
    start = time()
    rss0 = maxrss_bytes()
    gc0 = Base.gc_live_bytes()
    status = "ok"
    message = ""
    loss = NaN
    timer = TimerOutput()
    try
        result = Profile.@profile run_scenario(case; threaded=threaded,
                                               multiproc=multiproc)
        timer = result.timer
        loss = Float64(real(result.loss))
    catch err
        status = "error"
        message = sprint(showerror, err)
    end
    elapsed = time() - start
    gc1 = Base.gc_live_bytes()
    rss1 = maxrss_bytes()

    write_profile(profile_path)
    write_timer(timer_path, timer)

    open(summary_path, "a") do io
        println(io,
            "{\"name\":\"", case.name, "\",",
            "\"threaded\":", threaded, ",",
            "\"multiproc\":", multiproc, ",",
            "\"sample_nr\":", sample_nr, ",",
            "\"maxiter\":", maxiter, ",",
            "\"status\":\"", status, "\",",
            "\"elapsed_s\":", @sprintf("%0.6f", elapsed), ",",
            "\"loss\":", @sprintf("%0.17g", loss), ",",
            "\"gc_live_before\":", gc0, ",",
            "\"gc_live_after\":", gc1, ",",
            "\"maxrss_before\":", rss0, ",",
            "\"maxrss_after\":", rss1, ",",
            "\"message\":\"", replace(message, "\"" => "\\\"", "\n" => "\\n"), "\"",
            "}")
    end
    return (; case, status, elapsed, loss, profile_path, timer_path, message)
end

function main()
    default_out = joinpath("research", "peps_cuda", "profiles", "julia_cpu",
        Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    outdir = ensure_dir(arg_value("out", default_out))
    sample_nr = parse(Int, arg_value("samples", "4"))
    maxiter = parse(Int, arg_value("maxiter", "1"))
    threaded = has_flag("threaded")
    multiproc = has_flag("multiproc")
    worker_count = parse(Int, arg_value("workers", "0"))
    worker_threads = parse(Int, arg_value("worker-threads", "1"))
    do_warmup = !has_flag("no-warmup")

    if multiproc && worker_count > 0
        project = abspath("code/peps_cuda/julia_reference")
        addprocs(worker_count;
            exeflags="--project=$(project) --compiled-modules=no --threads=$(worker_threads)")
        for worker in workers()
            remotecall_fetch(worker) do
                Core.eval(Main, :(using Pkg))
                Base.invokelatest(getfield(Main, :Pkg).activate, project)
                Core.eval(Main, :(using LinearAlgebra))
                Core.eval(Main, :(using ITensors))
                Core.eval(Main, :(using QuantumNaturalGradient))
                Core.eval(Main, :(using QuantumNaturalfPEPS))
                Core.eval(Main, :(using TimerOutputs))
                qnpeps = getfield(Main, :QuantumNaturalfPEPS)
                itensors = getfield(Main, :ITensors)
                if !isdefined(qnpeps, Symbol("deprecate_make_inds_match!"))
                    Core.eval(qnpeps, quote
                        function deprecate_make_inds_match!(f, m1, m2, loginner;
                                                            make_inds_match=true)
                            return m1, m2
                        end
                    end)
                end
                if !isdefined(qnpeps, :check_hascommoninds)
                    Core.eval(qnpeps, quote
                        check_hascommoninds(args...) = nothing
                    end)
                end
                if !isdefined(itensors, Symbol("_log_or_not_dot"))
                    Core.eval(itensors, quote
                        function _log_or_not_dot(args...; kwargs...)
                            error("compatibility placeholder should be passed through only")
                        end
                    end)
                end
                getfield(Main, :LinearAlgebra).BLAS.set_num_threads(1)
                return true
            end
        end
    end

    factories = [
        make_heisenberg_example,
        make_csl_example,
        make_synthetic_real,
        make_synthetic_complex,
        make_synthetic_tfi,
    ]

    open(joinpath(outdir, "metadata.txt"), "w") do io
        println(io, "timestamp=", now())
        println(io, "julia_version=", VERSION)
        println(io, "threads=", Threads.nthreads())
        println(io, "blas_threads=", BLAS.get_num_threads())
        println(io, "sample_nr=", sample_nr)
        println(io, "maxiter=", maxiter)
        println(io, "threaded=", threaded)
        println(io, "multiproc=", multiproc)
        println(io, "workers=", workers())
        println(io, "worker_threads=", worker_threads)
        println(io, "compiled_modules_note=run with --compiled-modules=no for this reference harness")
    end

    for factory in factories
        result = run_profiled(factory, outdir;
            sample_nr, maxiter, threaded, multiproc, do_warmup)
        @info "profiled" name=result.case.name status=result.status elapsed=result.elapsed
        if result.status != "ok"
            @warn "scenario failed" name=result.case.name message=result.message
        end
        GC.gc()
    end
end

main()
