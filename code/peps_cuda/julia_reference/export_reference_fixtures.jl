using LinearAlgebra
using Printf
using Random

using ITensors
using QuantumNaturalfPEPS
using QuantumNaturalGradient

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

const get_logpsi_and_envs = getfield(QNPEPS, Symbol("get_logψ_and_envs"))
const logpsi_exact = getfield(QNPEPS, Symbol("logψ_exact"))
const get_all_horizontal_envs = getfield(QNPEPS, :get_all_horizontal_envs)
const get_Ek = getfield(QNPEPS, :get_Ek)
const get_Ok = getfield(QNPEPS, :get_Ok)

function json_string(x)
    s = string(x)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    s = replace(s, "\n" => "\\n")
    s = replace(s, "\t" => "\\t")
    return "\"" * s * "\""
end

complex_json(z) = @sprintf("[%0.17g,%0.17g]", real(z), imag(z))

function complex_array_json(values)
    return "[" * join(complex_json.(values), ",") * "]"
end

function array_json(values)
    return "[" * join(values, ",") * "]"
end

function nested_int_array_json(values)
    return "[" * join(("[" * join(v, ",") * "]" for v in values), ",") * "]"
end

function nested_string_array_json(values)
    return "[" * join(("[" * join(json_string.(v), ",") * "]" for v in values), ",") * "]"
end

function emit(io, fields)
    parts = String[]
    for (key, value) in fields
        push!(parts, json_string(key) * ":" * value)
    end
    println(io, "{" * join(parts, ",") * "}")
end

function heisenberg_opsum(lx, ly; j1=1.0)
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

function emit_logpsi_case(io; name, T, lx, ly, bond_dim, seed,
                          sample_values=nothing)
    Random.seed!(seed)
    hilbert = siteinds("S=1/2", lx, ly)
    peps = PEPS(T, hilbert;
        bond_dim=bond_dim,
        contract_dim=32,
        sample_dim=32,
        contract_cutoff=1e-12,
        sample_cutoff=1e-12,
        show_warning=false,
    )
    sample = zeros(Int64, lx, ly)
    if sample_values !== nothing
        @assert length(sample_values) == lx * ly
        for i in 1:lx, j in 1:ly
            sample[i, j] = sample_values[(i - 1) * ly + j]
        end
    end
    sample_row_major = [sample[i, j] for i in 1:lx for j in 1:ly]
    theta = vec(peps)
    native_axis_labels = [string.(inds(peps[i, j])) for i in 1:lx for j in 1:ly]
    theta_inds = [(siteind(peps, i, j), linkinds(peps, i, j)...) for i in 1:lx for j in 1:ly]
    site_dims = [dim.(inds(peps[i, j])) for i in 1:lx for j in 1:ly]
    theta_site_dims = [dim.(site_axes) for site_axes in theta_inds]
    theta_axis_labels = [string.(site_axes) for site_axes in theta_inds]
    exact = logpsi_exact(peps, sample)
    fields = [
        "kind" => json_string("logpsi"),
        "name" => json_string(name),
        "type" => json_string(string(T)),
        "lx" => string(lx),
        "ly" => string(ly),
        "bond_dim" => string(bond_dim),
        "seed" => string(seed),
        "site_dims" => nested_int_array_json(site_dims),
        "native_axis_labels" => nested_string_array_json(native_axis_labels),
        "theta_site_dims" => nested_int_array_json(theta_site_dims),
        "theta_axis_labels" => nested_string_array_json(theta_axis_labels),
        "theta_length" => string(length(theta)),
        "theta" => complex_array_json(theta),
        "sample" => array_json(string.(vec(sample))),
        "sample_row_major" => array_json(string.(sample_row_major)),
        "logpsi_exact" => complex_json(exact),
        "gc_live_bytes" => string(Base.gc_live_bytes()),
    ]

    try
        # The Julia reference has a default-position bug for two-row systems:
        # length(env_top) == 1 gives pos == 0. Use an explicit valid position.
        logpsi, env_top, env_down, max_bond =
            get_logpsi_and_envs(peps, sample; pos=1)
        push!(fields, "logpsi_env" => complex_json(logpsi))
        push!(fields, "env_top_maxlink" => string(maxlinkdim(env_top[end])))
        push!(fields, "env_down_maxlink" => string(maxlinkdim(env_down[1])))
        push!(fields, "max_bond" => string(max_bond))
        try
            ham_op = QuantumNaturalGradient.TensorOperatorSum(
                heisenberg_opsum(lx, ly; j1=1.0), hilbert)
            h_envs_r, h_envs_l =
                get_all_horizontal_envs(peps, env_top, env_down, sample)
            energy = get_Ek(peps, ham_op, env_top, env_down, sample, logpsi;
                            h_envs_r, h_envs_l)
            ok = get_Ok(peps, env_top, env_down, sample, logpsi;
                        h_envs_r, h_envs_l)
            push!(fields, "heisenberg_eloc" => complex_json(energy))
            push!(fields, "ok_length" => string(length(ok)))
            push!(fields, "ok_norm2" => @sprintf("%0.17g", sum(abs2, ok)))
            push!(fields,
                  "ok_first8" => complex_array_json(ok[1:min(8, length(ok))]))
        catch err
            push!(fields, "e_o_error" => json_string(sprint(showerror, err)))
        end
    catch err
        push!(fields, "logpsi_env_error" => json_string(sprint(showerror, err)))
    end

    emit(io, fields)
end

function main()
    outpath = length(ARGS) >= 1 ? ARGS[1] : "reference_fixtures.jsonl"
    mkpath(dirname(outpath))
    open(outpath, "w") do io
        emit(io, [
            "kind" => json_string("metadata"),
            "julia_version" => json_string(VERSION),
            "itensors_path" => json_string(pathof(ITensors)),
            "qnpeps_path" => json_string(pathof(QNPEPS)),
        ])
        emit_logpsi_case(io; name="real_3x2_D1_zero_sample", T=Float64,
                         lx=3, ly=2, bond_dim=1, seed=1)
        emit_logpsi_case(io; name="real_3x2_D2_zero_sample", T=Float64,
                         lx=3, ly=2, bond_dim=2, seed=2)
        emit_logpsi_case(io; name="real_3x2_D2_checker_sample", T=Float64,
                         lx=3, ly=2, bond_dim=2, seed=2,
                         sample_values=[0, 1, 1, 0, 0, 1])
        emit_logpsi_case(io; name="complex_3x2_D2_zero_sample", T=ComplexF64,
                         lx=3, ly=2, bond_dim=2, seed=3)
        emit_logpsi_case(io; name="complex_3x2_D2_checker_sample", T=ComplexF64,
                         lx=3, ly=2, bond_dim=2, seed=3,
                         sample_values=[0, 1, 1, 0, 0, 1])
        emit_logpsi_case(io; name="real_2x3_D2_striped_sample", T=Float64,
                         lx=2, ly=3, bond_dim=2, seed=4,
                         sample_values=[0, 1, 0, 1, 0, 1])
        emit_logpsi_case(io; name="complex_2x3_D2_striped_sample", T=ComplexF64,
                         lx=2, ly=3, bond_dim=2, seed=5,
                         sample_values=[0, 1, 0, 1, 0, 1])
        emit_logpsi_case(io; name="real_2x2_D3_checker_sample", T=Float64,
                         lx=2, ly=2, bond_dim=3, seed=6,
                         sample_values=[0, 1, 1, 0])
    end
end

main()
