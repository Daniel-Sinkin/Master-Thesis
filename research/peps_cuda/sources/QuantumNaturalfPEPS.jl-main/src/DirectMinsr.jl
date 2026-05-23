struct CompactSiteBlock
    i::Int
    j::Int
    dense_offset::Int
    compact_offset::Int
    slice_size::Int
    loc_dim::Int
end

function compact_parameter_layout(peps::AbstractPEPS; mask=peps.mask)
    blocks = CompactSiteBlock[]
    dense_pos = 1
    compact_pos = 1

    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if mask[i, j] != 0
            full_size = prod(dim.(inds(peps[i, j])))
            loc_dim = dim(siteind(peps, i, j))
            @assert full_size % loc_dim == 0
            slice_size = div(full_size, loc_dim)
            push!(blocks, CompactSiteBlock(i, j, dense_pos, compact_pos, slice_size, loc_dim))
            dense_pos += full_size
            compact_pos += slice_size
        end
    end

    return blocks
end

compact_parameter_count(layout::Vector{CompactSiteBlock}) =
    isempty(layout) ? 0 : last(layout).compact_offset + last(layout).slice_size - 1

function get_compact_Ok(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment},
                        S::Matrix{Int64}, logpsi::Number;
                        h_envs_r=nothing, h_envs_l=nothing, Ok=nothing,
                        layout=compact_parameter_layout(peps))
    if Ok === nothing
        Ok = Vector{eltype(peps)}(undef, compact_parameter_count(layout))
    end
    if h_envs_r === nothing || h_envs_l === nothing
        h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, S)
    end

    for block in layout
        ok_tensor = get_Ok(peps, env_top, env_down, logpsi, h_envs_r, h_envs_l, block.i, block.j)
        dst = @view Ok[block.compact_offset:block.compact_offset + block.slice_size - 1]
        permute_reshape_and_copy!(dst, ok_tensor, linkinds(peps, block.i, block.j))
    end

    return Ok
end

function _add_scaled_compact_to_dense!(dense::AbstractVector, row::AbstractVector, S::Matrix{Int},
                                       alpha, layout::Vector{CompactSiteBlock})
    for block in layout
        spin = S[block.i, block.j]
        src0 = block.compact_offset
        dst0 = block.dense_offset + spin
        @inbounds for k in 0:block.slice_size - 1
            dense[dst0 + block.loc_dim * k] += alpha * row[src0 + k]
        end
    end
    return dense
end

function _add_scaled_conj_compact_to_dense!(dense::AbstractVector, row::AbstractVector, S::Matrix{Int},
                                            alpha, layout::Vector{CompactSiteBlock})
    for block in layout
        spin = S[block.i, block.j]
        src0 = block.compact_offset
        dst0 = block.dense_offset + spin
        @inbounds for k in 0:block.slice_size - 1
            dense[dst0 + block.loc_dim * k] += alpha * conj(row[src0 + k])
        end
    end
    return dense
end

function scatter_compact_to_dense!(dense::AbstractVector, row::AbstractVector, S::Matrix{Int},
                                   layout::Vector{CompactSiteBlock}; alpha=1)
    return _add_scaled_compact_to_dense!(dense, row, S, alpha, layout)
end

function _sparse_dense_dot(row::AbstractVector, S::Matrix{Int}, dense::AbstractVector,
                           layout::Vector{CompactSiteBlock})
    out = zero(promote_type(eltype(row), eltype(dense)))
    for block in layout
        spin = S[block.i, block.j]
        src0 = block.compact_offset
        dst0 = block.dense_offset + spin
        @inbounds for k in 0:block.slice_size - 1
            out += row[src0 + k] * conj(dense[dst0 + block.loc_dim * k])
        end
    end
    return out
end

function _compact_weighted_mean_dense(rows::Vector, samples::Vector, weights, nparams::Int,
                                      layout::Vector{CompactSiteBlock})
    mean_dense = zeros(eltype(rows[1]), nparams)
    if weights === nothing
        alpha = inv(length(rows))
        for s in eachindex(rows)
            _add_scaled_compact_to_dense!(mean_dense, rows[s], samples[s], alpha, layout)
        end
    else
        norm = sum(weights)
        for s in eachindex(rows)
            _add_scaled_compact_to_dense!(mean_dense, rows[s], samples[s], weights[s] / norm, layout)
        end
    end
    return mean_dense
end

function _add_sparse_sector_gram!(Tmat::AbstractMatrix, rows::Vector, samples::Vector,
                                  layout::Vector{CompactSiteBlock})
    nsamples = length(rows)
    for block in layout
        src_range = block.compact_offset:block.compact_offset + block.slice_size - 1
        for spin in 0:block.loc_dim - 1
            idx = Int[]
            sizehint!(idx, div(nsamples, block.loc_dim) + 1)
            for s in 1:nsamples
                if samples[s][block.i, block.j] == spin
                    push!(idx, s)
                end
            end

            if isempty(idx)
                continue
            end

            M = Matrix{eltype(Tmat)}(undef, length(idx), block.slice_size)
            for (r, s) in enumerate(idx)
                copyto!(@view(M[r, :]), @view(rows[s][src_range]))
            end
            G = M * M'
            for b in 1:length(idx), a in 1:length(idx)
                Tmat[idx[a], idx[b]] += G[a, b]
            end
        end
    end
    return Tmat
end

function _direct_centered_T(rows::Vector, samples::Vector, mean_dense::AbstractVector,
                            weights, layout::Vector{CompactSiteBlock})
    nsamples = length(rows)
    value_type = promote_type(eltype(rows[1]), eltype(mean_dense))
    Tmat = Matrix{value_type}(undef, nsamples, nsamples)
    sparse_mu = Vector{value_type}(undef, nsamples)
    mu_norm = sum(abs2, mean_dense)

    for s in 1:nsamples
        sparse_mu[s] = _sparse_dense_dot(rows[s], samples[s], mean_dense, layout)
    end

    for b in 1:nsamples, a in 1:nsamples
        Tmat[a, b] = mu_norm - sparse_mu[a] - conj(sparse_mu[b])
    end
    _add_sparse_sector_gram!(Tmat, rows, samples, layout)

    if weights !== nothing
        sqrtw = sqrt.(weights)
        for b in 1:nsamples, a in 1:nsamples
            Tmat[a, b] *= sqrtw[a] * sqrtw[b]
        end
    end

    return Tmat
end

function _center_energy(Eks::Vector, weights)
    if weights === nothing
        mean_energy = mean(Eks)
        centered = Eks .- mean_energy
    else
        mean_energy = sum(weights .* Eks) / sum(weights)
        centered = (Eks .- mean_energy) .* sqrt.(weights)
    end

    if eltype(Eks) <: Complex && !any(abs.(imag.(centered)) .> 1e-10)
        return real.(centered), mean_energy
    end
    return centered, mean_energy
end

function _direct_theta_dot(rows::Vector, samples::Vector, mean_dense::AbstractVector, weights,
                           raw::AbstractVector, layout::Vector{CompactSiteBlock})
    theta_dot = zeros(promote_type(eltype(mean_dense), eltype(raw)), length(mean_dense))

    if weights === nothing
        mean_scale = sum(raw)
        for s in eachindex(rows)
            _add_scaled_conj_compact_to_dense!(theta_dot, rows[s], samples[s], raw[s], layout)
        end
    else
        sqrtw = sqrt.(weights)
        mean_scale = sum(sqrtw .* raw)
        for s in eachindex(rows)
            _add_scaled_conj_compact_to_dense!(theta_dot, rows[s], samples[s], sqrtw[s] * raw[s], layout)
        end
    end

    @inbounds for p in eachindex(theta_dot)
        theta_dot[p] -= mean_scale * conj(mean_dense[p])
    end

    return theta_dot
end

function Ok_and_Ek_compact(peps::AbstractPEPS, ham_op; timer=TimerOutput(), Ok=nothing,
                           layout=compact_parameter_layout(peps), sampling_mode=:full,
                           resample=false, correct_sampling_error=true, resample_energy=0)
    S, logpc, env_top = @timeit timer "sampling" get_sample(peps; mode=sampling_mode, timer)

    if resample
        S = QuantumNaturalGradient.resample_with_H(S, ham_op; resample_energy)
    end

    overwrite = !(sampling_mode == :full)

    logpsi, env_top, env_down, max_bond =
        @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top; overwrite)
    h_envs_r, h_envs_l =
        @timeit timer "horizontal_envs" get_all_horizontal_envs(peps, env_top, env_down, S)

    logpsi_flipped = Dict{Any, Number}()
    Ek_terms =
        @timeit timer "precomp_sHψ_elems" QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
    E_loc =
        @timeit timer "energy" get_Ek(peps, ham_op, env_top, env_down, S, logpsi;
                                      h_envs_r, h_envs_l, logψ_flipped=logpsi_flipped,
                                      Ek_terms, timer)
    grad =
        @timeit timer "compact_log_gradients" get_compact_Ok(peps, env_top, env_down, S, logpsi;
                                                             h_envs_r, h_envs_l, Ok, layout)

    if resample
        @assert !correct_sampling_error "Correcting the sampling error with resampling is not implemented"
        logpc = QuantumNaturalGradient.get_logprob_resample(S, Ek_terms, logpsi_flipped, ham_op; resample_energy)
    end

    if !correct_sampling_error
        logpc = 2 * real(logpsi)
    end

    return grad, E_loc, logpsi, S, logpc, max_bond
end

function direct_gram_minsr_singlethread(peps::AbstractPEPS, ham::OpSum, sample_nr::Integer; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return direct_gram_minsr_singlethread(peps, ham_op, sample_nr; kwargs...)
end

function direct_gram_minsr_singlethread(peps::AbstractPEPS, ham_op::TensorOperatorSum, sample_nr::Integer;
                                        solver=QuantumNaturalGradient.EigenSolver(),
                                        timer=TimerOutput(),
                                        double_layer_update=update_double_layer_envs!,
                                        update_double_layer=true,
                                        layout=compact_parameter_layout(peps),
                                        importance_weights=true,
                                        return_rows=false,
                                        kwargs...)
    eltype_peps = eltype(peps)
    eltype_real = real(eltype_peps)
    compact_count = compact_parameter_count(layout)

    if update_double_layer
        @timeit timer "double_layer_envs" double_layer_update(peps)
    end

    rows = Vector{Vector{eltype_peps}}(undef, sample_nr)
    Eks = Vector{eltype_peps}(undef, sample_nr)
    logpsis = Vector{Complex{eltype_real}}(undef, sample_nr)
    samples = Vector{Matrix{Int}}(undef, sample_nr)
    logpc = Vector{eltype_real}(undef, sample_nr)
    contract_dims = Vector{Int}(undef, sample_nr)

    @timeit timer "compact_Oks_and_Eks" begin
        for s in 1:sample_nr
            row = Vector{eltype_peps}(undef, compact_count)
            rows[s], Eks[s], logpsis[s], samples[s], logpc[s], contract_dims[s] =
                Ok_and_Ek_compact(peps, ham_op; timer, Ok=row, layout, kwargs...)
        end
    end

    raw_weights = compute_importance_weights(logpsis, logpc)
    weights = importance_weights ? copy(raw_weights) : nothing
    if weights !== nothing
        weights ./= mean(weights)
    end

    mean_dense =
        @timeit timer "direct_mean" _compact_weighted_mean_dense(rows, samples, weights, length(peps), layout)
    Tmat =
        @timeit timer "direct_T" _direct_centered_T(rows, samples, mean_dense, weights, layout)
    energy_centered, energy_mean = _center_energy(Eks, weights)
    raw =
        @timeit timer "direct_solve" -solver(Tmat, energy_centered)
    theta_dot =
        @timeit timer "direct_theta_dot" _direct_theta_dot(rows, samples, mean_dense, weights, raw, layout)

    out = Dict(
        :theta_dot => theta_dot,
        :T => Tmat,
        :Eks => Eks,
        :energy_mean => energy_mean,
        :logψs => logpsis,
        :samples => samples,
        :weights => raw_weights,
        :normalised_weights => weights,
        :contract_dims => contract_dims,
        :compact_parameter_count => compact_count,
        :dense_parameter_count => length(peps),
    )
    if return_rows
        out[:compact_rows] = rows
        out[:compact_mean_dense] = mean_dense
    end
    return out
end
