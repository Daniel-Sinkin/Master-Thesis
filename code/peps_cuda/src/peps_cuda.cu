#include "peps_cuda/cuda_kernels.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace peps_cuda {
namespace {

__device__ cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y,
                                a.x * b.y + a.y * b.x);
}

__device__ cuDoubleComplex conjv(cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x, -a.y);
}

__device__ cuDoubleComplex neg(cuDoubleComplex a) {
    return make_cuDoubleComplex(-a.x, -a.y);
}

__device__ cuDoubleComplex block_reduce_complex(cuDoubleComplex value) {
    __shared__ double real[256];
    __shared__ double imag[256];
    const int lane = threadIdx.x;
    real[lane] = value.x;
    imag[lane] = value.y;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            real[lane] += real[lane + stride];
            imag[lane] += imag[lane + stride];
        }
        __syncthreads();
    }
    return make_cuDoubleComplex(real[0], imag[0]);
}

} // namespace

__global__ void peps_abs2_kernel(const cuDoubleComplex *psi, double *weights,
                                 int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const cuDoubleComplex z = psi[idx];
    weights[idx] = z.x * z.x + z.y * z.y;
}

__global__ void project_physical_slices_kernel(
    const cuDoubleComplex *site_tensors, const int *sample_spins,
    cuDoubleComplex *projected_sites, int site_count, int local_dim,
    int slice_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = site_count * slice_size;
    if (idx >= total) {
        return;
    }
    const int site = idx / slice_size;
    const int within = idx - site * slice_size;
    const int spin = sample_spins[site];
    projected_sites[idx] =
        site_tensors[(site * local_dim + spin) * slice_size + within];
}

__global__ void project_physical_slices_batched_kernel(
    const cuDoubleComplex *site_tensors, const int *sample_spins,
    cuDoubleComplex *projected_sites, int sample_count, int site_count,
    int local_dim, int slice_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = sample_count * site_count * slice_size;
    if (idx >= total) {
        return;
    }
    const int within = idx % slice_size;
    const int site = (idx / slice_size) % site_count;
    const int sample = idx / (slice_size * site_count);
    const int spin = sample_spins[sample * site_count + site];
    projected_sites[idx] =
        site_tensors[(site * local_dim + spin) * slice_size + within];
}

__global__ void project_physical_slices_ragged_kernel(
    const cuDoubleComplex *site_tensors, const std::size_t *site_offsets,
    const std::size_t *slice_sizes, const int *sample_spins,
    cuDoubleComplex *projected_sites, const std::size_t *projected_offsets,
    std::size_t projected_sample_stride, int sample_count, int site_count) {
    const int site = blockIdx.x;
    const int sample = blockIdx.y;
    if (site >= site_count || sample >= sample_count) {
        return;
    }
    const std::size_t slice_size = slice_sizes[site];
    const int spin = sample_spins[sample * site_count + site];
    const std::size_t in_base = site_offsets[site] +
                                static_cast<std::size_t>(spin) * slice_size;
    const std::size_t out_base =
        static_cast<std::size_t>(sample) * projected_sample_stride +
        projected_offsets[site];
    for (std::size_t within = threadIdx.x; within < slice_size;
         within += blockDim.x) {
        projected_sites[out_base + within] = site_tensors[in_base + within];
    }
}

__global__ void accumulate_importance_weights_kernel(
    const cuDoubleComplex *log_psi, const double *log_sampling_probability,
    double *weights, int count, double log_normalization) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    weights[idx] =
        exp(2.0 * log_psi[idx].x - log_sampling_probability[idx] -
            log_normalization);
}

__global__ void diagonal_heisenberg_energy_kernel(
    const int *sample_spins, const int *bond_pairs, cuDoubleComplex *energy,
    int sample_count, int site_count, int bond_count, double coupling) {
    const int sample = blockIdx.x;
    if (sample >= sample_count) {
        return;
    }

    double partial = 0.0;
    for (int bond = threadIdx.x; bond < bond_count; bond += blockDim.x) {
        const int a = bond_pairs[2 * bond];
        const int b = bond_pairs[2 * bond + 1];
        const int sa = sample_spins[sample * site_count + a];
        const int sb = sample_spins[sample * site_count + b];
        const double za = (sa == 0) ? 0.5 : -0.5;
        const double zb = (sb == 0) ? 0.5 : -0.5;
        partial += coupling * za * zb;
    }

    const cuDoubleComplex reduced = block_reduce_complex(
        make_cuDoubleComplex(partial, 0.0));
    if (threadIdx.x == 0) {
        energy[sample] = add(energy[sample], reduced);
    }
}

__global__ void diagonal_two_site_energy_kernel(
    const int *sample_spins, const int *term_sites,
    const double *diagonal_values, cuDoubleComplex *energy, int sample_count,
    int site_count, int term_count, int local_dim) {
    const int sample = blockIdx.x;
    if (sample >= sample_count) {
        return;
    }

    double partial = 0.0;
    for (int term = threadIdx.x; term < term_count; term += blockDim.x) {
        const int a = term_sites[2 * term];
        const int b = term_sites[2 * term + 1];
        const int sa = sample_spins[sample * site_count + a];
        const int sb = sample_spins[sample * site_count + b];
        const int local_code = sa * local_dim + sb;
        partial += diagonal_values[term * local_dim * local_dim + local_code];
    }

    const cuDoubleComplex reduced =
        block_reduce_complex(make_cuDoubleComplex(partial, 0.0));
    if (threadIdx.x == 0) {
        energy[sample] = add(energy[sample], reduced);
    }
}

__global__ void diagonal_one_site_energy_kernel(
    const int *sample_spins, const int *term_sites,
    const double *diagonal_values, cuDoubleComplex *energy, int sample_count,
    int site_count, int term_count, int local_dim) {
    const int sample = blockIdx.x;
    if (sample >= sample_count) {
        return;
    }

    double partial = 0.0;
    for (int term = threadIdx.x; term < term_count; term += blockDim.x) {
        const int site = term_sites[term];
        const int spin = sample_spins[sample * site_count + site];
        partial += diagonal_values[term * local_dim + spin];
    }

    const cuDoubleComplex reduced =
        block_reduce_complex(make_cuDoubleComplex(partial, 0.0));
    if (threadIdx.x == 0) {
        energy[sample] = add(energy[sample], reduced);
    }
}

__global__ void dense_minsr_gram_kernel(const cuDoubleComplex *sample_major_o,
                                        cuDoubleComplex *gram,
                                        int sample_count, int parameter_count,
                                        double diagonal_shift) {
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);
    for (int p = threadIdx.x; p < parameter_count; p += blockDim.x) {
        const cuDoubleComplex a = sample_major_o[row * parameter_count + p];
        const cuDoubleComplex b = sample_major_o[col * parameter_count + p];
        partial = add(partial, mul(a, conjv(b)));
    }

    cuDoubleComplex reduced = block_reduce_complex(partial);
    if (threadIdx.x == 0) {
        if (row == col) {
            reduced.x += diagonal_shift;
        }
        gram[row * sample_count + col] = reduced;
    }
}

__global__ void dense_minsr_weighted_gram_kernel(
    const cuDoubleComplex *sample_major_o, const double *weights,
    cuDoubleComplex *gram, int sample_count, int parameter_count,
    double diagonal_shift) {
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    const double scale = sqrt(weights[row] * weights[col]);
    cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);
    for (int p = threadIdx.x; p < parameter_count; p += blockDim.x) {
        const cuDoubleComplex a = sample_major_o[row * parameter_count + p];
        const cuDoubleComplex b = sample_major_o[col * parameter_count + p];
        cuDoubleComplex contribution = mul(a, conjv(b));
        contribution.x *= scale;
        contribution.y *= scale;
        partial = add(partial, contribution);
    }

    cuDoubleComplex reduced = block_reduce_complex(partial);
    if (threadIdx.x == 0) {
        if (row == col) {
            reduced.x += diagonal_shift;
        }
        gram[row * sample_count + col] = reduced;
    }
}

__global__ void sampled_sector_minsr_gram_kernel(
    const cuDoubleComplex *sample_major_sampled_o, const int *sample_spins,
    const std::size_t *sampled_site_offsets, const std::size_t *slice_sizes,
    cuDoubleComplex *gram, int sample_count, int site_count,
    int sampled_parameter_count, double diagonal_shift) {
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);

    for (int site = 0; site < site_count; ++site) {
        const int row_spin = sample_spins[row * site_count + site];
        const int col_spin = sample_spins[col * site_count + site];
        if (row_spin != col_spin) {
            continue;
        }
        const std::size_t offset = sampled_site_offsets[site];
        const std::size_t slice_size = slice_sizes[site];
        const std::size_t row_base =
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(sampled_parameter_count) +
            offset;
        const std::size_t col_base =
            static_cast<std::size_t>(col) *
                static_cast<std::size_t>(sampled_parameter_count) +
            offset;
        for (std::size_t within = threadIdx.x; within < slice_size;
             within += blockDim.x) {
            const cuDoubleComplex a = sample_major_sampled_o[row_base + within];
            const cuDoubleComplex b = sample_major_sampled_o[col_base + within];
            partial = add(partial, mul(a, conjv(b)));
        }
    }

    cuDoubleComplex reduced = block_reduce_complex(partial);
    if (threadIdx.x == 0) {
        if (row == col) {
            reduced.x += diagonal_shift;
        }
        gram[row * sample_count + col] = reduced;
    }
}

__global__ void dense_minsr_apply_odag_kernel(
    const cuDoubleComplex *sample_major_o, const cuDoubleComplex *sample_vector,
    cuDoubleComplex *parameter_vector, int sample_count, int parameter_count) {
    const int parameter = blockIdx.x;
    cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);
    for (int sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        const cuDoubleComplex o =
            sample_major_o[sample * parameter_count + parameter];
        const cuDoubleComplex x = sample_vector[sample];
        partial = add(partial, mul(conjv(o), x));
    }
    const cuDoubleComplex reduced = block_reduce_complex(partial);
    if (threadIdx.x == 0) {
        parameter_vector[parameter] = neg(reduced);
    }
}

__global__ void dense_minsr_weighted_apply_odag_kernel(
    const cuDoubleComplex *sample_major_o, const double *weights,
    const cuDoubleComplex *sample_vector, cuDoubleComplex *parameter_vector,
    int sample_count, int parameter_count) {
    const int parameter = blockIdx.x;
    cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);
    for (int sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        const cuDoubleComplex o =
            sample_major_o[sample * parameter_count + parameter];
        const cuDoubleComplex x = sample_vector[sample];
        cuDoubleComplex contribution = mul(conjv(o), x);
        const double scale = sqrt(weights[sample]);
        contribution.x *= scale;
        contribution.y *= scale;
        partial = add(partial, contribution);
    }
    const cuDoubleComplex reduced = block_reduce_complex(partial);
    if (threadIdx.x == 0) {
        parameter_vector[parameter] = neg(reduced);
    }
}

__global__ void sampled_sector_minsr_apply_odag_kernel(
    const cuDoubleComplex *sample_major_sampled_o, const int *sample_spins,
    const std::size_t *sampled_site_offsets, const std::size_t *slice_sizes,
    const std::size_t *site_parameter_offsets,
    const std::size_t *physical_strides, const cuDoubleComplex *sample_vector,
    cuDoubleComplex *parameter_vector, int sample_count, int site_count,
    int sampled_parameter_count) {
    const int site = blockIdx.x;
    const int sample = blockIdx.y;
    if (site >= site_count || sample >= sample_count) {
        return;
    }

    const int spin = sample_spins[sample * site_count + site];
    const std::size_t slice_size = slice_sizes[site];
    const std::size_t src =
        static_cast<std::size_t>(sample) *
            static_cast<std::size_t>(sampled_parameter_count) +
        sampled_site_offsets[site];
    const std::size_t dst =
        site_parameter_offsets[site] +
        static_cast<std::size_t>(spin) * physical_strides[site];
    const cuDoubleComplex x = sample_vector[sample];

    for (std::size_t within = threadIdx.x; within < slice_size;
         within += blockDim.x) {
        const cuDoubleComplex o = sample_major_sampled_o[src + within];
        const cuDoubleComplex contribution = neg(mul(conjv(o), x));
        atomicAdd(&parameter_vector[dst + within].x, contribution.x);
        atomicAdd(&parameter_vector[dst + within].y, contribution.y);
    }
}

void launch_abs2(const cuDoubleComplex *psi, double *weights, int count,
                 cudaStream_t stream) {
    constexpr int block = 256;
    const int grid = (count + block - 1) / block;
    peps_abs2_kernel<<<grid, block, 0, stream>>>(psi, weights, count);
}

void launch_project_physical_slices(const cuDoubleComplex *site_tensors,
                                    const int *sample_spins,
                                    cuDoubleComplex *projected_sites,
                                    int site_count, int local_dim,
                                    int slice_size, cudaStream_t stream) {
    constexpr int block = 256;
    const int total = site_count * slice_size;
    const int grid = (total + block - 1) / block;
    project_physical_slices_kernel<<<grid, block, 0, stream>>>(
        site_tensors, sample_spins, projected_sites, site_count, local_dim,
        slice_size);
}

void launch_project_physical_slices_batched(
    const cuDoubleComplex *site_tensors, const int *sample_spins,
    cuDoubleComplex *projected_sites, int sample_count, int site_count,
    int local_dim, int slice_size, cudaStream_t stream) {
    constexpr int block = 256;
    const int total = sample_count * site_count * slice_size;
    const int grid = (total + block - 1) / block;
    project_physical_slices_batched_kernel<<<grid, block, 0, stream>>>(
        site_tensors, sample_spins, projected_sites, sample_count, site_count,
        local_dim, slice_size);
}

void launch_project_physical_slices_ragged(
    const cuDoubleComplex *site_tensors, const std::size_t *site_offsets,
    const std::size_t *slice_sizes, const int *sample_spins,
    cuDoubleComplex *projected_sites, const std::size_t *projected_offsets,
    std::size_t projected_sample_stride, int sample_count, int site_count,
    cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid(site_count, sample_count);
    project_physical_slices_ragged_kernel<<<grid, block, 0, stream>>>(
        site_tensors, site_offsets, slice_sizes, sample_spins, projected_sites,
        projected_offsets, projected_sample_stride, sample_count, site_count);
}

void launch_importance_weights(const cuDoubleComplex *log_psi,
                               const double *log_sampling_probability,
                               double *weights, int count,
                               double log_normalization,
                               cudaStream_t stream) {
    constexpr int block = 256;
    const int grid = (count + block - 1) / block;
    accumulate_importance_weights_kernel<<<grid, block, 0, stream>>>(
        log_psi, log_sampling_probability, weights, count, log_normalization);
}

void launch_diagonal_heisenberg_energy(const int *sample_spins,
                                       const int *bond_pairs,
                                       cuDoubleComplex *energy,
                                       int sample_count, int site_count,
                                       int bond_count, double coupling,
                                       cudaStream_t stream) {
    constexpr int block = 256;
    diagonal_heisenberg_energy_kernel<<<sample_count, block, 0, stream>>>(
        sample_spins, bond_pairs, energy, sample_count, site_count, bond_count,
        coupling);
}

void launch_diagonal_two_site_energy(const int *sample_spins,
                                     const int *term_sites,
                                     const double *diagonal_values,
                                     cuDoubleComplex *energy, int sample_count,
                                     int site_count, int term_count,
                                     int local_dim, cudaStream_t stream) {
    constexpr int block = 256;
    diagonal_two_site_energy_kernel<<<sample_count, block, 0, stream>>>(
        sample_spins, term_sites, diagonal_values, energy, sample_count,
        site_count, term_count, local_dim);
}

void launch_diagonal_one_site_energy(const int *sample_spins,
                                     const int *term_sites,
                                     const double *diagonal_values,
                                     cuDoubleComplex *energy, int sample_count,
                                     int site_count, int term_count,
                                     int local_dim, cudaStream_t stream) {
    constexpr int block = 256;
    diagonal_one_site_energy_kernel<<<sample_count, block, 0, stream>>>(
        sample_spins, term_sites, diagonal_values, energy, sample_count,
        site_count, term_count, local_dim);
}

void launch_dense_minsr_gram(const cuDoubleComplex *sample_major_o,
                             cuDoubleComplex *gram, int sample_count,
                             int parameter_count, double diagonal_shift,
                             cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid(sample_count, sample_count);
    dense_minsr_gram_kernel<<<grid, block, 0, stream>>>(
        sample_major_o, gram, sample_count, parameter_count, diagonal_shift);
}

void launch_dense_minsr_weighted_gram(const cuDoubleComplex *sample_major_o,
                                      const double *weights,
                                      cuDoubleComplex *gram, int sample_count,
                                      int parameter_count,
                                      double diagonal_shift,
                                      cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid(sample_count, sample_count);
    dense_minsr_weighted_gram_kernel<<<grid, block, 0, stream>>>(
        sample_major_o, weights, gram, sample_count, parameter_count,
        diagonal_shift);
}

void launch_sampled_sector_minsr_gram(
    const cuDoubleComplex *sample_major_sampled_o, const int *sample_spins,
    const std::size_t *sampled_site_offsets, const std::size_t *slice_sizes,
    cuDoubleComplex *gram, int sample_count, int site_count,
    int sampled_parameter_count, double diagonal_shift, cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid(sample_count, sample_count);
    sampled_sector_minsr_gram_kernel<<<grid, block, 0, stream>>>(
        sample_major_sampled_o, sample_spins, sampled_site_offsets, slice_sizes,
        gram, sample_count, site_count, sampled_parameter_count, diagonal_shift);
}

void launch_dense_minsr_apply_odag(const cuDoubleComplex *sample_major_o,
                                   const cuDoubleComplex *sample_vector,
                                   cuDoubleComplex *parameter_vector,
                                   int sample_count, int parameter_count,
                                   cudaStream_t stream) {
    constexpr int block = 256;
    dense_minsr_apply_odag_kernel<<<parameter_count, block, 0, stream>>>(
        sample_major_o, sample_vector, parameter_vector, sample_count,
        parameter_count);
}

void launch_dense_minsr_weighted_apply_odag(
    const cuDoubleComplex *sample_major_o, const double *weights,
    const cuDoubleComplex *sample_vector, cuDoubleComplex *parameter_vector,
    int sample_count, int parameter_count, cudaStream_t stream) {
    constexpr int block = 256;
    dense_minsr_weighted_apply_odag_kernel<<<parameter_count, block, 0, stream>>>(
        sample_major_o, weights, sample_vector, parameter_vector, sample_count,
        parameter_count);
}

void launch_sampled_sector_minsr_apply_odag(
    const cuDoubleComplex *sample_major_sampled_o, const int *sample_spins,
    const std::size_t *sampled_site_offsets, const std::size_t *slice_sizes,
    const std::size_t *site_parameter_offsets,
    const std::size_t *physical_strides, const cuDoubleComplex *sample_vector,
    cuDoubleComplex *parameter_vector, int sample_count, int site_count,
    int sampled_parameter_count, cudaStream_t stream) {
    constexpr int block = 256;
    const dim3 grid(site_count, sample_count);
    sampled_sector_minsr_apply_odag_kernel<<<grid, block, 0, stream>>>(
        sample_major_sampled_o, sample_spins, sampled_site_offsets, slice_sizes,
        site_parameter_offsets, physical_strides, sample_vector,
        parameter_vector, sample_count, site_count, sampled_parameter_count);
}

} // namespace peps_cuda
