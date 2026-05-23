#pragma once

#include <cstddef>

#ifdef __CUDACC__
#include <cuComplex.h>
#include <cuda_runtime.h>
#else
struct cuDoubleComplex {
    double x;
    double y;
};
using cudaStream_t = void *;
#endif

namespace peps_cuda {

void launch_abs2(const cuDoubleComplex *psi, double *weights, int count,
                 cudaStream_t stream);

void launch_project_physical_slices(const cuDoubleComplex *site_tensors,
                                    const int *sample_spins,
                                    cuDoubleComplex *projected_sites,
                                    int site_count, int local_dim,
                                    int slice_size, cudaStream_t stream);

void launch_project_physical_slices_batched(
    const cuDoubleComplex *site_tensors, const int *sample_spins,
    cuDoubleComplex *projected_sites, int sample_count, int site_count,
    int local_dim, int slice_size, cudaStream_t stream);

void launch_project_physical_slices_ragged(
    const cuDoubleComplex *site_tensors, const std::size_t *site_offsets,
    const std::size_t *slice_sizes, const int *sample_spins,
    cuDoubleComplex *projected_sites, const std::size_t *projected_offsets,
    std::size_t projected_sample_stride, int sample_count, int site_count,
    cudaStream_t stream);

void launch_importance_weights(const cuDoubleComplex *log_psi,
                               const double *log_sampling_probability,
                               double *weights, int count,
                               double log_normalization,
                               cudaStream_t stream);

void launch_diagonal_heisenberg_energy(const int *sample_spins,
                                       const int *bond_pairs,
                                       cuDoubleComplex *energy,
                                       int sample_count, int site_count,
                                       int bond_count, double coupling,
                                       cudaStream_t stream);

void launch_diagonal_two_site_energy(const int *sample_spins,
                                     const int *term_sites,
                                     const double *diagonal_values,
                                     cuDoubleComplex *energy, int sample_count,
                                     int site_count, int term_count,
                                     int local_dim, cudaStream_t stream);

void launch_diagonal_one_site_energy(const int *sample_spins,
                                     const int *term_sites,
                                     const double *diagonal_values,
                                     cuDoubleComplex *energy, int sample_count,
                                     int site_count, int term_count,
                                     int local_dim, cudaStream_t stream);

void launch_dense_minsr_gram(const cuDoubleComplex *sample_major_o,
                             cuDoubleComplex *gram, int sample_count,
                             int parameter_count, double diagonal_shift,
                             cudaStream_t stream);

void launch_dense_minsr_weighted_gram(const cuDoubleComplex *sample_major_o,
                                      const double *weights,
                                      cuDoubleComplex *gram, int sample_count,
                                      int parameter_count,
                                      double diagonal_shift,
                                      cudaStream_t stream);

void launch_sampled_sector_minsr_gram(
    const cuDoubleComplex *sample_major_sampled_o, const int *sample_spins,
    const std::size_t *sampled_site_offsets, const std::size_t *slice_sizes,
    cuDoubleComplex *gram, int sample_count, int site_count,
    int sampled_parameter_count, double diagonal_shift, cudaStream_t stream);

void launch_dense_minsr_apply_odag(const cuDoubleComplex *sample_major_o,
                                   const cuDoubleComplex *sample_vector,
                                   cuDoubleComplex *parameter_vector,
                                   int sample_count, int parameter_count,
                                   cudaStream_t stream);

void launch_dense_minsr_weighted_apply_odag(
    const cuDoubleComplex *sample_major_o, const double *weights,
    const cuDoubleComplex *sample_vector, cuDoubleComplex *parameter_vector,
    int sample_count, int parameter_count, cudaStream_t stream);

void launch_sampled_sector_minsr_apply_odag(
    const cuDoubleComplex *sample_major_sampled_o, const int *sample_spins,
    const std::size_t *sampled_site_offsets, const std::size_t *slice_sizes,
    const std::size_t *site_parameter_offsets,
    const std::size_t *physical_strides, const cuDoubleComplex *sample_vector,
    cuDoubleComplex *parameter_vector, int sample_count, int site_count,
    int sampled_parameter_count, cudaStream_t stream);

} // namespace peps_cuda
