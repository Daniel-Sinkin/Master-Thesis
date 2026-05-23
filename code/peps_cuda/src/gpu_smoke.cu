#include "peps_cuda/cuda_kernels.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(err));
    }
}

void check_close(double value, double expected, const char *what) {
    if (std::abs(value - expected) > 1.0e-10) {
        throw std::runtime_error(std::string(what) + " expected " +
                                 std::to_string(expected) + " got " +
                                 std::to_string(value));
    }
}

template <class T>
T *copy_to_device(const std::vector<T> &host) {
    T *device = nullptr;
    check(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
    return device;
}

template <class T>
std::vector<T> copy_to_host(const T *device, std::size_t n) {
    std::vector<T> host(n);
    check(cudaMemcpy(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
    return host;
}

} // namespace

int main() {
    try {
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "cudaStreamCreate");

        const std::vector<cuDoubleComplex> psi = {
            make_cuDoubleComplex(3.0, 4.0), make_cuDoubleComplex(1.0, -2.0)};
        cuDoubleComplex *d_psi = copy_to_device(psi);
        double *d_weights = nullptr;
        check(cudaMalloc(&d_weights, psi.size() * sizeof(double)), "cudaMalloc");
        peps_cuda::launch_abs2(d_psi, d_weights, static_cast<int>(psi.size()),
                               stream);
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize abs2");
        const std::vector<double> weights = copy_to_host(d_weights, psi.size());
        check_close(weights[0], 25.0, "abs2[0]");
        check_close(weights[1], 5.0, "abs2[1]");

        const int sample_count = 2;
        const int site_count = 2;
        const int local_dim = 2;
        const int slice_size = 2;
        const std::vector<cuDoubleComplex> site_tensors = {
            make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(2.0, 0.0),
            make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(4.0, 0.0),
            make_cuDoubleComplex(5.0, 0.0), make_cuDoubleComplex(6.0, 0.0),
            make_cuDoubleComplex(7.0, 0.0), make_cuDoubleComplex(8.0, 0.0)};
        const std::vector<int> sample_spins = {0, 1, 1, 0};
        cuDoubleComplex *d_sites = copy_to_device(site_tensors);
        int *d_spins = copy_to_device(sample_spins);
        cuDoubleComplex *d_projected = nullptr;
        check(cudaMalloc(&d_projected,
                         sample_count * site_count * slice_size *
                             sizeof(cuDoubleComplex)),
              "cudaMalloc projected");
        peps_cuda::launch_project_physical_slices_batched(
            d_sites, d_spins, d_projected, sample_count, site_count, local_dim,
            slice_size, stream);
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize project");
        const std::vector<cuDoubleComplex> projected =
            copy_to_host(d_projected, sample_count * site_count * slice_size);
        check_close(projected[0].x, 1.0, "projected[0]");
        check_close(projected[2].x, 7.0, "projected[2]");
        check_close(projected[4].x, 3.0, "projected[4]");
        check_close(projected[6].x, 5.0, "projected[6]");

        const std::vector<cuDoubleComplex> ragged_sites = {
            make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(2.0, 0.0),
            make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(4.0, 0.0),
            make_cuDoubleComplex(5.0, 0.0), make_cuDoubleComplex(6.0, 0.0)};
        const std::vector<std::size_t> ragged_offsets = {0, 4};
        const std::vector<std::size_t> ragged_slices = {2, 1};
        const std::vector<std::size_t> ragged_projected_offsets = {0, 2};
        cuDoubleComplex *d_ragged_sites = copy_to_device(ragged_sites);
        std::size_t *d_ragged_offsets = copy_to_device(ragged_offsets);
        std::size_t *d_ragged_slices = copy_to_device(ragged_slices);
        std::size_t *d_ragged_projected_offsets =
            copy_to_device(ragged_projected_offsets);
        cuDoubleComplex *d_ragged_projected = nullptr;
        check(cudaMalloc(&d_ragged_projected,
                         sample_count * 3 * sizeof(cuDoubleComplex)),
              "cudaMalloc ragged projected");
        peps_cuda::launch_project_physical_slices_ragged(
            d_ragged_sites, d_ragged_offsets, d_ragged_slices, d_spins,
            d_ragged_projected, d_ragged_projected_offsets, 3, sample_count,
            site_count, stream);
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize ragged project");
        const std::vector<cuDoubleComplex> ragged_projected =
            copy_to_host(d_ragged_projected, sample_count * 3);
        check_close(ragged_projected[0].x, 1.0, "ragged projected[0]");
        check_close(ragged_projected[1].x, 2.0, "ragged projected[1]");
        check_close(ragged_projected[2].x, 6.0, "ragged projected[2]");
        check_close(ragged_projected[3].x, 3.0, "ragged projected[3]");
        check_close(ragged_projected[4].x, 4.0, "ragged projected[4]");
        check_close(ragged_projected[5].x, 5.0, "ragged projected[5]");

        const std::vector<int> pairs = {0, 1};
        int *d_pairs = copy_to_device(pairs);
        cuDoubleComplex *d_energy = nullptr;
        check(cudaMalloc(&d_energy, sample_count * sizeof(cuDoubleComplex)),
              "cudaMalloc energy");
        check(cudaMemset(d_energy, 0, sample_count * sizeof(cuDoubleComplex)),
              "cudaMemset energy");
        peps_cuda::launch_diagonal_heisenberg_energy(
            d_spins, d_pairs, d_energy, sample_count, site_count, 1, 1.0, stream);
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize energy");
        const std::vector<cuDoubleComplex> energy =
            copy_to_host(d_energy, sample_count);
        check_close(energy[0].x, -0.25, "energy[0]");
        check_close(energy[1].x, -0.25, "energy[1]");

        const std::vector<double> diagonal_values = {10.0, 11.0, 12.0, 13.0};
        double *d_diagonal_values = copy_to_device(diagonal_values);
        check(cudaMemset(d_energy, 0, sample_count * sizeof(cuDoubleComplex)),
              "cudaMemset generic energy");
        peps_cuda::launch_diagonal_two_site_energy(
            d_spins, d_pairs, d_diagonal_values, d_energy, sample_count,
            site_count, 1, 2, stream);
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize generic energy");
        const std::vector<cuDoubleComplex> generic_energy =
            copy_to_host(d_energy, sample_count);
        check_close(generic_energy[0].x, 11.0, "generic energy[0]");
        check_close(generic_energy[1].x, 12.0, "generic energy[1]");

        const std::vector<int> one_site_terms = {0, 1};
        const std::vector<double> one_site_diagonal = {1.0, 2.0, 10.0, 20.0};
        int *d_one_site_terms = copy_to_device(one_site_terms);
        double *d_one_site_diagonal = copy_to_device(one_site_diagonal);
        check(cudaMemset(d_energy, 0, sample_count * sizeof(cuDoubleComplex)),
              "cudaMemset one-site energy");
        peps_cuda::launch_diagonal_one_site_energy(
            d_spins, d_one_site_terms, d_one_site_diagonal, d_energy,
            sample_count, site_count, 2, 2, stream);
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize one-site energy");
        const std::vector<cuDoubleComplex> one_site_energy =
            copy_to_host(d_energy, sample_count);
        check_close(one_site_energy[0].x, 21.0, "one-site energy[0]");
        check_close(one_site_energy[1].x, 12.0, "one-site energy[1]");

        const std::vector<cuDoubleComplex> log_psi = {
            make_cuDoubleComplex(0.0, 0.0),
            make_cuDoubleComplex(std::log(2.0), 0.0)};
        const std::vector<double> log_pc = {0.0, std::log(4.0)};
        cuDoubleComplex *d_log_psi = copy_to_device(log_psi);
        double *d_log_pc = copy_to_device(log_pc);
        peps_cuda::launch_importance_weights(d_log_psi, d_log_pc, d_weights, 2, 0.0,
                                             stream);
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize importance");
        const std::vector<double> imp = copy_to_host(d_weights, 2);
        check_close(imp[0], 1.0, "importance[0]");
        check_close(imp[1], 1.0, "importance[1]");

        const int ns = 2;
        const int np = 3;
        const std::vector<cuDoubleComplex> o = {
            make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(2.0, 0.0),
            make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
            make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(1.0, 0.0)};
        cuDoubleComplex *d_o = copy_to_device(o);
        cuDoubleComplex *d_gram = nullptr;
        check(cudaMalloc(&d_gram, ns * ns * sizeof(cuDoubleComplex)),
              "cudaMalloc gram");
        peps_cuda::launch_dense_minsr_gram(d_o, d_gram, ns, np, 0.5, stream);
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize gram");
        const std::vector<cuDoubleComplex> gram = copy_to_host(d_gram, ns * ns);
        check_close(gram[0].x, 5.5, "gram[0,0]");
        check_close(gram[1].x, 2.0, "gram[0,1]");
        check_close(gram[2].x, 2.0, "gram[1,0]");
        check_close(gram[3].x, 2.5, "gram[1,1]");

        const std::vector<cuDoubleComplex> sampled_o = {
            make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(2.0, 0.0),
            make_cuDoubleComplex(3.0, 0.0), make_cuDoubleComplex(4.0, 0.0),
            make_cuDoubleComplex(10.0, 0.0), make_cuDoubleComplex(20.0, 0.0),
            make_cuDoubleComplex(30.0, 0.0), make_cuDoubleComplex(40.0, 0.0)};
        const std::vector<int> sampled_spins = {0, 1, 0, 0};
        const std::vector<std::size_t> sampled_offsets = {0, 2};
        const std::vector<std::size_t> sampled_slices = {2, 2};
        cuDoubleComplex *d_sampled_o = copy_to_device(sampled_o);
        int *d_sampled_spins = copy_to_device(sampled_spins);
        std::size_t *d_sampled_offsets = copy_to_device(sampled_offsets);
        std::size_t *d_sampled_slices = copy_to_device(sampled_slices);
        peps_cuda::launch_sampled_sector_minsr_gram(
            d_sampled_o, d_sampled_spins, d_sampled_offsets, d_sampled_slices,
            d_gram, ns, site_count, 4, 0.5, stream);
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize sampled-sector gram");
        const std::vector<cuDoubleComplex> sampled_gram =
            copy_to_host(d_gram, ns * ns);
        check_close(sampled_gram[0].x, 30.5, "sampled gram[0,0]");
        check_close(sampled_gram[1].x, 50.0, "sampled gram[0,1]");
        check_close(sampled_gram[2].x, 50.0, "sampled gram[1,0]");
        check_close(sampled_gram[3].x, 3000.5, "sampled gram[1,1]");

        const std::vector<cuDoubleComplex> sample_vector = {
            make_cuDoubleComplex(2.0, 0.0), make_cuDoubleComplex(3.0, 0.0)};
        const std::vector<std::size_t> site_parameter_offsets = {0, 4};
        const std::vector<std::size_t> physical_strides = {2, 2};
        cuDoubleComplex *d_sample_vector = copy_to_device(sample_vector);
        std::size_t *d_site_parameter_offsets =
            copy_to_device(site_parameter_offsets);
        std::size_t *d_physical_strides = copy_to_device(physical_strides);
        cuDoubleComplex *d_parameter_vector = nullptr;
        check(cudaMalloc(&d_parameter_vector, 8 * sizeof(cuDoubleComplex)),
              "cudaMalloc sampled-sector parameter vector");
        check(cudaMemset(d_parameter_vector, 0, 8 * sizeof(cuDoubleComplex)),
              "cudaMemset sampled-sector parameter vector");
        peps_cuda::launch_sampled_sector_minsr_apply_odag(
            d_sampled_o, d_sampled_spins, d_sampled_offsets, d_sampled_slices,
            d_site_parameter_offsets, d_physical_strides, d_sample_vector,
            d_parameter_vector, ns, site_count, 4, stream);
        check(cudaStreamSynchronize(stream),
              "cudaStreamSynchronize sampled-sector apply");
        const std::vector<cuDoubleComplex> parameter_vector =
            copy_to_host(d_parameter_vector, 8);
        check_close(parameter_vector[0].x, -32.0, "sampled apply[0]");
        check_close(parameter_vector[1].x, -64.0, "sampled apply[1]");
        check_close(parameter_vector[2].x, 0.0, "sampled apply[2]");
        check_close(parameter_vector[3].x, 0.0, "sampled apply[3]");
        check_close(parameter_vector[4].x, -90.0, "sampled apply[4]");
        check_close(parameter_vector[5].x, -120.0, "sampled apply[5]");
        check_close(parameter_vector[6].x, -6.0, "sampled apply[6]");
        check_close(parameter_vector[7].x, -8.0, "sampled apply[7]");

        cudaFree(d_psi);
        cudaFree(d_weights);
        cudaFree(d_sites);
        cudaFree(d_spins);
        cudaFree(d_projected);
        cudaFree(d_ragged_sites);
        cudaFree(d_ragged_offsets);
        cudaFree(d_ragged_slices);
        cudaFree(d_ragged_projected_offsets);
        cudaFree(d_ragged_projected);
        cudaFree(d_pairs);
        cudaFree(d_energy);
        cudaFree(d_diagonal_values);
        cudaFree(d_one_site_terms);
        cudaFree(d_one_site_diagonal);
        cudaFree(d_log_psi);
        cudaFree(d_log_pc);
        cudaFree(d_o);
        cudaFree(d_gram);
        cudaFree(d_sampled_o);
        cudaFree(d_sampled_spins);
        cudaFree(d_sampled_offsets);
        cudaFree(d_sampled_slices);
        cudaFree(d_sample_vector);
        cudaFree(d_site_parameter_offsets);
        cudaFree(d_physical_strides);
        cudaFree(d_parameter_vector);
        check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    } catch (const std::exception &err) {
        std::cerr << "gpu smoke failure: " << err.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "peps_cuda_gpu_smoke passed\n";
    return EXIT_SUCCESS;
}
