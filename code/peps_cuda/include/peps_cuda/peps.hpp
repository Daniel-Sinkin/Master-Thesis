#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace peps_cuda {

#if defined(PEPS_CUDA_REAL_FLOAT)
using Real = float;
#else
using Real = double;
#endif

using Complex = std::complex<Real>;

struct SiteTensor {
    int phys = 0;
    int north = 0;
    int east = 0;
    int south = 0;
    int west = 0;
    std::vector<Complex> data;

    [[nodiscard]] std::size_t parameter_count() const;
    [[nodiscard]] std::size_t index(int p, int n, int e, int s, int w) const;

    Complex &operator()(int p, int n, int e, int s, int w);
    const Complex &operator()(int p, int n, int e, int s, int w) const;
};

struct PEPS {
    int lx = 0;
    int ly = 0;
    int local_dim = 0;
    int max_bond_dim = 0;
    std::vector<SiteTensor> sites;

    SiteTensor &at(int x, int y);
    const SiteTensor &at(int x, int y) const;
    [[nodiscard]] std::size_t site_index(int x, int y) const;
    [[nodiscard]] std::size_t parameter_count() const;
};

struct Sample {
    int lx = 0;
    int ly = 0;
    int local_dim = 0;
    std::vector<int> spin;

    int &at(int x, int y);
    const int &at(int x, int y) const;
    [[nodiscard]] std::size_t site_index(int x, int y) const;
};

struct LocalOperatorTerm {
    std::vector<int> sites;
    int local_dim = 0;
    Complex coefficient = {1.0, 0.0};
    std::vector<Complex> matrix;
};

enum class FlipKind {
    Diagonal,
    SingleSite,
    HorizontalNearest,
    VerticalNearest,
    Plaquette2x2,
    HorizontalLong,
    Other,
};

struct FlipContribution {
    std::vector<int> sites;
    std::vector<int> output_values;
    Complex matrix_element = {0.0, 0.0};
    FlipKind kind = FlipKind::Other;
};

struct SampleEvaluation {
    Sample sample;
    Complex psi = {0.0, 0.0};
    Complex log_psi = {0.0, 0.0};
    Complex local_energy = {0.0, 0.0};
    std::vector<Complex> log_gradients;
};

struct SampleBatch {
    std::vector<Sample> samples;
    std::vector<Complex> log_psi;
    std::vector<Complex> local_energy;
    std::vector<double> importance_weights;
    std::vector<std::vector<Complex>> O;
};

struct PackedSiteMetadata {
    int phys = 0;
    int north = 0;
    int east = 0;
    int south = 0;
    int west = 0;
    std::size_t offset = 0;
    std::size_t physical_slice_size = 0;
};

struct PackedPEPS {
    int lx = 0;
    int ly = 0;
    int local_dim = 0;
    std::size_t projected_sample_stride = 0;
    std::vector<PackedSiteMetadata> sites;
    std::vector<std::size_t> projected_offsets;
    std::vector<Complex> data;
};

struct ParameterBlock {
    std::size_t site = 0;
    std::size_t offset = 0;
    std::size_t count = 0;
    std::size_t physical_stride = 0;
};

struct ParameterLayout {
    std::vector<ParameterBlock> blocks;
    std::size_t total_parameters = 0;
};

PEPS make_random_open_peps(int lx, int ly, int local_dim, int bond_dim,
                           std::uint64_t seed, double mean_shift = 0.0,
                           double scale = 1.0);

Sample make_zero_sample(int lx, int ly, int local_dim);

Complex contract_amplitude_exact(const PEPS &peps, const Sample &sample);

FlipKind classify_flip_sites(const std::vector<int> &sites, int ly);

std::string to_string(FlipKind kind);

constexpr std::size_t flip_kind_count = 7;

std::size_t flip_kind_index(FlipKind kind);

std::vector<FlipContribution>
enumerate_flip_contributions(const Sample &sample,
                             const LocalOperatorTerm &term);

std::array<std::size_t, flip_kind_count>
summarize_flip_buckets(const Sample &sample,
                       const std::vector<LocalOperatorTerm> &hamiltonian);

Complex local_energy_exact(const PEPS &peps, const Sample &sample,
                           const std::vector<LocalOperatorTerm> &hamiltonian,
                           Complex psi);

std::vector<Complex> log_gradients_exact(const PEPS &peps,
                                         const Sample &sample, Complex psi,
                                         std::size_t max_parameters = 200000);

Sample sample_exact_small(const PEPS &peps, std::mt19937_64 &rng,
                          std::size_t max_enumerated_states = 1ULL << 20);

SampleEvaluation evaluate_sample_exact(
    const PEPS &peps, const Sample &sample,
    const std::vector<LocalOperatorTerm> &hamiltonian,
    std::size_t max_gradient_parameters = 200000);

SampleBatch generate_Oks_and_Eks_exact(
    const PEPS &peps, const std::vector<LocalOperatorTerm> &hamiltonian,
    int sample_count, std::uint64_t seed,
    std::size_t max_enumerated_states = 1ULL << 20,
    std::size_t max_gradient_parameters = 200000);

std::vector<Complex> minsr_direction(const std::vector<std::vector<Complex>> &O,
                                     const std::vector<Complex> &E,
                                     double diagonal_shift = 1.0e-8);

std::vector<Complex>
sr_direction_parameter_space(const std::vector<std::vector<Complex>> &O,
                             const std::vector<Complex> &E,
                             double diagonal_shift = 1.0e-8);

std::vector<Complex>
minsr_direction_weighted(const std::vector<std::vector<Complex>> &O,
                         const std::vector<Complex> &E,
                         const std::vector<double> &weights,
                         double diagonal_shift = 1.0e-8);

std::vector<Complex> minsr_direction_sampled_sector(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &sampled_sector_rows,
    const std::vector<Complex> &E, double diagonal_shift = 1.0e-8);

std::vector<Complex> minsr_direction_sampled_sector_weighted(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &sampled_sector_rows,
    const std::vector<Complex> &E, const std::vector<double> &weights,
    double diagonal_shift = 1.0e-8);

std::vector<LocalOperatorTerm> make_nearest_neighbor_heisenberg(int lx, int ly,
                                                                double j1);

std::vector<LocalOperatorTerm>
make_transverse_field_ising(int lx, int ly, double jz, double hx);

std::vector<LocalOperatorTerm>
make_square_rydberg_hamiltonian(int lx, int ly, double omega, double detuning,
                                double c6, double cutoff_radius = -1.0);

std::vector<int> make_nearest_neighbor_bond_pairs(int lx, int ly);

std::vector<int> flatten_samples_sample_major(const std::vector<Sample> &samples);

std::size_t dense_o_bytes(int sample_count, std::size_t parameter_count);

std::size_t sampled_sector_parameter_count(const PEPS &peps);

std::size_t sampled_sector_o_bytes(int sample_count,
                                   std::size_t sampled_parameter_count);

std::vector<Complex>
compact_sampled_sector_log_gradients(const PEPS &peps, const Sample &sample,
                                     const std::vector<Complex> &dense_row);

std::vector<std::vector<Complex>> compact_sampled_sector_log_gradients(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &dense_rows);

std::vector<std::vector<Complex>> sampled_sector_gram(
    const PEPS &peps, const std::vector<Sample> &samples,
    const std::vector<std::vector<Complex>> &sampled_sector_rows,
    double diagonal_shift = 0.0);

std::vector<double>
compute_importance_weights(const std::vector<Complex> &log_psi,
                           const std::vector<double> &log_sampling_probability);

PackedPEPS pack_site_tensors_physical_major(const PEPS &peps,
                                            std::size_t alignment = 1);

ParameterLayout make_parameter_layout(const PEPS &peps);

double squared_norm(const std::vector<Complex> &values);

} // namespace peps_cuda
