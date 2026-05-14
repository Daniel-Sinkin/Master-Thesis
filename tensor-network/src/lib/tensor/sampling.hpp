// lib/tensor/sampling.hpp
#pragma once

#include "common.hpp"
#include "ndarray/ndarray.hpp"
#include "tensor/peps.hpp"

#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace ds_tn
{

using SpinConfiguration = std::vector<usize>;

struct ConfigurationProbability
{
    usize encoded{};
    SpinConfiguration spins{};
    f64 amplitude{};
    f64 weight{};
    f64 probability{};
};

struct ExactPepsDistribution
{
    usize n_rows{};
    usize n_cols{};
    usize physical_dim{};
    f64 norm_squared{};
    std::vector<ConfigurationProbability> states{};
};

struct RowProbability
{
    usize encoded{};
    SpinConfiguration spins{};
    f64 weight{};
    f64 probability{};
};

struct DirectSamplingStep
{
    usize row{};
    SpinConfiguration prefix_before{};
    std::vector<RowProbability> options{};
    SpinConfiguration selected_row{};
    f64 selected_probability{};
};

struct DirectSample
{
    SpinConfiguration spins{};
    f64 probability{};
    f64 log_probability{};
    std::vector<DirectSamplingStep> steps{};
};

struct ExactDirectSamplingConfig
{
    std::optional<NDArraySeed> seed{0};
};

[[nodiscard]] auto encode_base(std::span<const usize> digits, usize base) -> usize;
[[nodiscard]] auto decode_base(usize encoded, usize length, usize base) -> SpinConfiguration;
[[nodiscard]] auto spin_configuration_to_string(std::span<const usize> spins) -> std::string;
[[nodiscard]] auto peps_amplitude(const Peps& peps, std::span<const usize> spins) -> f64;
[[nodiscard]] auto exact_peps_distribution(const Peps& peps) -> ExactPepsDistribution;
[[nodiscard]] auto conditional_row_probabilities(
    const ExactPepsDistribution& distribution, usize row, std::span<const usize> prefix
) -> std::vector<RowProbability>;
[[nodiscard]] auto
sample_direct_exact(const ExactPepsDistribution& distribution, std::mt19937_64& rng)
    -> DirectSample;
[[nodiscard]] auto sample_direct_exact(const Peps& peps, std::mt19937_64& rng) -> DirectSample;
[[nodiscard]] auto sample_direct_exact(const Peps& peps, ExactDirectSamplingConfig cfg = {})
    -> DirectSample;

}  // namespace ds_tn
