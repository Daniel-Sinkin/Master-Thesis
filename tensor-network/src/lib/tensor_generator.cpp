// lib/tensor_generator.cpp
#include "tensor_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <ranges>
#include <stdexcept>

namespace ds_tn {
namespace {

[[nodiscard]] auto make_seeded_engine() -> std::mt19937_64 {
    auto seed_data = std::array<std::random_device::result_type, 8>{};
    auto random_device = std::random_device{};
    for (auto &seed : seed_data) {
        seed = random_device();
    }

    auto seed_sequence = std::seed_seq(seed_data.begin(), seed_data.end());
    return std::mt19937_64{seed_sequence};
}

} // namespace

TensorGenerator::TensorGenerator()
    : engine_(make_seeded_engine()) {}

TensorGenerator::TensorGenerator(TensorSeed seed)
    : engine_(seed) {}

auto TensorGenerator::uniform(std::vector<usize> shape, f64 lower, f64 upper) -> Tensor {
    if (not std::isfinite(lower) or not std::isfinite(upper)) {
        throw std::invalid_argument("TensorGenerator::uniform requires finite range endpoints.");
    }
    if (lower > upper) {
        throw std::invalid_argument("TensorGenerator::uniform requires lower <= upper.");
    }

    auto out = Tensor(std::move(shape));
    if (lower == upper) {
        std::ranges::fill(out.data(), out.data() + out.size(), lower);
        return out;
    }

    auto distribution =
        std::uniform_real_distribution<f64>{lower, std::nextafter(upper, std::numeric_limits<f64>::infinity())};
    std::ranges::generate(out.data(), out.data() + out.size(), [&]() { return distribution(engine_); });
    return out;
}

auto TensorGenerator::normal(std::vector<usize> shape, f64 mu, f64 sigma) -> Tensor {
    if (not std::isfinite(mu) or not std::isfinite(sigma)) {
        throw std::invalid_argument("TensorGenerator::normal requires finite distribution parameters.");
    }
    if (sigma < 0.0) {
        throw std::invalid_argument("TensorGenerator::normal requires sigma >= 0.");
    }

    auto out = Tensor(std::move(shape));
    if (sigma == 0.0) {
        std::ranges::fill(out.data(), out.data() + out.size(), mu);
        return out;
    }

    auto distribution = std::normal_distribution<f64>{mu, sigma};
    std::ranges::generate(out.data(), out.data() + out.size(), [&]() { return distribution(engine_); });
    return out;
}

} // namespace ds_tn
