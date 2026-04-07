// lib/ndarray/generator.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <random>
#include <vector>

namespace ds_tn {

class NDArrayGenerator {
public:
    NDArrayGenerator();
    explicit NDArrayGenerator(NDArraySeed seed);

    [[nodiscard]] auto uniform(std::vector<usize> shape, f64 lower = 0.0, f64 upper = 1.0) -> NDArray;
    [[nodiscard]] auto normal(std::vector<usize> shape, f64 mu = 0.0, f64 sigma = 1.0) -> NDArray;

private:
    std::mt19937_64 engine_{};
};

} // namespace ds_tn
