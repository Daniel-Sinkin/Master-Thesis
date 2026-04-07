// lib/tensor_generator.hpp
#pragma once

#include "tensor.hpp"

namespace ds_tn {

class TensorGenerator {
public:
    TensorGenerator();
    explicit TensorGenerator(TensorSeed seed);

    [[nodiscard]] auto uniform(std::vector<usize> shape, f64 lower = 0.0, f64 upper = 1.0) -> Tensor;
    [[nodiscard]] auto normal(std::vector<usize> shape, f64 mu = 0.0, f64 sigma = 1.0) -> Tensor;

private:
    std::mt19937_64 engine_{};
};

} // namespace ds_tn
