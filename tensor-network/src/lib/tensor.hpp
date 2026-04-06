#pragma once

#include "common.hpp"

#include <iostream>
#include <span>
#include <vector>

namespace ds_tn {

class Tensor {
public:
    explicit Tensor(std::vector<usize> shape);

    [[nodiscard]] auto rank() const noexcept -> usize;
    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto shape() const noexcept -> std::span<const usize>;
    [[nodiscard]] auto data() noexcept -> f64 *;
    [[nodiscard]] auto data() const noexcept -> const f64 *;

    auto operator()(std::span<const usize> indices) -> f64 &;
    auto operator()(std::span<const usize> indices) const -> const f64 &;
    [[nodiscard]] auto indices_from_linear(usize linear_index) const -> std::vector<usize>;
    auto print(usize precision = 4, bool show_metadata = true, std::ostream &out = std::cout) const -> void;

    [[nodiscard]] auto is_scalar() const noexcept -> bool;
    [[nodiscard]] auto is_trivial() const noexcept -> bool;
    [[nodiscard]] auto is_vector() const noexcept -> bool;
    [[nodiscard]] auto is_matrix() const noexcept -> bool;
    [[nodiscard]] auto is_tensor3() const noexcept -> bool;

private:
    [[nodiscard]] auto linear_index(std::span<const usize> indices) const -> usize;

    std::vector<usize> shape_{};
    std::vector<usize> strides_{};
    std::vector<f64> data_{};
};

} // namespace ds_tn
