// lib/tensor.hpp
#pragma once

#include "common.hpp"

#include <array>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ds_tn {

using TensorSeed = std::mt19937_64::result_type;

enum class TensorValidity : u8 {
    valid = 0,
    shape_stride_size_mismatch,
    shape_leg_name_size_mismatch,
    empty_leg_name,
    duplicate_leg_name,
    scalar_storage_size_mismatch,
    stride_mismatch,
    data_size_mismatch,
};

class Tensor {
public:
    explicit Tensor(std::vector<usize> shape);
    Tensor(std::vector<usize> shape, std::span<const std::string> leg_names);
    Tensor(std::vector<usize> shape, std::initializer_list<std::string> leg_names);
    [[nodiscard]] static auto scalar(f64 value) -> Tensor;
    [[nodiscard]] static auto vector(std::initializer_list<f64> values) -> Tensor;
    [[nodiscard]] static auto uniform_random(
        std::vector<usize> shape,
        f64 lower = 0.0,
        f64 upper = 1.0,
        std::optional<TensorSeed> seed = std::nullopt) -> Tensor;
    [[nodiscard]] static auto normal_random(
        std::vector<usize> shape,
        f64 mu = 0.0,
        f64 sigma = 1.0,
        std::optional<TensorSeed> seed = std::nullopt) -> Tensor;

    template <typename... Values>
        requires(sizeof...(Values) > 0) and (std::convertible_to<Values, f64> and ...)
    [[nodiscard]] static auto vector(Values... values) -> Tensor {
        return Tensor::vector({static_cast<f64>(values)...});
    }

    [[nodiscard]] static auto matrix(std::initializer_list<std::initializer_list<f64>> rows) -> Tensor;

    [[nodiscard]] auto rank() const noexcept -> usize;
    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto shape() const noexcept -> std::span<const usize>;
    [[nodiscard]] auto leg_names() const noexcept -> std::span<const std::string>;
    [[nodiscard]] auto leg_name(usize axis) const -> const std::string &;
    [[nodiscard]] auto data() noexcept -> f64 *;
    [[nodiscard]] auto data() const noexcept -> const f64 *;

    auto operator()(std::span<const usize> indices) -> f64 &;
    auto operator()(std::span<const usize> indices) const -> const f64 &;

    template <typename... Indices>
        requires(std::integral<Indices> and ...)
    auto operator()(Indices... indices) -> f64 & {
        auto axis = usize{0};
        const auto normalized =
            std::array<usize, sizeof...(Indices)>{normalize_integral_index(indices, axis++)...};
        return (*this)(std::span<const usize>{normalized});
    }

    template <typename... Indices>
        requires(std::integral<Indices> and ...)
    auto operator()(Indices... indices) const -> const f64 & {
        auto axis = usize{0};
        const auto normalized =
            std::array<usize, sizeof...(Indices)>{normalize_integral_index(indices, axis++)...};
        return (*this)(std::span<const usize>{normalized});
    }

    [[nodiscard]] auto indices_from_linear(usize linear_index) const -> std::vector<usize>;
    [[nodiscard]] auto validity() const noexcept -> TensorValidity;
    [[nodiscard]] auto normalized() const -> Tensor;
    auto normalize_in_place() -> void;
    auto add_scalar_in_place(f64 scalar) -> Tensor &;
    auto subtract_scalar_in_place(f64 scalar) -> Tensor &;
    auto multiply_scalar_in_place(f64 scalar) -> Tensor &;
    auto divide_scalar_in_place(f64 scalar) -> Tensor &;
    auto operator+=(f64 scalar) -> Tensor &;
    auto operator-=(f64 scalar) -> Tensor &;
    auto operator*=(f64 scalar) -> Tensor &;
    auto operator/=(f64 scalar) -> Tensor &;
    auto print(usize precision = 4, bool show_metadata = true, std::ostream &out = std::cout) const -> void;

    [[nodiscard]] auto is_scalar() const noexcept -> bool;
    [[nodiscard]] auto is_trivial() const noexcept -> bool;
    [[nodiscard]] auto is_vector() const noexcept -> bool;
    [[nodiscard]] auto is_matrix() const noexcept -> bool;
    [[nodiscard]] auto is_tensor3() const noexcept -> bool;

private:
    auto initialize_storage() -> void;
    [[nodiscard]] auto linear_index(std::span<const usize> indices) const -> usize;

    template <typename Index>
        requires std::integral<Index>
    [[nodiscard]] auto normalize_integral_index(Index index, usize axis) const -> usize {
        if (axis >= shape_.size()) {
            throw std::invalid_argument("Tensor index rank does not match tensor rank.");
        }

        if constexpr (std::signed_integral<Index>) {
            auto normalized = static_cast<isize>(index);
            if (normalized < 0) {
                normalized += static_cast<isize>(shape_[axis]);
            }
            if (normalized < 0 || normalized >= static_cast<isize>(shape_[axis])) {
                throw std::out_of_range("Tensor index exceeds tensor extent.");
            }
            return static_cast<usize>(normalized);
        } else {
            if (index > std::numeric_limits<usize>::max()) {
                throw std::out_of_range("Tensor index exceeds tensor extent.");
            }
            const auto normalized = static_cast<usize>(index);
            if (normalized >= shape_[axis]) {
                throw std::out_of_range("Tensor index exceeds tensor extent.");
            }
            return normalized;
        }
    }

    std::vector<usize> shape_{};
    std::vector<usize> strides_{};
    std::vector<std::string> leg_names_{};
    std::vector<f64> data_{};
};

} // namespace ds_tn
