// lib/ndarray/ndarray.hpp
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
#include <vector>

namespace ds_tn {

using NDArraySeed = std::mt19937_64::result_type;

enum class NDArrayValidity : u8 {
    valid = 0,
    shape_stride_size_mismatch,
    scalar_storage_size_mismatch,
    stride_mismatch,
    data_size_mismatch,
};

class NDArray {
public:
    explicit NDArray(std::vector<usize> shape);

    [[nodiscard]] static auto scalar(f64 value) -> NDArray;
    [[nodiscard]] static auto vector(std::initializer_list<f64> values) -> NDArray;
    [[nodiscard]] static auto uniform_random(
        std::vector<usize> shape,
        f64 lower = 0.0,
        f64 upper = 1.0,
        std::optional<NDArraySeed> seed = std::nullopt) -> NDArray;
    [[nodiscard]] static auto normal_random(
        std::vector<usize> shape,
        f64 mu = 0.0,
        f64 sigma = 1.0,
        std::optional<NDArraySeed> seed = std::nullopt) -> NDArray;

    template <typename... Values>
        requires(sizeof...(Values) > 0) and (std::convertible_to<Values, f64> and ...)
    [[nodiscard]] static auto vector(Values... values) -> NDArray {
        return NDArray::vector({static_cast<f64>(values)...});
    }

    [[nodiscard]] static auto matrix(std::initializer_list<std::initializer_list<f64>> rows) -> NDArray;

    [[nodiscard]] auto rank() const noexcept -> usize;
    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto shape() const noexcept -> std::span<const usize>;
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
    [[nodiscard]] auto validity() const noexcept -> NDArrayValidity;
    [[nodiscard]] auto normalized() const -> NDArray;
    auto normalize() -> void;
    auto add_scalar(f64 scalar) -> NDArray &;
    auto subtract_scalar(f64 scalar) -> NDArray &;
    auto multiply_scalar(f64 scalar) -> NDArray &;
    auto divide_scalar(f64 scalar) -> NDArray &;
    auto operator+=(f64 scalar) -> NDArray &;
    auto operator-=(f64 scalar) -> NDArray &;
    auto operator*=(f64 scalar) -> NDArray &;
    auto operator/=(f64 scalar) -> NDArray &;
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
            throw std::invalid_argument("NDArray index rank does not match array rank.");
        }

        if constexpr (std::signed_integral<Index>) {
            auto normalized = static_cast<isize>(index);
            if (normalized < 0) {
                normalized += static_cast<isize>(shape_[axis]);
            }
            if (normalized < 0 || normalized >= static_cast<isize>(shape_[axis])) {
                throw std::out_of_range("NDArray index exceeds array extent.");
            }
            return static_cast<usize>(normalized);
        } else {
            if (index > std::numeric_limits<usize>::max()) {
                throw std::out_of_range("NDArray index exceeds array extent.");
            }
            const auto normalized = static_cast<usize>(index);
            if (normalized >= shape_[axis]) {
                throw std::out_of_range("NDArray index exceeds array extent.");
            }
            return normalized;
        }
    }

    std::vector<usize> shape_{};
    std::vector<usize> strides_{};
    std::vector<f64> data_{};
};

} // namespace ds_tn
