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
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ds_tn
{

using NDArraySeed = std::mt19937_64::result_type;

class NDArray;
[[nodiscard]] auto truncate_cols(const NDArray& mat, usize cols) -> NDArray;
[[nodiscard]] auto truncate_rows(const NDArray& mat, usize rows) -> NDArray;
[[nodiscard]] auto transpose_matrix(const NDArray& matrix) -> NDArray;

struct RandomUniformOptions
{
    f64 lower{0.0};
    f64 upper{1.0};
};

struct RandomNormalOptions
{
    f64 mu{0.0};
    f64 sigma{1.0};
};

using RandomOptions = std::variant<RandomUniformOptions, RandomNormalOptions>;

enum class NDArrayValidity : u8
{
    valid = 0,
    shape_stride_size_mismatch,
    scalar_storage_size_mismatch,
    stride_mismatch,
    data_size_mismatch,
};

class NDArray
{
  public:
    NDArray();
    explicit NDArray(std::vector<usize> shape);

    [[nodiscard]] static auto scalar(f64 value) -> NDArray;
    [[nodiscard]] static auto diag(const NDArray& vector) -> NDArray;
    [[nodiscard]] static auto iota(usize size) -> NDArray;
    [[nodiscard]] static auto vector(std::initializer_list<f64> values) -> NDArray;
    [[nodiscard]] static auto random(
        std::vector<usize> shape,
        RandomOptions options,
        std::optional<NDArraySeed> seed = std::nullopt
    ) -> NDArray;
    [[nodiscard]] static auto random_uniform(
        std::vector<usize> shape,
        RandomUniformOptions options = {},
        std::optional<NDArraySeed> seed = std::nullopt
    ) -> NDArray;
    [[nodiscard]] static auto random_normal(
        std::vector<usize> shape,
        RandomNormalOptions options = {},
        std::optional<NDArraySeed> seed = std::nullopt
    ) -> NDArray;
    [[nodiscard]] static auto zeros_like(const NDArray& other) -> NDArray;

    template <typename... Values>
        requires(sizeof...(Values) > 0) and (std::convertible_to<Values, f64> and ...)
    [[nodiscard]] static auto vector(Values... values) -> NDArray
    {
        return NDArray::vector({static_cast<f64>(values)...});
    }

    [[nodiscard]] static auto matrix(std::initializer_list<std::initializer_list<f64>> rows)
        -> NDArray;
    [[nodiscard]] static auto rank3(
        std::initializer_list<std::initializer_list<std::initializer_list<f64>>> slices
    ) -> NDArray;
    [[nodiscard]] static auto reshape(const NDArray& array, std::span<const usize> new_shape)
        -> NDArray;
    [[nodiscard]] static auto reshape(const NDArray& array, std::initializer_list<usize> new_shape)
        -> NDArray;
    [[nodiscard]] static auto squeeze(const NDArray& array) -> NDArray;
    [[nodiscard]] static auto same_shape(const NDArray& lhs, const NDArray& rhs) noexcept -> bool;

    [[nodiscard]] auto rank() const noexcept -> usize;
    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto shape() const noexcept -> std::span<const usize>;
    [[nodiscard]] auto shape(usize axis) const -> usize;
    [[nodiscard]] auto data() noexcept -> f64*;
    [[nodiscard]] auto data() const noexcept -> const f64*;
    [[nodiscard]] auto data(usize linear_index) noexcept -> f64&;
    [[nodiscard]] auto data(usize linear_index) const noexcept -> const f64&;
    [[nodiscard]] auto same_shape(const NDArray& other) const noexcept -> bool;

    auto operator()(std::span<const usize> indices) -> f64&;
    auto operator()(std::span<const usize> indices) const -> const f64&;

    template <typename... Indices>
        requires(std::integral<Indices> and ...)
    auto operator()(Indices... indices) -> f64&
    {
        auto axis = usize{0};
        const auto normalized =
            std::array<usize, sizeof...(Indices)>{normalize_integral_index(indices, axis++)...};
        return (*this)(std::span<const usize>{normalized});
    }

    template <typename... Indices>
        requires(std::integral<Indices> and ...)
    auto operator()(Indices... indices) const -> const f64&
    {
        auto axis = usize{0};
        const auto normalized =
            std::array<usize, sizeof...(Indices)>{normalize_integral_index(indices, axis++)...};
        return (*this)(std::span<const usize>{normalized});
    }

    [[nodiscard]] auto indices_from_linear(usize linear_index) const -> std::vector<usize>;
    [[nodiscard]] auto validity() const noexcept -> NDArrayValidity;
    [[nodiscard]] auto l2_norm() const -> f64;
    [[nodiscard]] auto normalized() const -> NDArray;
    [[nodiscard]] auto zeros_like() const -> NDArray;
    [[nodiscard]] auto diag() const -> NDArray;
    [[nodiscard]] auto reshape(std::span<const usize> new_shape) const -> NDArray;
    [[nodiscard]] auto reshape(std::initializer_list<usize> new_shape) const -> NDArray;
    [[nodiscard]] auto squeeze() const -> NDArray;
    [[nodiscard]] auto format_metadata() const -> std::string;
    auto normalize() -> void;
    auto add_scalar(f64 scalar) -> NDArray&;
    auto subtract_scalar(f64 scalar) -> NDArray&;
    auto multiply_scalar(f64 scalar) -> NDArray&;
    auto divide_scalar(f64 scalar) -> NDArray&;
    auto operator+=(const NDArray& rhs) -> NDArray&;
    auto operator-=(const NDArray& rhs) -> NDArray&;
    auto operator+=(f64 scalar) -> NDArray&;
    auto operator-=(f64 scalar) -> NDArray&;
    auto operator*=(f64 scalar) -> NDArray&;
    auto operator/=(f64 scalar) -> NDArray&;
    auto print(usize precision = 4, bool show_metadata = true, std::ostream& out = std::cout) const
        -> void;

    [[nodiscard]] auto is_scalar() const noexcept -> bool;
    [[nodiscard]] auto is_trivial() const noexcept -> bool;
    [[nodiscard]] auto is_vector() const noexcept -> bool;
    [[nodiscard]] auto is_matrix() const noexcept -> bool;
    [[nodiscard]] auto is_tensor3() const noexcept -> bool;

  private:
    auto initialize_storage() -> void;
    [[nodiscard]] auto linear_index(std::span<const usize> indices) const -> usize;

    friend auto truncate_cols(const NDArray& mat, usize cols) -> NDArray;
    friend auto truncate_rows(const NDArray& mat, usize rows) -> NDArray;

    template <typename Index>
        requires std::integral<Index>
    [[nodiscard]] auto normalize_integral_index(Index index, usize axis) const -> usize
    {
        if (axis >= shape_.size())
        {
            throw std::invalid_argument("NDArray index rank does not match array rank.");
        }

        if constexpr (std::signed_integral<Index>)
        {
            auto normalized = static_cast<isize>(index);
            if (normalized < 0)
            {
                normalized += static_cast<isize>(shape_[axis]);
            }
            if (normalized < 0 || normalized >= static_cast<isize>(shape_[axis]))
            {
                throw std::out_of_range("NDArray index exceeds array extent.");
            }
            return static_cast<usize>(normalized);
        }
        else
        {
            if (index > std::numeric_limits<usize>::max())
            {
                throw std::out_of_range("NDArray index exceeds array extent.");
            }
            const auto normalized = static_cast<usize>(index);
            if (normalized >= shape_[axis])
            {
                throw std::out_of_range("NDArray index exceeds array extent.");
            }
            return normalized;
        }
    }

    std::vector<usize> shape_{};
    std::vector<usize> strides_{};
    std::vector<f64> data_{};
};

[[nodiscard]] auto operator+(NDArray lhs, const NDArray& rhs) -> NDArray;

}  // namespace ds_tn
