// lib/tensor/tensor.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <concepts>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ds_tn
{

using TensorSeed = NDArraySeed;

enum class TensorValidity : u8
{
    valid = 0,
    array_invalid,
    shape_leg_name_size_mismatch,
    empty_leg_name,
    duplicate_leg_name,
};

class Tensor
{
  public:
    Tensor() = default;
    explicit Tensor(NDArray array);
    Tensor(NDArray array, std::vector<std::string> leg_names);
    Tensor(NDArray array, std::span<const std::string> leg_names);
    Tensor(NDArray array, std::initializer_list<std::string> leg_names);
    explicit Tensor(std::vector<usize> shape);
    Tensor(std::vector<usize> shape, std::vector<std::string> leg_names);
    Tensor(std::vector<usize> shape, std::span<const std::string> leg_names);
    Tensor(std::vector<usize> shape, std::initializer_list<std::string> leg_names);

    [[nodiscard]] static auto scalar(f64 value) -> Tensor;
    [[nodiscard]] static auto diag(const Tensor& vector) -> Tensor;
    [[nodiscard]] static auto iota(usize size) -> Tensor;
    [[nodiscard]] static auto vector(std::initializer_list<f64> values) -> Tensor;
    [[nodiscard]] static auto random(
        std::vector<usize> shape,
        RandomOptions options,
        std::optional<TensorSeed> seed = std::nullopt
    ) -> Tensor;
    [[nodiscard]] static auto random_uniform(
        std::vector<usize> shape,
        RandomUniformOptions options = {},
        std::optional<TensorSeed> seed = std::nullopt
    ) -> Tensor;
    [[nodiscard]] static auto random_normal(
        std::vector<usize> shape,
        RandomNormalOptions options = {},
        std::optional<TensorSeed> seed = std::nullopt
    ) -> Tensor;

    template <typename... Values>
        requires(sizeof...(Values) > 0) and (std::convertible_to<Values, f64> and ...)
    [[nodiscard]] static auto vector(Values... values) -> Tensor
    {
        return Tensor::vector({static_cast<f64>(values)...});
    }

    [[nodiscard]] static auto matrix(std::initializer_list<std::initializer_list<f64>> rows)
        -> Tensor;
    [[nodiscard]] static auto rank3(
        std::initializer_list<std::initializer_list<std::initializer_list<f64>>> slices
    ) -> Tensor;

    [[nodiscard]] auto rank() const noexcept -> usize;
    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto shape() const noexcept -> std::span<const usize>;
    [[nodiscard]] auto shape(usize axis) const -> usize;
    [[nodiscard]] auto leg_names() const noexcept -> std::span<const std::string>;
    [[nodiscard]] auto leg_name(usize axis) const -> const std::string&;
    [[nodiscard]] auto array() noexcept -> NDArray&;
    [[nodiscard]] auto array() const noexcept -> const NDArray&;
    [[nodiscard]] auto data() noexcept -> f64*;
    [[nodiscard]] auto data() const noexcept -> const f64*;

    auto operator()(std::span<const usize> indices) -> f64&;
    auto operator()(std::span<const usize> indices) const -> const f64&;

    template <typename... Indices>
        requires(std::integral<Indices> and ...)
    auto operator()(Indices... indices) -> f64&
    {
        return values_(indices...);
    }

    template <typename... Indices>
        requires(std::integral<Indices> and ...)
    auto operator()(Indices... indices) const -> const f64&
    {
        return values_(indices...);
    }

    [[nodiscard]] auto indices_from_linear(usize linear_index) const -> std::vector<usize>;
    [[nodiscard]] auto validity() const noexcept -> TensorValidity;
    [[nodiscard]] auto diag() const -> Tensor;
    [[nodiscard]] auto format_metadata() const -> std::string;
    auto rename_leg(const std::string& old_name, const std::string& new_name) -> void;
    auto print_metadata(LogSettings settings, std::ostream& out = std::cout) const -> void;
    auto print_metadata(std::string_view name, std::ostream& out = std::cout) const -> void;
    auto print_metadata(std::ostream& out = std::cout) const -> void;
    auto print(usize precision = 4, bool show_metadata = true, std::ostream& out = std::cout) const
        -> void;

    [[nodiscard]] auto is_scalar() const noexcept -> bool;
    [[nodiscard]] auto is_trivial() const noexcept -> bool;
    [[nodiscard]] auto is_vector() const noexcept -> bool;
    [[nodiscard]] auto is_matrix() const noexcept -> bool;
    [[nodiscard]] auto is_tensor3() const noexcept -> bool;

  private:
    NDArray values_{};
    std::vector<std::string> leg_names_{};
};

}  // namespace ds_tn
