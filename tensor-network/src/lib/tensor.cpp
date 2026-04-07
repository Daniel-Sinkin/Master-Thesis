// lib/tensor.cpp
#include "tensor.hpp"
#include "tensor_generator.hpp"
#include "tensor_stats.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>

#include <vecLib/cblas_new.h>

namespace ds_tn {
namespace {

[[nodiscard]] auto shape_to_string(std::span<const usize> shape) -> std::string {
    auto buffer = std::ostringstream{};
    buffer << '[';
    for (usize axis = 0; axis < shape.size(); ++axis) {
        if (axis != 0) {
            buffer << " x ";
        }
        buffer << shape[axis];
    }
    buffer << ']';
    return buffer.str();
}

[[nodiscard]] auto leg_names_to_string(std::span<const std::string> leg_names) -> std::string {
    auto buffer = std::ostringstream{};
    buffer << '[';
    for (usize axis = 0; axis < leg_names.size(); ++axis) {
        if (axis != 0) {
            buffer << ", ";
        }
        buffer << leg_names[axis];
    }
    buffer << ']';
    return buffer.str();
}

[[nodiscard]] auto make_default_leg_names(usize rank) -> std::vector<std::string> {
    static auto next_tensor_id = std::atomic<u64>{0};

    const auto tensor_id = next_tensor_id.fetch_add(1, std::memory_order_relaxed);
    auto names = std::vector<std::string>{};
    names.reserve(rank);

    for (const auto axis : iota_n(rank)) {
        names.push_back("t" + std::to_string(tensor_id) + "_l" + std::to_string(axis));
    }

    return names;
}

[[nodiscard]] auto as_blas_int(usize value) -> __LAPACK_int {
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max())) {
        throw std::overflow_error("Tensor extent exceeds BLAS integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

[[nodiscard]] auto leg_names_validity(
    std::span<const usize> shape,
    std::span<const std::string> leg_names) -> TensorValidity {
    if (leg_names.size() != shape.size()) {
        return TensorValidity::shape_leg_name_size_mismatch;
    }

    for (usize axis = 0; axis < leg_names.size(); ++axis) {
        if (leg_names[axis].empty()) {
            return TensorValidity::empty_leg_name;
        }

        for (usize other_axis = axis + 1; other_axis < leg_names.size(); ++other_axis) {
            if (leg_names[axis] == leg_names[other_axis]) {
                return TensorValidity::duplicate_leg_name;
            }
        }
    }

    return TensorValidity::valid;
}

[[nodiscard]] auto format_value(f64 value, usize precision) -> std::string {
    auto buffer = std::ostringstream{};
    buffer << std::fixed << std::setprecision(static_cast<int>(precision)) << value;
    return buffer.str();
}

[[nodiscard]] auto element_width(const Tensor &tensor, usize precision) -> usize {
    auto width = usize{0};
    for (usize index = 0; index < tensor.size(); ++index) {
        width = std::max(width, format_value(tensor.data()[index], precision).size());
    }
    return width;
}

auto append_matrix(
    std::ostringstream &buffer,
    const Tensor &tensor,
    usize precision,
    usize width,
    usize slice = 0,
    bool rank3 = false) -> void {
    const auto rows = rank3 ? tensor.shape()[1] : tensor.shape()[0];
    const auto cols = rank3 ? tensor.shape()[2] : tensor.shape()[1];

    for (usize row = 0; row < rows; ++row) {
        buffer << '[';
        for (usize col = 0; col < cols; ++col) {
            if (col != 0) {
                buffer << ' ';
            }

            const auto value = [&]() -> f64 {
                if (rank3) {
                    return tensor(std::array<usize, 3>{slice, row, col});
                }
                return tensor(std::array<usize, 2>{row, col});
            }();

            buffer << std::setw(static_cast<int>(width)) << format_value(value, precision);
        }
        buffer << ']';
        if (row + 1 != rows) {
            buffer << '\n';
        }
    }
}

} // namespace

Tensor::Tensor(std::vector<usize> shape)
    : shape_(std::move(shape)),
      strides_(shape_.size(), 1),
      leg_names_(make_default_leg_names(shape_.size())) {
    initialize_storage();
}

Tensor::Tensor(std::vector<usize> shape, std::span<const std::string> leg_names)
    : shape_(std::move(shape)),
      strides_(shape_.size(), 1),
      leg_names_(leg_names.begin(), leg_names.end()) {
    if (leg_names_validity(shape_, leg_names_) != TensorValidity::valid) {
        throw std::invalid_argument("Tensor leg names must match rank, be non-empty, and be unique.");
    }
    initialize_storage();
}

Tensor::Tensor(std::vector<usize> shape, std::initializer_list<std::string> leg_names)
    : Tensor(std::move(shape), std::span<const std::string>{leg_names.begin(), leg_names.size()}) {}

auto Tensor::initialize_storage() -> void {
    std::exclusive_scan(shape_.rbegin(), shape_.rend(), strides_.rbegin(), usize{1}, std::multiplies<>{});

    if (shape_.empty()) {
        data_.resize(1);
        return;
    }

    data_.resize(strides_.front() * shape_.front());
}

auto Tensor::scalar(f64 value) -> Tensor {
    auto out = Tensor({});
    out.data_[0] = value;
    return out;
}

auto Tensor::vector(std::initializer_list<f64> values) -> Tensor {
    auto out = Tensor({values.size()});
    std::ranges::copy(values, out.data_.begin());
    return out;
}

auto Tensor::uniform_random(std::vector<usize> shape, f64 lower, f64 upper, std::optional<TensorSeed> seed) -> Tensor {
    if (seed.has_value()) {
        return TensorGenerator(*seed).uniform(std::move(shape), lower, upper);
    }

    return TensorGenerator{}.uniform(std::move(shape), lower, upper);
}

auto Tensor::normal_random(std::vector<usize> shape, f64 mu, f64 sigma, std::optional<TensorSeed> seed) -> Tensor {
    if (seed.has_value()) {
        return TensorGenerator(*seed).normal(std::move(shape), mu, sigma);
    }

    return TensorGenerator{}.normal(std::move(shape), mu, sigma);
}

auto Tensor::matrix(std::initializer_list<std::initializer_list<f64>> rows) -> Tensor {
    const auto row_count = rows.size();
    const auto col_count = row_count == 0 ? usize{0} : rows.begin()->size();

    auto out = Tensor({row_count, col_count});
    auto *cursor = out.data();

    for (const auto &row : rows) {
        if (row.size() != col_count) {
            throw std::invalid_argument("Tensor::matrix requires all rows to have the same length.");
        }
        cursor = std::ranges::copy(row, cursor).out;
    }

    return out;
}

auto Tensor::rank() const noexcept -> usize {
    return shape_.size();
}

auto Tensor::size() const noexcept -> usize {
    return data_.size();
}

auto Tensor::shape() const noexcept -> std::span<const usize> {
    return shape_;
}

auto Tensor::leg_names() const noexcept -> std::span<const std::string> {
    return leg_names_;
}

auto Tensor::leg_name(usize axis) const -> const std::string & {
    if (axis >= leg_names_.size()) {
        throw std::out_of_range("Tensor leg index exceeds tensor rank.");
    }
    return leg_names_[axis];
}

auto Tensor::data() noexcept -> f64 * {
    return data_.data();
}

auto Tensor::data() const noexcept -> const f64 * {
    return data_.data();
}

auto Tensor::normalized() const -> Tensor {
    auto out = *this;
    out.normalize_in_place();
    return out;
}

auto Tensor::normalize_in_place() -> void {
    if (validity() != TensorValidity::valid) {
        throw std::invalid_argument("normalize_in_place requires a valid tensor.");
    }

    const auto norm = l2_norm(*this);
    if (not std::isfinite(norm) or norm == 0.0) {
        throw std::runtime_error("Cannot normalize a tensor with zero or non-finite L2 norm.");
    }

    const auto inverse_norm = f64{1.0} / norm;
    // Vector * scalar multiplication
    cblas_dscal(as_blas_int(size()), inverse_norm, data(), 1);
}

auto Tensor::add_scalar_in_place(f64 scalar) -> Tensor & {
    if (validity() != TensorValidity::valid) {
        throw std::invalid_argument("add_scalar_in_place requires a valid tensor.");
    }

    for (auto &value : data_) {
        value += scalar;
    }
    return *this;
}

auto Tensor::subtract_scalar_in_place(f64 scalar) -> Tensor & {
    if (validity() != TensorValidity::valid) {
        throw std::invalid_argument("subtract_scalar_in_place requires a valid tensor.");
    }

    for (auto &value : data_) {
        value -= scalar;
    }
    return *this;
}

auto Tensor::multiply_scalar_in_place(f64 scalar) -> Tensor & {
    if (validity() != TensorValidity::valid) {
        throw std::invalid_argument("multiply_scalar_in_place requires a valid tensor.");
    }

    cblas_dscal(as_blas_int(size()), scalar, data(), 1);
    return *this;
}

auto Tensor::divide_scalar_in_place(f64 scalar) -> Tensor & {
    if (validity() != TensorValidity::valid) {
        throw std::invalid_argument("divide_scalar_in_place requires a valid tensor.");
    }
    if (scalar == 0.0) {
        throw std::invalid_argument("divide_scalar_in_place requires a non-zero scalar.");
    }

    cblas_dscal(as_blas_int(size()), f64{1.0} / scalar, data(), 1);
    return *this;
}

auto Tensor::operator+=(f64 scalar) -> Tensor & {
    return add_scalar_in_place(scalar);
}

auto Tensor::operator-=(f64 scalar) -> Tensor & {
    return subtract_scalar_in_place(scalar);
}

auto Tensor::operator*=(f64 scalar) -> Tensor & {
    return multiply_scalar_in_place(scalar);
}

auto Tensor::operator/=(f64 scalar) -> Tensor & {
    return divide_scalar_in_place(scalar);
}

auto Tensor::linear_index(std::span<const usize> indices) const -> usize {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Tensor index rank does not match tensor rank.");
    }
    for (auto i = 0zu; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index exceeds tensor extent.");
        }
    }
    return inner_product(indices, std::span<const usize>{strides_});
}

auto Tensor::operator()(std::span<const usize> indices) -> f64 & {
    return data_[linear_index(indices)];
}

auto Tensor::operator()(std::span<const usize> indices) const -> const f64 & {
    return data_[linear_index(indices)];
}

auto Tensor::indices_from_linear(usize linear_index) const -> std::vector<usize> {
    if (linear_index >= data_.size()) {
        throw std::out_of_range("Linear index exceeds tensor storage.");
    }

    auto indices = std::vector<usize>(shape_.size(), 0);
    for (const auto axis : iota_n(shape_.size())) {
        indices[axis] = linear_index / strides_[axis];
        linear_index %= strides_[axis];
    }

    return indices;
}

auto Tensor::validity() const noexcept -> TensorValidity {
    if (shape_.size() != strides_.size()) {
        return TensorValidity::shape_stride_size_mismatch;
    }

    if (const auto leg_name_validity = leg_names_validity(shape_, leg_names_);
        leg_name_validity != TensorValidity::valid) {
        return leg_name_validity;
    }

    if (shape_.empty()) {
        return data_.size() == 1 ? TensorValidity::valid : TensorValidity::scalar_storage_size_mismatch;
    }

    auto expected_stride = usize{1};
    for (usize axis = shape_.size(); axis > 0; --axis) {
        const auto current_axis = axis - 1;
        if (strides_[current_axis] != expected_stride) {
            return TensorValidity::stride_mismatch;
        }
        expected_stride *= shape_[current_axis];
    }

    return data_.size() == expected_stride ? TensorValidity::valid : TensorValidity::data_size_mismatch;
}

auto Tensor::print(usize precision, bool show_metadata, std::ostream &out) const -> void {
    auto buffer = std::ostringstream{};
    const auto width = element_width(*this, precision);

    if (show_metadata) {
        buffer << "Tensor(rank=" << rank() << ", shape=" << shape_to_string(shape_)
               << ", legs=" << leg_names_to_string(leg_names_) << ")\n";
    }

    if (is_scalar()) {
        buffer << format_value(data_[0], precision);
    } else if (is_vector()) {
        buffer << '[';
        for (usize index = 0; index < size(); ++index) {
            if (index != 0) {
                buffer << ' ';
            }
            buffer << std::setw(static_cast<int>(width)) << format_value(data_[index], precision);
        }
        buffer << ']';
    } else if (is_matrix()) {
        append_matrix(buffer, *this, precision, width);
    } else if (is_tensor3()) {
        for (usize slice = 0; slice < shape_[0]; ++slice) {
            buffer << "slice " << slice << '\n';
            append_matrix(buffer, *this, precision, width, slice, true);
            if (slice + 1 != shape_[0]) {
                buffer << "\n\n";
            }
        }
    } else {
        buffer << "<printing for rank > 3 is not implemented>";
    }

    out << buffer.str() << '\n';
}

auto Tensor::is_scalar() const noexcept -> bool {
    return shape_.empty();
}

auto Tensor::is_trivial() const noexcept -> bool {
    return is_scalar();
}

auto Tensor::is_vector() const noexcept -> bool {
    return rank() == 1;
}

auto Tensor::is_matrix() const noexcept -> bool {
    return rank() == 2;
}

auto Tensor::is_tensor3() const noexcept -> bool {
    return rank() == 3;
}

} // namespace ds_tn
