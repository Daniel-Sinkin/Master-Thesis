#include "tensor.hpp"

#include <array>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>

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

auto append_matrix(std::ostringstream &buffer, const Tensor &tensor, usize slice = 0, bool rank3 = false) -> void {
    const auto rows = rank3 ? tensor.shape()[1] : tensor.shape()[0];
    const auto cols = rank3 ? tensor.shape()[2] : tensor.shape()[1];

    for (usize row = 0; row < rows; ++row) {
        buffer << '[';
        for (usize col = 0; col < cols; ++col) {
            if (col != 0) {
                buffer << ' ';
            }

            if (rank3) {
                buffer << tensor(std::array<usize, 3>{slice, row, col});
            } else {
                buffer << tensor(std::array<usize, 2>{row, col});
            }
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
      strides_(shape_.size(), 1) {
    std::exclusive_scan(shape_.rbegin(), shape_.rend(), strides_.rbegin(), usize{1}, std::multiplies<>{});

    if (shape_.empty()) {
        data_.resize(1);
        return;
    }

    data_.resize(strides_.front() * shape_.front());
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

auto Tensor::data() noexcept -> f64 * {
    return data_.data();
}

auto Tensor::data() const noexcept -> const f64 * {
    return data_.data();
}

auto Tensor::linear_index(std::span<const usize> indices) const -> usize {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Tensor index rank does not match tensor rank.");
    }

    for (const auto axis : std::views::iota(usize{0}, indices.size())) {
        if (indices[axis] >= shape_[axis]) {
            throw std::out_of_range("Tensor index exceeds tensor extent.");
        }
    }

    return std::ranges::fold_left(
        std::views::iota(usize{0}, indices.size()) |
            std::views::transform([&](usize axis) { return indices[axis] * strides_[axis]; }),
        usize{0},
        std::plus<>{});
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
    for (const auto axis : std::views::iota(usize{0}, shape_.size())) {
        indices[axis] = linear_index / strides_[axis];
        linear_index %= strides_[axis];
    }

    return indices;
}

auto Tensor::print(usize precision, bool show_metadata, std::ostream &out) const -> void {
    auto buffer = std::ostringstream{};
    buffer << std::fixed << std::setprecision(static_cast<int>(precision));

    if (show_metadata) {
        buffer << "Tensor(rank=" << rank() << ", shape=" << shape_to_string(shape_) << ")\n";
    }

    if (is_scalar()) {
        buffer << data_[0];
    } else if (is_vector()) {
        buffer << '[';
        for (usize index = 0; index < size(); ++index) {
            if (index != 0) {
                buffer << ' ';
            }
            buffer << data_[index];
        }
        buffer << ']';
    } else if (is_matrix()) {
        append_matrix(buffer, *this);
    } else if (is_tensor3()) {
        for (usize slice = 0; slice < shape_[0]; ++slice) {
            buffer << "slice " << slice << '\n';
            append_matrix(buffer, *this, slice, true);
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
