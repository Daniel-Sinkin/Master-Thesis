// lib/ndarray/ndarray.cpp
#include "ndarray/ndarray.hpp"

#include "ndarray/generator.hpp"
#include "ndarray/stats.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vecLib/cblas_new.h>

namespace ds_tn
{
namespace
{

template <typename T>
    requires requires(T t, T s) {
        { t * s } -> std::convertible_to<T>;
        T{1};
    }
[[nodiscard]] auto product(std::span<const T> values) -> T
{
    return std::accumulate(values.begin(), values.end(), T{1}, std::multiplies<>{});
}

[[nodiscard]] auto as_blas_int(usize value) -> __LAPACK_int
{
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max()))
    {
        throw std::overflow_error("NDArray extent exceeds BLAS integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

[[nodiscard]] auto shape_to_string(std::span<const usize> shape) -> std::string
{
    auto buffer = std::ostringstream{};
    buffer << '[';
    for (usize axis = 0; axis < shape.size(); ++axis)
    {
        if (axis != 0)
        {
            buffer << " x ";
        }
        buffer << shape[axis];
    }
    buffer << ']';
    return buffer.str();
}

[[nodiscard]] auto format_value(f64 value, usize precision) -> std::string
{
    auto buffer = std::ostringstream{};
    buffer << std::fixed << std::setprecision(static_cast<int>(precision)) << value;
    return buffer.str();
}

[[nodiscard]] auto element_width(const NDArray& array, usize precision) -> usize
{
    auto width = usize{0};
    for (usize index = 0; index < array.size(); ++index)
    {
        width = std::max(width, format_value(array.data()[index], precision).size());
    }
    return width;
}

auto append_matrix(
    std::ostringstream& buffer,
    const NDArray& array,
    usize precision,
    usize width,
    usize slice = 0,
    bool rank3 = false
) -> void
{
    const auto rows = rank3 ? array.shape(1) : array.shape(0);
    const auto cols = rank3 ? array.shape(2) : array.shape(1);

    for (usize row = 0; row < rows; ++row)
    {
        buffer << '[';
        for (usize col = 0; col < cols; ++col)
        {
            if (col != 0)
            {
                buffer << ' ';
            }

            const auto value = [&]() -> f64
            {
                if (rank3)
                {
                    return array(std::array<usize, 3>{slice, row, col});
                }
                return array(std::array<usize, 2>{row, col});
            }();

            buffer << std::setw(static_cast<int>(width)) << format_value(value, precision);
        }
        buffer << ']';
        if (row + 1 != rows)
        {
            buffer << '\n';
        }
    }
}

}  // namespace

NDArray::NDArray() : NDArray(std::vector<usize>{})
{
}

NDArray::NDArray(std::vector<usize> shape) : shape_(std::move(shape)), strides_(shape_.size(), 1)
{
    initialize_storage();
}

auto NDArray::initialize_storage() -> void
{
    std::exclusive_scan(
        shape_.rbegin(), shape_.rend(), strides_.rbegin(), usize{1}, std::multiplies<>{}
    );

    if (shape_.empty())
    {
        data_.resize(1);
        return;
    }

    data_.resize(strides_.front() * shape_.front());
}

auto NDArray::scalar(f64 value) -> NDArray
{
    auto out = NDArray{};
    out.data_[0] = value;
    return out;
}

auto NDArray::diag(const NDArray& vector) -> NDArray
{
    if (vector.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("NDArray::diag requires a valid NDArray.");
    }
    if (!vector.is_vector())
    {
        throw std::invalid_argument("NDArray::diag requires a rank-1 NDArray.");
    }

    auto out = NDArray({vector.shape(0), vector.shape(0)});
    for (auto i = 0zu; i < vector.shape(0); ++i)
    {
        out(i, i) = vector(i);
    }
    return out;
}

auto NDArray::iota(usize size) -> NDArray
{
    auto out = NDArray({size});
    for (auto i = 0zu; i < size; ++i)
    {
        out.data(i) = static_cast<f64>(i);
    }
    return out;
}

auto NDArray::vector(std::initializer_list<f64> values) -> NDArray
{
    auto out = NDArray({values.size()});
    std::ranges::copy(values, out.data_.begin());
    return out;
}

auto NDArray::random(
    std::vector<usize> shape, RandomOptions options, std::optional<NDArraySeed> seed
) -> NDArray
{
    auto generator = seed.has_value() ? NDArrayGenerator(*seed) : NDArrayGenerator{};
    return generator.generate(std::move(shape), std::move(options));
}

auto NDArray::random_uniform(
    std::vector<usize> shape, RandomUniformOptions options, std::optional<NDArraySeed> seed
) -> NDArray
{
    return NDArray::random(std::move(shape), options, seed);
}

auto NDArray::random_normal(
    std::vector<usize> shape, RandomNormalOptions options, std::optional<NDArraySeed> seed
) -> NDArray
{
    return NDArray::random(std::move(shape), options, seed);
}

auto NDArray::zeros_like(const NDArray& other) -> NDArray
{
    return NDArray(std::vector<usize>{other.shape().begin(), other.shape().end()});
}

auto NDArray::reshape(const NDArray& array, std::span<const usize> new_shape) -> NDArray
{
    if (array.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("NDArray::reshape requires a valid NDArray.");
    }
    if (product<usize>(new_shape) != array.size())
    {
        throw std::invalid_argument("NDArray::reshape requires product(new_shape) == array.size().");
    }

    auto out = NDArray{std::vector<usize>{new_shape.begin(), new_shape.end()}};
    std::ranges::copy(array.data(), array.data() + array.size(), out.data());
    return out;
}

auto NDArray::reshape(const NDArray& array, std::initializer_list<usize> new_shape) -> NDArray
{
    return NDArray::reshape(array, std::span<const usize>{new_shape.begin(), new_shape.size()});
}

auto NDArray::squeeze(const NDArray& array) -> NDArray
{
    if (array.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("NDArray::squeeze requires a valid NDArray.");
    }

    auto squeezed_shape = std::vector<usize>{};
    squeezed_shape.reserve(array.rank());
    for (const auto extent : array.shape())
    {
        if (extent != 1)
        {
            squeezed_shape.push_back(extent);
        }
    }

    return NDArray::reshape(array, std::span<const usize>{squeezed_shape});
}

auto truncate_cols(const NDArray& mat, usize cols) -> NDArray
{
    if (mat.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("truncate_cols requires a valid NDArray.");
    }
    if (!mat.is_matrix())
    {
        throw std::invalid_argument("truncate_cols requires a rank-2 NDArray.");
    }
    if (cols > mat.shape_[1])
    {
        throw std::invalid_argument("truncate_cols requires cols <= mat.cols.");
    }

    auto out = NDArray({mat.shape_[0], cols});
    for (auto row = 0zu; row < mat.shape_[0]; ++row)
    {
        for (auto col = 0zu; col < cols; ++col)
        {
            out(row, col) = mat(row, col);
        }
    }
    return out;
}

auto truncate_rows(const NDArray& mat, usize rows) -> NDArray
{
    if (mat.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("truncate_rows requires a valid NDArray.");
    }
    if (!mat.is_matrix())
    {
        throw std::invalid_argument("truncate_rows requires a rank-2 NDArray.");
    }
    if (rows > mat.shape_[0])
    {
        throw std::invalid_argument("truncate_rows requires rows <= mat.rows.");
    }

    auto out = mat;
    out.shape_[0] = rows;
    out.data_.resize(rows * mat.shape_[1]);
    return out;
}

auto transpose_matrix(const NDArray& matrix) -> NDArray
{
    if (matrix.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("transpose_matrix requires a valid NDArray.");
    }
    if (!matrix.is_matrix())
    {
        throw std::invalid_argument("transpose_matrix requires a rank-2 NDArray.");
    }

    auto out = NDArray({matrix.shape(1), matrix.shape(0)});
    for (auto row = 0zu; row < matrix.shape(0); ++row)
    {
        for (auto col = 0zu; col < matrix.shape(1); ++col)
        {
            out(col, row) = matrix(row, col);
        }
    }
    return out;
}

auto NDArray::matrix(std::initializer_list<std::initializer_list<f64>> rows) -> NDArray
{
    const auto row_count = rows.size();
    const auto col_count = row_count == 0 ? usize{0} : rows.begin()->size();

    auto out = NDArray({row_count, col_count});
    auto* cursor = out.data();

    for (const auto& row : rows)
    {
        if (row.size() != col_count)
        {
            throw std::invalid_argument(
                "NDArray::matrix requires all rows to have the same length."
            );
        }
        cursor = std::ranges::copy(row, cursor).out;
    }

    return out;
}

auto NDArray::rank3(
    std::initializer_list<std::initializer_list<std::initializer_list<f64>>> slices
) -> NDArray
{
    const auto slice_count = slices.size();
    const auto row_count = slice_count == 0 ? usize{0} : slices.begin()->size();
    const auto col_count =
        row_count == 0 ? usize{0} : slices.begin()->begin()->size();

    auto out = NDArray({slice_count, row_count, col_count});
    auto* cursor = out.data();

    for (const auto& slice : slices)
    {
        if (slice.size() != row_count)
        {
                throw std::invalid_argument(
                    "NDArray::rank3 requires all slices to have the same number of rows."
                );
            }

        for (const auto& row : slice)
        {
            if (row.size() != col_count)
            {
                throw std::invalid_argument(
                    "NDArray::rank3 requires all rows to have the same length."
                );
            }
            cursor = std::ranges::copy(row, cursor).out;
        }
    }

    return out;
}

auto NDArray::same_shape(const NDArray& lhs, const NDArray& rhs) noexcept -> bool
{
    return std::ranges::equal(lhs.shape_, rhs.shape_);
}

auto NDArray::rank() const noexcept -> usize
{
    return shape_.size();
}

auto NDArray::size() const noexcept -> usize
{
    return data_.size();
}

auto NDArray::shape() const noexcept -> std::span<const usize>
{
    return shape_;
}

auto NDArray::shape(usize axis) const -> usize
{
    if (axis >= shape_.size())
    {
        throw std::out_of_range("NDArray shape axis exceeds array rank.");
    }
    return shape_[axis];
}

auto NDArray::data() noexcept -> f64*
{
    return data_.data();
}

auto NDArray::data() const noexcept -> const f64*
{
    return data_.data();
}

auto NDArray::data(usize linear_index) noexcept -> f64&
{
    return data_[linear_index];
}

auto NDArray::data(usize linear_index) const noexcept -> const f64&
{
    return data_[linear_index];
}

auto NDArray::same_shape(const NDArray& other) const noexcept -> bool
{
    return NDArray::same_shape(*this, other);
}

auto NDArray::l2_norm() const -> f64
{
    return ds_tn::l2_norm(*this);
}

auto NDArray::normalized() const -> NDArray
{
    auto out = *this;
    out.normalize();
    return out;
}

auto NDArray::zeros_like() const -> NDArray
{
    return NDArray::zeros_like(*this);
}

auto NDArray::diag() const -> NDArray
{
    return NDArray::diag(*this);
}

auto NDArray::reshape(std::span<const usize> new_shape) const -> NDArray
{
    return NDArray::reshape(*this, new_shape);
}

auto NDArray::reshape(std::initializer_list<usize> new_shape) const -> NDArray
{
    return NDArray::reshape(*this, new_shape);
}

auto NDArray::squeeze() const -> NDArray
{
    return NDArray::squeeze(*this);
}

auto NDArray::format_metadata() const -> std::string
{
    return "NDArray(shape=" + shape_to_string(shape()) + ")";
}

auto NDArray::normalize() -> void
{
    if (validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("normalize requires a valid NDArray.");
    }

    const auto norm = l2_norm();
    if (not std::isfinite(norm) or norm == 0.0)
    {
        throw std::runtime_error("Cannot normalize an NDArray with zero or non-finite L2 norm.");
    }

    cblas_dscal(as_blas_int(size()), f64{1.0} / norm, data(), 1);
}

auto NDArray::add_scalar(f64 scalar) -> NDArray&
{
    if (validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("add_scalar requires a valid NDArray.");
    }

    for (auto& value : data_)
    {
        value += scalar;
    }
    return *this;
}

auto NDArray::subtract_scalar(f64 scalar) -> NDArray&
{
    if (validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("subtract_scalar requires a valid NDArray.");
    }

    for (auto& value : data_)
    {
        value -= scalar;
    }
    return *this;
}

auto NDArray::multiply_scalar(f64 scalar) -> NDArray&
{
    if (validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("multiply_scalar requires a valid NDArray.");
    }

    cblas_dscal(as_blas_int(size()), scalar, data(), 1);
    return *this;
}

auto NDArray::divide_scalar(f64 scalar) -> NDArray&
{
    if (validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("divide_scalar requires a valid NDArray.");
    }
    if (scalar == 0.0)
    {
        throw std::invalid_argument("divide_scalar requires a non-zero scalar.");
    }

    cblas_dscal(as_blas_int(size()), f64{1.0} / scalar, data(), 1);
    return *this;
}

auto NDArray::operator+=(const NDArray& rhs) -> NDArray&
{
    if (validity() != NDArrayValidity::valid or rhs.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("NDArray array addition requires valid NDArrays.");
    }
    if (shape_ != rhs.shape_)
    {
        throw std::invalid_argument("NDArray array addition requires matching shapes.");
    }

    cblas_daxpy(as_blas_int(size()), 1.0, rhs.data(), 1, data(), 1);
    return *this;
}

auto NDArray::operator-=(const NDArray& rhs) -> NDArray&
{
    if (validity() != NDArrayValidity::valid or rhs.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("NDArray array subtraction requires valid NDArrays.");
    }
    if (shape_ != rhs.shape_)
    {
        throw std::invalid_argument("NDArray array subtraction requires matching shapes.");
    }

    cblas_daxpy(as_blas_int(size()), -1.0, rhs.data(), 1, data(), 1);
    return *this;
}

auto NDArray::operator+=(f64 scalar) -> NDArray&
{
    return add_scalar(scalar);
}

auto NDArray::operator-=(f64 scalar) -> NDArray&
{
    return subtract_scalar(scalar);
}

auto NDArray::operator*=(f64 scalar) -> NDArray&
{
    return multiply_scalar(scalar);
}

auto NDArray::operator/=(f64 scalar) -> NDArray&
{
    return divide_scalar(scalar);
}

auto NDArray::linear_index(std::span<const usize> indices) const -> usize
{
    if (indices.size() != shape_.size())
    {
        throw std::invalid_argument("NDArray index rank does not match array rank.");
    }
    for (auto i = 0zu; i < indices.size(); ++i)
    {
        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range("NDArray index exceeds array extent.");
        }
    }

    return inner_product(indices, std::span<const usize>{strides_});
}

auto NDArray::operator()(std::span<const usize> indices) -> f64&
{
    return data_[linear_index(indices)];
}

auto NDArray::operator()(std::span<const usize> indices) const -> const f64&
{
    return data_[linear_index(indices)];
}

auto NDArray::indices_from_linear(usize linear_index) const -> std::vector<usize>
{
    if (linear_index >= data_.size())
    {
        throw std::out_of_range("Linear index exceeds NDArray storage.");
    }

    auto indices = std::vector<usize>(shape_.size(), 0);
    for (auto axis = 0zu; axis < shape_.size(); ++axis)
    {
        indices[axis] = linear_index / strides_[axis];
        linear_index %= strides_[axis];
    }

    return indices;
}

auto NDArray::validity() const noexcept -> NDArrayValidity
{
    if (shape_.size() != strides_.size())
    {
        return NDArrayValidity::shape_stride_size_mismatch;
    }

    if (shape_.empty())
    {
        return data_.size() == 1 ? NDArrayValidity::valid
                                 : NDArrayValidity::scalar_storage_size_mismatch;
    }

    auto expected_stride = usize{1};
    for (usize axis = shape_.size(); axis > 0; --axis)
    {
        const auto current_axis = axis - 1;
        if (strides_[current_axis] != expected_stride)
        {
            return NDArrayValidity::stride_mismatch;
        }
        expected_stride *= shape_[current_axis];
    }

    return data_.size() == expected_stride ? NDArrayValidity::valid
                                           : NDArrayValidity::data_size_mismatch;
}

auto NDArray::print(usize precision, bool show_metadata, std::ostream& out) const -> void
{
    auto buffer = std::ostringstream{};
    const auto width = element_width(*this, precision);

    if (show_metadata)
    {
        buffer << "NDArray(rank=" << rank() << ", shape=" << shape_to_string(shape_) << ")\n";
    }

    if (is_scalar())
    {
        buffer << format_value(data_[0], precision);
    }
    else if (is_vector())
    {
        buffer << '[';
        for (usize index = 0; index < size(); ++index)
        {
            if (index != 0)
            {
                buffer << ' ';
            }
            buffer << std::setw(static_cast<int>(width)) << format_value(data_[index], precision);
        }
        buffer << ']';
    }
    else if (is_matrix())
    {
        append_matrix(buffer, *this, precision, width);
    }
    else if (is_tensor3())
    {
        for (usize slice = 0; slice < shape_[0]; ++slice)
        {
            buffer << "slice " << slice << '\n';
            append_matrix(buffer, *this, precision, width, slice, true);
            if (slice + 1 != shape_[0])
            {
                buffer << "\n\n";
            }
        }
    }
    else
    {
        buffer << "<printing for rank > 3 is not implemented>";
    }

    out << buffer.str() << '\n';
}

auto NDArray::is_scalar() const noexcept -> bool
{
    return shape_.empty();
}

auto NDArray::is_trivial() const noexcept -> bool
{
    return is_scalar();
}

auto NDArray::is_vector() const noexcept -> bool
{
    return rank() == 1;
}

auto NDArray::is_matrix() const noexcept -> bool
{
    return rank() == 2;
}

auto NDArray::is_tensor3() const noexcept -> bool
{
    return rank() == 3;
}

auto operator+(NDArray lhs, const NDArray& rhs) -> NDArray
{
    lhs += rhs;
    return lhs;
}

}  // namespace ds_tn
