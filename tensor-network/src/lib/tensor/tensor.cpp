// lib/tensor/tensor.cpp
#include "tensor/tensor.hpp"

#include <atomic>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ds_tn
{
namespace
{

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

[[nodiscard]] auto leg_names_to_string(std::span<const std::string> leg_names) -> std::string
{
    auto buffer = std::ostringstream{};
    buffer << '[';
    for (usize axis = 0; axis < leg_names.size(); ++axis)
    {
        if (axis != 0)
        {
            buffer << ", ";
        }
        buffer << leg_names[axis];
    }
    buffer << ']';
    return buffer.str();
}

[[nodiscard]] auto make_default_leg_names(usize rank) -> std::vector<std::string>
{
    static auto next_tensor_id = std::atomic<u64>{0};

    const auto tensor_id = next_tensor_id.fetch_add(1, std::memory_order_relaxed);
    auto names = std::vector<std::string>{};
    names.reserve(rank);

    for (usize axis = 0; axis < rank; ++axis)
    {
        names.push_back("t" + std::to_string(tensor_id) + "_l" + std::to_string(axis));
    }

    return names;
}

[[nodiscard]] auto
leg_names_validity(std::span<const usize> shape, std::span<const std::string> leg_names)
    -> TensorValidity
{
    if (leg_names.size() != shape.size())
    {
        return TensorValidity::shape_leg_name_size_mismatch;
    }

    for (usize axis = 0; axis < leg_names.size(); ++axis)
    {
        if (leg_names[axis].empty())
        {
            return TensorValidity::empty_leg_name;
        }

        for (usize other_axis = axis + 1; other_axis < leg_names.size(); ++other_axis)
        {
            if (leg_names[axis] == leg_names[other_axis])
            {
                return TensorValidity::duplicate_leg_name;
            }
        }
    }

    return TensorValidity::valid;
}

}  // namespace

Tensor::Tensor(NDArray array)
    : values_(std::move(array)), leg_names_(make_default_leg_names(values_.rank()))
{
}

Tensor::Tensor(NDArray array, std::vector<std::string> leg_names)
    : values_(std::move(array)), leg_names_(std::move(leg_names))
{
    if (leg_names_validity(values_.shape(), leg_names_) != TensorValidity::valid)
    {
        throw std::invalid_argument(
            "Tensor leg names must match rank, be non-empty, and be unique."
        );
    }
}

Tensor::Tensor(NDArray array, std::span<const std::string> leg_names)
    : values_(std::move(array)), leg_names_(leg_names.begin(), leg_names.end())
{
    if (leg_names_validity(values_.shape(), leg_names_) != TensorValidity::valid)
    {
        throw std::invalid_argument(
            "Tensor leg names must match rank, be non-empty, and be unique."
        );
    }
}

Tensor::Tensor(NDArray array, std::initializer_list<std::string> leg_names)
    : Tensor(std::move(array), std::span<const std::string>{leg_names.begin(), leg_names.size()})
{
}

Tensor::Tensor(std::vector<usize> shape) : Tensor(NDArray(std::move(shape)))
{
}

Tensor::Tensor(std::vector<usize> shape, std::vector<std::string> leg_names)
    : Tensor(NDArray(std::move(shape)), std::move(leg_names))
{
}

Tensor::Tensor(std::vector<usize> shape, std::span<const std::string> leg_names)
    : Tensor(NDArray(std::move(shape)), leg_names)
{
}

Tensor::Tensor(std::vector<usize> shape, std::initializer_list<std::string> leg_names)
    : Tensor(NDArray(std::move(shape)), leg_names)
{
}

auto Tensor::scalar(f64 value) -> Tensor
{
    return Tensor{NDArray::scalar(value)};
}

auto Tensor::diag(const Tensor& vector) -> Tensor
{
    if (vector.validity() != TensorValidity::valid)
    {
        throw std::invalid_argument("Tensor::diag requires a valid Tensor.");
    }
    if (!vector.is_vector())
    {
        throw std::invalid_argument("Tensor::diag requires a rank-1 Tensor.");
    }

    const auto base = vector.leg_name(0);
    return Tensor{
        NDArray::diag(vector.array()),
        {
            base + "_row",
            base + "_col",
        },
    };
}

auto Tensor::iota(usize size) -> Tensor
{
    return Tensor{NDArray::iota(size)};
}

auto Tensor::vector(std::initializer_list<f64> values) -> Tensor
{
    return Tensor{NDArray::vector(values)};
}

auto Tensor::random(std::vector<usize> shape, RandomOptions options, std::optional<TensorSeed> seed)
    -> Tensor
{
    return Tensor{NDArray::random(std::move(shape), std::move(options), seed)};
}

auto Tensor::random_uniform(
    std::vector<usize> shape, RandomUniformOptions options, std::optional<TensorSeed> seed
) -> Tensor
{
    return Tensor::random(std::move(shape), options, seed);
}

auto Tensor::random_normal(
    std::vector<usize> shape, RandomNormalOptions options, std::optional<TensorSeed> seed
) -> Tensor
{
    return Tensor::random(std::move(shape), options, seed);
}

auto Tensor::matrix(std::initializer_list<std::initializer_list<f64>> rows) -> Tensor
{
    return Tensor{NDArray::matrix(rows)};
}

auto Tensor::rank3(
    std::initializer_list<std::initializer_list<std::initializer_list<f64>>> slices
) -> Tensor
{
    return Tensor{NDArray::rank3(slices)};
}

auto Tensor::rank() const noexcept -> usize
{
    return values_.rank();
}

auto Tensor::size() const noexcept -> usize
{
    return values_.size();
}

auto Tensor::shape() const noexcept -> std::span<const usize>
{
    return values_.shape();
}

auto Tensor::shape(usize axis) const -> usize
{
    return values_.shape(axis);
}

auto Tensor::leg_names() const noexcept -> std::span<const std::string>
{
    return leg_names_;
}

auto Tensor::leg_name(usize axis) const -> const std::string&
{
    if (axis >= leg_names_.size())
    {
        throw std::out_of_range("Tensor leg index exceeds tensor rank.");
    }
    return leg_names_[axis];
}

auto Tensor::array() noexcept -> NDArray&
{
    return values_;
}

auto Tensor::array() const noexcept -> const NDArray&
{
    return values_;
}

auto Tensor::data() noexcept -> f64*
{
    return values_.data();
}

auto Tensor::data() const noexcept -> const f64*
{
    return values_.data();
}

auto Tensor::operator()(std::span<const usize> indices) -> f64&
{
    return values_(indices);
}

auto Tensor::operator()(std::span<const usize> indices) const -> const f64&
{
    return values_(indices);
}

auto Tensor::indices_from_linear(usize linear_index) const -> std::vector<usize>
{
    return values_.indices_from_linear(linear_index);
}

auto Tensor::validity() const noexcept -> TensorValidity
{
    if (values_.validity() != NDArrayValidity::valid)
    {
        return TensorValidity::array_invalid;
    }

    return leg_names_validity(values_.shape(), leg_names_);
}

auto Tensor::diag() const -> Tensor
{
    return Tensor::diag(*this);
}

auto Tensor::format_metadata() const -> std::string
{
    return "Tensor(shape=" + shape_to_string(shape()) + ", legs=" + leg_names_to_string(leg_names_)
           + ")";
}

auto Tensor::print_metadata(std::ostream& out) const -> void
{
    out << format_metadata() << '\n';
}

auto Tensor::print(usize precision, bool show_metadata, std::ostream& out) const -> void
{
    if (show_metadata)
    {
        print_metadata(out);
    }

    values_.print(precision, false, out);
}

auto Tensor::is_scalar() const noexcept -> bool
{
    return values_.is_scalar();
}

auto Tensor::is_trivial() const noexcept -> bool
{
    return values_.is_trivial();
}

auto Tensor::is_vector() const noexcept -> bool
{
    return values_.is_vector();
}

auto Tensor::is_matrix() const noexcept -> bool
{
    return values_.is_matrix();
}

auto Tensor::is_tensor3() const noexcept -> bool
{
    return values_.is_tensor3();
}

}  // namespace ds_tn
