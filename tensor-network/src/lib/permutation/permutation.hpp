// lib/permutation/permutation.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace ds_tn
{

class Permutation
{
  public:
    Permutation() = default;
    explicit Permutation(std::vector<usize> mapping);
    explicit Permutation(std::initializer_list<usize> mapping);

    template <typename T>
    [[nodiscard]] auto apply(std::span<const T> values) const -> std::vector<T>
    {
        if (values.size() != mapping_.size())
        {
            throw std::runtime_error("Permutation must be applied to a span of the same size.");
        }

        std::vector<T> out(values.size());
        for (auto i = 0zu; i < mapping_.size(); ++i)
        {
            out[mapping_[i]] = values[i];
        }
        return out;
    }

    template <typename T>
    [[nodiscard]] auto apply(const std::vector<T>& values) const -> std::vector<T>
    {
        return apply(std::span<const T>{values});
    }

    template <typename T>
    [[nodiscard]] auto apply_inverse(std::span<const T> values) const -> std::vector<T>
    {
        if (values.size() != mapping_.size())
        {
            throw std::runtime_error("Permutation must be applied to a span of the same size.");
        }

        std::vector<T> out(values.size());
        for (auto i = 0zu; i < mapping_.size(); ++i)
        {
            out[i] = values[mapping_[i]];
        }
        return out;
    }

    template <typename T>
    [[nodiscard]] auto apply_inverse(const std::vector<T>& values) const -> std::vector<T>
    {
        return apply_inverse(std::span<const T>{values});
    }

    [[nodiscard]] auto at(usize index) const -> usize;
    [[nodiscard]] auto operator[](usize index) const noexcept -> usize;
    [[nodiscard]] auto size() const noexcept -> usize;

  private:
    [[nodiscard]] static auto is_valid_mapping(std::span<const usize> mapping) noexcept -> bool;

    const std::vector<usize> mapping_{};
};

[[nodiscard]] auto apply_permutation(const NDArray& array, const Permutation& permutation)
    -> NDArray;

}  // namespace ds_tn
