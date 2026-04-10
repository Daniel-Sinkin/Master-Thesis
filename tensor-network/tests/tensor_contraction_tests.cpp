#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "tensor/compare.hpp"
#include "tensor/contraction.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto
shape_over_selected_axes(std::span<const usize> shape, std::span<const usize> indices)
    -> std::vector<usize>
{
    auto out = std::vector<usize>{};
    out.reserve(indices.size());
    for (const auto axis : indices)
    {
        out.push_back(shape[axis]);
    }
    return out;
}

[[nodiscard]] auto contract_reference(const Tensor& left, const Tensor& right) -> Tensor
{
    const auto partition = partition_indices(left, right);
    auto out = Tensor{contraction_output_shape(left, right), contraction_output_legs(left, right)};
    auto left_indices = std::vector<usize>(left.rank());
    auto right_indices = std::vector<usize>(right.rank());
    const auto shared_shape = shape_over_selected_axes(left.shape(), partition.left_shared);

    for (auto out_linear = 0zu; out_linear < out.size(); ++out_linear)
    {
        const auto out_indices = out.indices_from_linear(out_linear);

        for (auto i = 0zu; i < partition.left_not_shared.size(); ++i)
        {
            left_indices[partition.left_not_shared[i]] = out_indices[i];
        }
        for (auto i = 0zu; i < partition.right_not_shared.size(); ++i)
        {
            right_indices[partition.right_not_shared[i]] =
                out_indices[partition.left_not_shared.size() + i];
        }

        auto sum = 0.0;
        if (shared_shape.empty())
        {
            sum = left(std::span<const usize>{left_indices})
                  * right(std::span<const usize>{right_indices});
        }
        else
        {
            const auto shared_indexer = NDArray{shared_shape};
            for (auto shared_linear = 0zu; shared_linear < shared_indexer.size(); ++shared_linear)
            {
                const auto shared_indices = shared_indexer.indices_from_linear(shared_linear);
                for (auto i = 0zu; i < partition.left_shared.size(); ++i)
                {
                    left_indices[partition.left_shared[i]] = shared_indices[i];
                    right_indices[partition.right_shared[i]] = shared_indices[i];
                }

                sum += left(std::span<const usize>{left_indices})
                       * right(std::span<const usize>{right_indices});
            }
        }

        out.array().data(out_linear) = sum;
    }

    return out;
}

auto fill_deterministic(Tensor& tensor, f64 base) -> void
{
    for (auto i = 0zu; i < tensor.size(); ++i)
    {
        tensor.array().data(i) =
            base + static_cast<f64>((i * 17zu + 3zu) % 23zu) / 7.0 - static_cast<f64>(i % 5zu);
    }
}

[[nodiscard]] auto make_name(std::string_view prefix, usize idx) -> std::string
{
    return std::string{prefix} + '_' + (idx < 10zu ? "0" : "") + std::to_string(idx);
}

[[nodiscard]] auto interleave_names(
    std::span<const std::string> first,
    std::span<const std::string> second,
    bool take_first_first
) -> std::vector<std::string>
{
    auto out = std::vector<std::string>{};
    out.reserve(first.size() + second.size());

    auto i = 0zu;
    auto j = 0zu;
    auto take_first = take_first_first;

    while (i < first.size() or j < second.size())
    {
        if ((take_first and i < first.size()) or j >= second.size())
        {
            out.push_back(first[i]);
            ++i;
        }
        else
        {
            out.push_back(second[j]);
            ++j;
        }
        take_first = !take_first;
    }

    return out;
}

struct GeneratedCase
{
    Tensor left{};
    Tensor right{};
    std::vector<std::string> expected_legs{};
    std::vector<usize> expected_shape{};
};

[[nodiscard]] auto generate_rank_case(usize rank) -> GeneratedCase
{
    const auto shared_count = rank / 2zu;
    const auto left_only_count = rank - shared_count;
    const auto right_only_count = rank - shared_count;

    auto shared_names = std::vector<std::string>{};
    auto left_only_names = std::vector<std::string>{};
    auto right_only_names = std::vector<std::string>{};
    shared_names.reserve(shared_count);
    left_only_names.reserve(left_only_count);
    right_only_names.reserve(right_only_count);

    for (auto i = 0zu; i < shared_count; ++i)
    {
        shared_names.push_back(make_name("shared", i));
    }
    for (auto i = 0zu; i < left_only_count; ++i)
    {
        left_only_names.push_back(make_name("left", i));
    }
    for (auto i = 0zu; i < right_only_count; ++i)
    {
        right_only_names.push_back(make_name("right", i));
    }

    const auto shared_names_reversed =
        std::vector<std::string>{shared_names.rbegin(), shared_names.rend()};
    const auto left_names = interleave_names(shared_names_reversed, left_only_names, rank % 2zu == 0zu);
    const auto right_names = interleave_names(shared_names, right_only_names, rank % 2zu != 0zu);

    auto all_names = std::vector<std::string>{};
    all_names.reserve(shared_names.size() + left_only_names.size() + right_only_names.size());
    all_names.insert(all_names.end(), shared_names.begin(), shared_names.end());
    all_names.insert(all_names.end(), left_only_names.begin(), left_only_names.end());
    all_names.insert(all_names.end(), right_only_names.begin(), right_only_names.end());

    const auto extent_for_name = [&](std::string_view name) -> usize
    {
        const auto it = std::ranges::find(all_names, name);
        REQUIRE(it != all_names.end());
        const auto idx = static_cast<usize>(std::distance(all_names.begin(), it));
        return idx < 6zu and (idx + rank) % 3zu != 0zu ? 2zu : 1zu;
    };

    const auto shape_from_names = [&](std::span<const std::string> names) -> std::vector<usize>
    {
        auto shape = std::vector<usize>{};
        shape.reserve(names.size());
        for (const auto& name : names)
        {
            shape.push_back(extent_for_name(name));
        }
        return shape;
    };

    auto left = Tensor{shape_from_names(left_names), std::span<const std::string>{left_names}};
    auto right = Tensor{shape_from_names(right_names), std::span<const std::string>{right_names}};
    fill_deterministic(left, 3.0);
    fill_deterministic(right, -5.0);

    auto expected_legs = std::vector<std::string>{};
    expected_legs.reserve(left_only_names.size() + right_only_names.size());
    for (const auto& name : left_names)
    {
        if (!std::ranges::contains(shared_names, name))
        {
            expected_legs.push_back(name);
        }
    }
    for (const auto& name : right_names)
    {
        if (!std::ranges::contains(shared_names, name))
        {
            expected_legs.push_back(name);
        }
    }

    auto expected_shape = std::vector<usize>{};
    expected_shape.reserve(expected_legs.size());
    for (const auto& name : expected_legs)
    {
        expected_shape.push_back(extent_for_name(name));
    }

    return {
        .left = std::move(left),
        .right = std::move(right),
        .expected_legs = std::move(expected_legs),
        .expected_shape = std::move(expected_shape),
    };
}

}  // namespace

TEST_CASE("partition_indices separates shared and unshared legs with stable ordering", "[tensor]")
{
    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});

    const auto partition = partition_indices(left, right);

    REQUIRE(partition.left_not_shared == std::vector<usize>{2, 3});
    REQUIRE(partition.left_shared == std::vector<usize>{1, 0});
    REQUIRE(partition.right_shared == std::vector<usize>{2, 3});
    REQUIRE(partition.right_not_shared == std::vector<usize>{0, 1});
}

TEST_CASE("contraction_output_shape and legs preserve left then right ordering", "[tensor]")
{
    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 3, 2}, {"c", "d", "i", "j"});

    const std::array<std::string, 4> expected_legs{"a", "b", "c", "d"};
    const std::array<usize, 4> expected_shape{5, 7, 11, 13};

    REQUIRE(std::ranges::equal(
        contraction_output_shape(left, right),
        std::span<const usize>{expected_shape}
    ));
    REQUIRE(std::ranges::equal(
        contraction_output_legs(left, right),
        std::span<const std::string>{expected_legs}
    ));
}

TEST_CASE("contraction_output_shape rejects mismatched shared extents", "[tensor]")
{
    const auto left = Tensor({2, 3, 5, 7}, {"j", "i", "a", "b"});
    const auto right = Tensor({11, 13, 4, 2}, {"c", "d", "i", "j"});

    REQUIRE_THROWS_AS(contraction_output_shape(left, right), std::invalid_argument);
    REQUIRE_THROWS_AS(contract(left, right), std::invalid_argument);
}

TEST_CASE("contract matches matrix-matrix multiplication as a special case", "[tensor]")
{
    auto left = Tensor({2, 3}, {"row", "shared"});
    auto right = Tensor({3, 4}, {"shared", "col"});
    fill_deterministic(left, 1.0);
    fill_deterministic(right, -2.0);

    const auto result = contract(left, right);
    const auto expected = matrix_matrix_product(left.array(), right.array());
    const std::array<std::string, 2> expected_legs{"row", "col"};

    REQUIRE(result.is_matrix());
    REQUIRE(std::ranges::equal(result.leg_names(), std::span<const std::string>{expected_legs}));
    REQUIRE(close_per_element(result.array(), expected));
}

TEST_CASE("contract matches matrix-vector multiplication as a special case", "[tensor]")
{
    auto matrix = Tensor({3, 4}, {"row", "shared"});
    auto vector = Tensor({4}, {"shared"});
    fill_deterministic(matrix, 2.5);
    fill_deterministic(vector, -1.0);

    const auto result = contract(matrix, vector);
    const auto expected = matrix_vector_product(matrix.array(), vector.array());
    const std::array<std::string, 1> expected_legs{"row"};

    REQUIRE(result.is_vector());
    REQUIRE(std::ranges::equal(result.leg_names(), std::span<const std::string>{expected_legs}));
    REQUIRE(close_per_element(result.array(), expected));
}

TEST_CASE("contract matches dot product as a special case", "[tensor]")
{
    auto lhs = Tensor({5}, {"shared"});
    auto rhs = Tensor({5}, {"shared"});
    fill_deterministic(lhs, 0.5);
    fill_deterministic(rhs, -3.0);

    const auto result = contract(lhs, rhs);
    const auto expected = dot_product(lhs.array(), rhs.array());

    REQUIRE(result.is_scalar());
    REQUIRE(result.leg_names().empty());
    REQUIRE(result() == Catch::Approx(expected));
}

TEST_CASE("contract handles tensors with no shared legs as an outer product", "[tensor]")
{
    auto left = Tensor({2, 3}, {"a", "b"});
    auto right = Tensor({4, 1, 2}, {"c", "d", "e"});
    fill_deterministic(left, 4.0);
    fill_deterministic(right, -2.0);

    const auto result = contract(left, right);
    const auto reference = contract_reference(left, right);
    const std::array<std::string, 5> expected_legs{"a", "b", "c", "d", "e"};
    const std::array<usize, 5> expected_shape{2, 3, 4, 1, 2};

    REQUIRE(std::ranges::equal(result.leg_names(), std::span<const std::string>{expected_legs}));
    REQUIRE(std::ranges::equal(result.shape(), std::span<const usize>{expected_shape}));
    REQUIRE(close_per_element(result, reference));
}

TEST_CASE("contract handles tensors whose legs are all shared and returns a scalar", "[tensor]")
{
    auto left = Tensor({2, 3, 1, 2}, {"s03", "s01", "s02", "s00"});
    auto right = Tensor({2, 3, 2, 1}, {"s00", "s01", "s03", "s02"});
    fill_deterministic(left, -1.0);
    fill_deterministic(right, 6.0);

    const auto result = contract(left, right);
    const auto reference = contract_reference(left, right);

    REQUIRE(result.is_scalar());
    REQUIRE(result.leg_names().empty());
    REQUIRE(result.shape().empty());
    REQUIRE(close_per_element(result, reference));
}

TEST_CASE("contract handles scalar-tensor contraction by scaling the tensor", "[tensor]")
{
    auto scalar = Tensor::scalar(2.5);
    auto tensor = Tensor({2, 1, 3}, {"a", "b", "c"});
    fill_deterministic(tensor, 1.25);

    const auto result = contract(scalar, tensor);
    const auto reference = contract_reference(scalar, tensor);
    const std::array<std::string, 3> expected_legs{"a", "b", "c"};

    REQUIRE(std::ranges::equal(result.leg_names(), std::span<const std::string>{expected_legs}));
    REQUIRE(close_per_element(result, reference));
}

TEST_CASE("contract matches reference for mixed contractions up to rank 30", "[tensor]")
{
    for (auto rank = 0zu; rank <= 30zu; ++rank)
    {
        CAPTURE(rank);

        const auto generated = generate_rank_case(rank);
        const auto partition = partition_indices(generated.left, generated.right);

        auto shared_from_left = std::vector<std::string>{};
        auto shared_from_right = std::vector<std::string>{};
        shared_from_left.reserve(partition.left_shared.size());
        shared_from_right.reserve(partition.right_shared.size());
        for (const auto axis : partition.left_shared)
        {
            shared_from_left.push_back(generated.left.leg_name(axis));
        }
        for (const auto axis : partition.right_shared)
        {
            shared_from_right.push_back(generated.right.leg_name(axis));
        }

        REQUIRE(std::ranges::is_sorted(shared_from_left));
        REQUIRE(shared_from_left == shared_from_right);
        REQUIRE(std::ranges::equal(
            contraction_output_legs(generated.left, generated.right),
            std::span<const std::string>{generated.expected_legs}
        ));
        REQUIRE(std::ranges::equal(
            contraction_output_shape(generated.left, generated.right),
            std::span<const usize>{generated.expected_shape}
        ));

        const auto result = contract(generated.left, generated.right);
        const auto reference = contract_reference(generated.left, generated.right);

        REQUIRE(std::ranges::equal(result.leg_names(), std::span<const std::string>{generated.expected_legs}));
        REQUIRE(std::ranges::equal(result.shape(), std::span<const usize>{generated.expected_shape}));
        REQUIRE(close_per_element(result, reference));
    }
}

TEST_CASE("contract matches reference for high-rank edge cases with rank 30", "[tensor]")
{
    auto left_names = std::vector<std::string>{};
    auto right_names = std::vector<std::string>{};
    left_names.reserve(30zu);
    right_names.reserve(30zu);
    for (auto i = 0zu; i < 30zu; ++i)
    {
        left_names.push_back(make_name("no_shared_left", i));
        right_names.push_back(make_name("no_shared_right", i));
    }

    auto left_shape = std::vector<usize>(30zu, 1zu);
    auto right_shape = std::vector<usize>(30zu, 1zu);
    for (auto i = 0zu; i < 6zu; ++i)
    {
        left_shape[i] = 2zu;
        right_shape[i] = i % 2zu == 0zu ? 2zu : 1zu;
    }

    auto no_shared_left = Tensor{left_shape, std::span<const std::string>{left_names}};
    auto no_shared_right = Tensor{right_shape, std::span<const std::string>{right_names}};
    fill_deterministic(no_shared_left, -8.0);
    fill_deterministic(no_shared_right, 5.0);

    const auto no_shared_reference = contract_reference(no_shared_left, no_shared_right);
    const auto no_shared_result = contract(no_shared_left, no_shared_right);
    REQUIRE(close_per_element(no_shared_result, no_shared_reference));

    auto shared_names = std::vector<std::string>{};
    shared_names.reserve(30zu);
    for (auto i = 0zu; i < 30zu; ++i)
    {
        shared_names.push_back(make_name("all_shared", i));
    }
    const auto shared_names_reversed =
        std::vector<std::string>{shared_names.rbegin(), shared_names.rend()};

    auto all_shared_left_shape = std::vector<usize>(30zu, 1zu);
    auto all_shared_right_shape = std::vector<usize>(30zu, 1zu);
    for (auto i = 0zu; i < 6zu; ++i)
    {
        all_shared_left_shape[i] = 2zu;
        all_shared_right_shape[29zu - i] = 2zu;
    }

    auto left = Tensor{all_shared_left_shape, std::span<const std::string>{shared_names_reversed}};
    auto right = Tensor{all_shared_right_shape, std::span<const std::string>{shared_names}};
    fill_deterministic(left, 7.0);
    fill_deterministic(right, -4.0);

    const auto all_shared_result = contract(left, right);
    const auto all_shared_reference = contract_reference(left, right);
    REQUIRE(all_shared_result.is_scalar());
    REQUIRE(close_per_element(all_shared_result, all_shared_reference));
}

}  // namespace ds_tn
