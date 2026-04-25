#include "common.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <vector>

namespace ds_tn
{
namespace
{

using namespace literals;

template <typename Range>
[[nodiscard]] auto collect(Range&& range)
{
    auto values = std::vector<usize>{};
    for (const auto value : range)
    {
        values.push_back(value);
    }
    return values;
}

}  // namespace

TEST_CASE("iota_n helper covers its supported forms", "[common]")
{
    REQUIRE(collect(iota_n(4zu)) == std::vector<usize>{0, 1, 2, 3});
    REQUIRE(collect(iota_n(2zu, 5zu)) == std::vector<usize>{2, 3, 4});

    const auto values = std::array<usize, 3>{9, 8, 7};
    REQUIRE(collect(iota_n(std::span{values})) == std::vector<usize>{0, 1, 2});
}

TEST_CASE("inner_product helper computes the explicit transform reduce form", "[common]")
{
    const auto lhs = std::array<usize, 3>{1, 2, 3};
    const auto rhs = std::array<usize, 3>{4, 5, 6};

    REQUIRE(inner_product(std::span{lhs}, std::span{rhs}) == 32zu);

    const auto shorter = std::array<usize, 2>{1, 2};
    REQUIRE_THROWS_AS(inner_product(std::span{lhs}, std::span{shorter}), std::invalid_argument);
}

TEST_CASE("format_bytes selects binary units and honours precision", "[common]")
{
    REQUIRE(format_bytes(999) == "999 B");
    REQUIRE(format_bytes(1024) == "1.00 KiB");
    REQUIRE(format_bytes(1536) == "1.50 KiB");
    REQUIRE(format_bytes(1536, 1) == "1.5 KiB");
    REQUIRE(format_bytes(1_tib) == "1.00 TiB");
}

}  // namespace ds_tn
