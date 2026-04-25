// lib/common.hpp
#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#if __has_include(<stdfloat>)
#    include <stdfloat>
#endif

#include <ranges>

namespace ds_tn
{

using usize = std::size_t;
using isize = std::ptrdiff_t;
using uptr = std::uintptr_t;
using iptr = std::intptr_t;

using i64 = std::int64_t;
using i32 = std::int32_t;
using i16 = std::int16_t;
using i8 = std::int8_t;

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u8 = std::uint8_t;

#if defined(__cpp_lib_stdfloat) and __cpp_lib_stdfloat >= 202207L
using f32 = std::float32_t;
using f64 = std::float64_t;
#else
using f32 = float;
using f64 = double;
#endif
static_assert(sizeof(f32) == 4);
static_assert(sizeof(f64) == 8);

struct LogSettings
{
    std::string_view name{};
};

[[nodiscard]] inline auto format_bytes(usize bytes, usize digits = 2) -> std::string
{
    constexpr auto units = std::array<std::string_view, 5>{"B", "KiB", "MiB", "GiB", "TiB"};

    auto unit_index = 0zu;
    auto value = static_cast<long double>(bytes);
    while (value >= 1024.0L and unit_index + 1 < units.size())
    {
        value /= 1024.0L;
        ++unit_index;
    }

    if (unit_index == 0)
    {
        return std::to_string(bytes) + " B";
    }

    auto buffer = std::ostringstream{};
    buffer << std::fixed << std::setprecision(static_cast<int>(digits)) << value << ' '
           << units[unit_index];
    return buffer.str();
}

template <std::integral Integer>
[[nodiscard]] constexpr auto iota_n(Integer end)
{
    return std::views::iota(Integer{0}, end);
}

template <std::integral Integer>
[[nodiscard]] constexpr auto iota_n(Integer begin, Integer end)
{
    return std::views::iota(begin, end);
}

template <typename T, usize Extent>
[[nodiscard]] constexpr auto iota_n(std::span<T, Extent> values)
{
    return iota_n(values.size());
}

template <typename T, usize LeftExtent, usize RightExtent>
[[nodiscard]] constexpr auto
inner_product(std::span<const T, LeftExtent> lhs, std::span<const T, RightExtent> rhs) -> T
{
    {  // Expects
        if (lhs.size() != rhs.size())
        {
            throw std::invalid_argument("inner_product requires spans of the same size.");
        }
    }

    return std::transform_reduce(
        lhs.begin(), lhs.end(), rhs.begin(), T{0}, std::plus<T>{}, std::multiplies<T>{}
    );
}

namespace literals
{

[[nodiscard]] constexpr auto operator""_b(unsigned long long value) noexcept -> usize
{
    return static_cast<usize>(value);
}

[[nodiscard]] constexpr auto operator""_kib(unsigned long long value) noexcept -> usize
{
    return static_cast<usize>(value * 1024ULL);
}

[[nodiscard]] constexpr auto operator""_mib(unsigned long long value) noexcept -> usize
{
    return static_cast<usize>(value * 1024ULL * 1024ULL);
}

[[nodiscard]] constexpr auto operator""_gib(unsigned long long value) noexcept -> usize
{
    return static_cast<usize>(value * 1024ULL * 1024ULL * 1024ULL);
}

[[nodiscard]] constexpr auto operator""_tib(unsigned long long value) noexcept -> usize
{
    return static_cast<usize>(value * 1024ULL * 1024ULL * 1024ULL * 1024ULL);
}

}  // namespace literals

}  // namespace ds_tn
