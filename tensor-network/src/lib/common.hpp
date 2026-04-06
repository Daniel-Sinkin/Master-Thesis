// lib/common.hpp
#pragma once

#include <cstddef>
#include <cstdint>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace ds_tn {

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

#if defined(__cpp_lib_stdfloat) && __cpp_lib_stdfloat >= 202207L
using f32 = std::float32_t;
using f64 = std::float64_t;
#else
using f32 = float;
using f64 = double;
#endif
static_assert(sizeof(f32) == 4);
static_assert(sizeof(f64) == 8);

namespace literals {

[[nodiscard]] constexpr auto operator""_b(unsigned long long value) noexcept -> usize {
    return static_cast<usize>(value);
}

[[nodiscard]] constexpr auto operator""_kib(unsigned long long value) noexcept -> usize {
    return static_cast<usize>(value * 1024ULL);
}

[[nodiscard]] constexpr auto operator""_mib(unsigned long long value) noexcept -> usize {
    return static_cast<usize>(value * 1024ULL * 1024ULL);
}

} // namespace literals

} // namespace ds_tn
