#include <Accelerate/Accelerate.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <print>

auto main() -> int {
    constexpr auto lhs = std::array<double, 6>{
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    };
    constexpr auto rhs = std::array<double, 6>{
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    };
    auto result = std::array<double, 4>{};
    constexpr auto expected = std::array<double, 4>{
        58.0,
        64.0,
        139.0,
        154.0,
    };
    constexpr auto tolerance = 1.0e-12;

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        2,
        2,
        3,
        1.0,
        lhs.data(),
        3,
        rhs.data(),
        2,
        0.0,
        result.data(),
        2);

    auto passed = true;
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (std::abs(result[index] - expected[index]) > tolerance) {
            passed = false;
        }
    }

    std::println("Accelerate BLAS dgemm test: {}", passed ? "passed" : "failed");
    std::println("[[{:.1f}, {:.1f}], [{:.1f}, {:.1f}]]", result[0], result[1], result[2], result[3]);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
