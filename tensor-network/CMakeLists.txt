cmake_minimum_required(VERSION 3.28)
project(tensor-network LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BLA_VENDOR Apple)
find_package(BLAS REQUIRED)

file(CREATE_LINK
  "${CMAKE_BINARY_DIR}/compile_commands.json"
  "${CMAKE_SOURCE_DIR}/compile_commands.json"
  SYMBOLIC
  COPY_ON_ERROR
)

add_library(tn_pch OBJECT src/lib/pch.cpp)
target_include_directories(tn_pch PUBLIC ${CMAKE_SOURCE_DIR}/src/lib)
target_precompile_headers(tn_pch PUBLIC src/lib/pch.hpp)

add_library(tn_lib STATIC
  src/lib/models/transverse_ising.cpp
  src/lib/permutation/permutation.cpp
  src/lib/ndarray/ndarray.cpp
  src/lib/ndarray/blas.cpp
  src/lib/ndarray/compare.cpp
  src/lib/ndarray/generator.cpp
  src/lib/ndarray/lanczos.cpp
  src/lib/ndarray/lapack.cpp
  src/lib/ndarray/stats.cpp
  src/lib/tensor/contraction.cpp
  src/lib/tensor/compare.cpp
  src/lib/tensor/mps.cpp
  src/lib/tensor/tensor.cpp
)
target_include_directories(tn_lib PUBLIC ${CMAKE_SOURCE_DIR}/src/lib)
target_compile_definitions(tn_lib PRIVATE ACCELERATE_NEW_LAPACK)
target_precompile_headers(tn_lib REUSE_FROM tn_pch)
if (TARGET BLAS::BLAS)
  target_link_libraries(tn_lib PUBLIC BLAS::BLAS)
else()
  target_link_libraries(tn_lib PUBLIC ${BLAS_LIBRARIES})
endif()

add_executable(tn_app src/app/main.cpp)
set_target_properties(tn_app PROPERTIES OUTPUT_NAME tensor-network)
target_link_libraries(tn_app PRIVATE tn_lib)
target_precompile_headers(tn_app REUSE_FROM tn_pch)

option(TN_BUILD_TESTS "Build the tensor-network test suite" ON)

function(set_strict_warnings target_name)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target_name} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wshadow
      -Wconversion
      -Wfloat-conversion
      -Wsign-conversion
      -Wcast-qual
      -Wcast-align
      -Wformat=2
      -Wundef
      -Wnull-dereference
      -Wnon-virtual-dtor
      -Woverloaded-virtual
      -Wold-style-cast
      -Wzero-as-null-pointer-constant
      -Wdouble-promotion
      -Wimplicit-fallthrough
      -Wreturn-type
      -Wshorten-64-to-32
      -Wimplicit-int-conversion
      -Wmissing-declarations
      -Wnewline-eof
    )
  else()
    target_compile_options(${target_name} PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wshadow
      -Wconversion
      -Wfloat-conversion
      -Wsign-conversion
      -Wcast-qual
      -Wcast-align
      -Wformat=2
      -Wundef
      -Wnull-dereference
      -Wnon-virtual-dtor
      -Woverloaded-virtual
      -Wold-style-cast
      -Wzero-as-null-pointer-constant
      -Wdouble-promotion
      -Wimplicit-fallthrough
      -Wreturn-type
    )
  endif()
endfunction()

set_strict_warnings(tn_pch)
set_strict_warnings(tn_lib)
set_strict_warnings(tn_app)

if (TN_BUILD_TESTS)
  include(FetchContent)

  find_package(Catch2 3 CONFIG QUIET)
  if (NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.11.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
  endif()

  add_executable(tn_tests
    tests/common_tests.cpp
    tests/model_tests.cpp
    tests/permutation_tests.cpp
    tests/ndarray_tests.cpp
    tests/ndarray_random_stats_compare_tests.cpp
    tests/ndarray_lanczos_tests.cpp
    tests/ndarray_lapack_tests.cpp
    tests/ndarray_blas_tests.cpp
    tests/tensor_contraction_tests.cpp
    tests/mps_tests.cpp
    tests/tensor_tests.cpp
  )
  set_target_properties(tn_tests PROPERTIES OUTPUT_NAME tensor-network-tests)
  target_link_libraries(tn_tests PRIVATE tn_lib Catch2::Catch2WithMain)
  target_precompile_headers(tn_tests REUSE_FROM tn_pch)
  set_strict_warnings(tn_tests)

  add_test(NAME tn_tests COMMAND tn_tests)

  add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS tn_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
  )
endif()
