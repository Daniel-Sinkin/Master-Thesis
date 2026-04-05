cmake_minimum_required(VERSION 3.28)
project(tensor-network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(CREATE_LINK
  "${CMAKE_BINARY_DIR}/compile_commands.json"
  "${CMAKE_SOURCE_DIR}/compile_commands.json"
  SYMBOLIC
  COPY_ON_ERROR
)

add_library(tn_pch OBJECT src/lib/pch.cpp)
target_include_directories(tn_pch PUBLIC ${CMAKE_SOURCE_DIR}/src/lib)
target_precompile_headers(tn_pch PUBLIC src/lib/pch.hpp)

add_library(tn_lib INTERFACE)
target_include_directories(tn_lib INTERFACE ${CMAKE_SOURCE_DIR}/src/lib)

add_executable(tn_app src/app/main.cpp)
set_target_properties(tn_app PROPERTIES OUTPUT_NAME tensor-network)
target_link_libraries(tn_app PRIVATE tn_lib)
target_precompile_headers(tn_app REUSE_FROM tn_pch)

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
set_strict_warnings(tn_app)
