cmake_minimum_required(VERSION 3.24)
project(c4tts LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# c4tts: a standalone, Apple-Silicon-optimized C++ inference engine for
# Confucius4-TTS. No Python runtime dependency. MLX is the primary compute
# backend (Metal under the hood); custom Metal kernels are added later for
# hotspots. See docs/PLAN.md for the full roadmap.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT APPLE OR NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"))
  message(WARNING "c4tts targets Apple Silicon (arm64 macOS). Building elsewhere is unsupported.")
endif()

# ---------------------------------------------------------------------------
# Locate MLX. The C++ library, headers and CMake package config all ship
# inside the `mlx` pip wheel, so we ask Python where it lives and point
# find_package() at the bundled share/cmake/MLX directory. Override with
# -DMLX_ROOT=/path/to/mlx if you build MLX from source.
# ---------------------------------------------------------------------------
if(NOT MLX_ROOT)
  find_program(PYTHON_EXECUTABLE NAMES python3 python)
  if(PYTHON_EXECUTABLE)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import mlx.core as mx, os; print(os.path.dirname(mx.__file__))"
      OUTPUT_VARIABLE MLX_PKG_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  if(MLX_PKG_DIR AND EXISTS "${MLX_PKG_DIR}/share/cmake/MLX")
    set(MLX_ROOT "${MLX_PKG_DIR}")
    message(STATUS "c4tts: found MLX in pip package at ${MLX_ROOT}")
  endif()
endif()

if(MLX_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${MLX_ROOT}/share/cmake/MLX")
endif()

find_package(MLX CONFIG REQUIRED)
message(STATUS "c4tts: MLX_INCLUDE_DIRS=${MLX_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# Core library (grows over the phases in docs/PLAN.md).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE C4TTS_CORE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(c4tts_core STATIC ${C4TTS_CORE_SRC})
target_include_directories(c4tts_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(c4tts_core PUBLIC mlx)

# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
add_executable(c4tts_cli apps/c4tts_cli.cpp)
target_link_libraries(c4tts_cli PRIVATE c4tts_core)

# ---------------------------------------------------------------------------
# Tests (opt-in via -DC4TTS_BUILD_TESTS=ON).
# ---------------------------------------------------------------------------
option(C4TTS_BUILD_TESTS "Build c4tts unit tests" OFF)
if(C4TTS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
