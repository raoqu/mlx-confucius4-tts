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
# MLX backend. Two ways to obtain it:
#
#  (default) C4TTS_MLX_FROM_SOURCE=ON: build MLX from source (FetchContent,
#  pinned to MLX_VERSION) as a STATIC lib with no Python bindings. This makes
#  c4tts fully Python-free at runtime AND build — libmlx is linked statically
#  and the only loose artifact is mlx.metallib, which we copy next to the
#  binary (MLX searches for it colocated with the executable). Building MLX
#  core does not use Python (the only find_package(Python) is in the python/
#  bindings subdir, which stays OFF).
#
#  C4TTS_MLX_FROM_SOURCE=OFF: consume a prebuilt MLX. Uses -DMLX_ROOT if given,
#  else asks the `mlx` pip wheel where it lives (fast for development, but the
#  resulting binary rpaths libmlx.dylib out of site-packages).
# ---------------------------------------------------------------------------
# NOTE: building MLX from source needs the Metal shader compiler (`xcrun metal`),
# which ships only with full Xcode — the Command Line Tools alone are not enough
# (even MLX_METAL_JIT still precompiles a base metallib). Default OFF so builds
# don't break on CLT-only machines; turn ON when full Xcode is installed for a
# fully Python-free, statically-linked binary.
option(C4TTS_MLX_FROM_SOURCE "Build MLX from source (Python-free; needs full Xcode)" OFF)
# Git tag of MLX to build (kept distinct from MLX's own MLX_VERSION, which it
# derives from its version.h — don't set that one).
set(C4TTS_MLX_TAG "v0.31.2" CACHE STRING "MLX git tag to build from source")

if(C4TTS_MLX_FROM_SOURCE)
  include(FetchContent)
  # MLX core build options: no python bindings, no tests/examples, static lib.
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  # JIT-compile Metal kernels at runtime (via the Metal framework) instead of
  # precompiling a metallib with `xcrun metal`, which needs full Xcode. This
  # lets the source build work with only the Command Line Tools; kernels are
  # compiled on first use (a one-time warmup). Set C4TTS_MLX_METAL_JIT=OFF if
  # you have full Xcode and prefer the precompiled metallib.
  option(C4TTS_MLX_METAL_JIT "JIT-compile MLX Metal kernels (no Xcode needed)" ON)
  set(MLX_METAL_JIT ${C4TTS_MLX_METAL_JIT} CACHE BOOL "" FORCE)
  # Put the compiled Metal kernels at a known path so we can colocate it.
  set(MLX_METAL_PATH "${CMAKE_BINARY_DIR}/mlx-kernels" CACHE PATH "" FORCE)
  # Fail early with guidance if the Metal shader compiler is missing.
  execute_process(COMMAND xcrun --find metal RESULT_VARIABLE _no_metal OUTPUT_QUIET ERROR_QUIET)
  if(_no_metal)
    message(FATAL_ERROR
      "C4TTS_MLX_FROM_SOURCE=ON requires the Metal shader compiler (`xcrun metal`), "
      "which needs full Xcode (not just the Command Line Tools).\n"
      "  Install Xcode + `sudo xcode-select -s /Applications/Xcode.app`, then rebuild.\n"
      "  Or set -DC4TTS_MLX_FROM_SOURCE=OFF to use a prebuilt MLX.")
  endif()
  message(STATUS "c4tts: building MLX ${C4TTS_MLX_TAG} from source (Python-free)")
  FetchContent_Declare(mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG ${C4TTS_MLX_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(mlx)
else()
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
endif()

# ---------------------------------------------------------------------------
# Core library (grows over the phases in docs/PLAN.md).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE C4TTS_CORE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(c4tts_core STATIC ${C4TTS_CORE_SRC})
target_include_directories(c4tts_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(c4tts_core PUBLIC mlx)

# ---------------------------------------------------------------------------
# SQLite — backs the web server's voice store (voices.sqlite). Ships with the
# macOS SDK; prefer the CMake-provided target, fall back to plain -lsqlite3.
# ---------------------------------------------------------------------------
find_package(SQLite3 QUIET)
if(TARGET SQLite3::SQLite3)
  target_link_libraries(c4tts_core PUBLIC SQLite3::SQLite3)
elseif(SQLite3_FOUND)
  target_link_libraries(c4tts_core PUBLIC SQLite::SQLite3)
else()
  target_link_libraries(c4tts_core PUBLIC sqlite3)
endif()

# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
add_executable(c4tts_cli apps/c4tts_cli.cpp)
target_link_libraries(c4tts_cli PRIVATE c4tts_core)

# Self-contained runtime: MLX searches for its Metal kernels colocated with the
# executable first, so copy mlx.metallib next to the binary. With a source
# (static) MLX build this is the only loose runtime artifact — no Python, no
# site-packages dylib.
if(C4TTS_MLX_FROM_SOURCE)
  add_custom_command(TARGET c4tts_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MLX_METAL_PATH}/mlx.metallib"
            "$<TARGET_FILE_DIR:c4tts_cli>/mlx.metallib"
    COMMENT "Colocating mlx.metallib next to c4tts_cli")
endif()

# Self-contained runtime WITHOUT building MLX from source (no Xcode needed):
# vendor the prebuilt libmlx.dylib + mlx.metallib next to the binary and make
# @executable_path the only rpath, so the running binary no longer depends on
# MLX living in the Python/conda site-packages. The dylib/metallib are binary
# artifacts (Apple-compiled GPU kernels), not a Python dependency.
option(C4TTS_SELFCONTAINED "Vendor prebuilt MLX dylib+metallib next to the binary" OFF)
if(C4TTS_SELFCONTAINED AND NOT C4TTS_MLX_FROM_SOURCE)
  set_target_properties(c4tts_cli PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "@executable_path")
  # Copy every MLX runtime artifact (libmlx.dylib, its libjaccl.dylib dep, and
  # mlx.metallib); the @rpath deps resolve to these via @executable_path.
  file(GLOB _mlx_dylibs "${MLX_ROOT}/lib/*.dylib")
  set(_vendor_cmds "")
  foreach(_d ${_mlx_dylibs})
    get_filename_component(_dn "${_d}" NAME)
    list(APPEND _vendor_cmds COMMAND ${CMAKE_COMMAND} -E copy_if_different
         "${_d}" "$<TARGET_FILE_DIR:c4tts_cli>/${_dn}")
  endforeach()
  add_custom_command(TARGET c4tts_cli POST_BUILD ${_vendor_cmds}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MLX_ROOT}/lib/mlx.metallib" "$<TARGET_FILE_DIR:c4tts_cli>/mlx.metallib"
    COMMENT "Vendoring MLX dylibs + mlx.metallib next to c4tts_cli (self-contained)")
endif()

# ---------------------------------------------------------------------------
# Tests (opt-in via -DC4TTS_BUILD_TESTS=ON).
# ---------------------------------------------------------------------------
option(C4TTS_BUILD_TESTS "Build c4tts unit tests" OFF)
if(C4TTS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
