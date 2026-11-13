cmake_minimum_required(VERSION 3.18)
project(constrained_potential_game LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# PyPI CasADi Linux wheels use the old libstdc++ string ABI.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
set(PYBIND11_FINDPYTHON ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../extern/pybind11"
                 "${CMAKE_CURRENT_BINARY_DIR}/pybind11" EXCLUDE_FROM_ALL)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import casadi; print(casadi.__path__[0])"
    OUTPUT_VARIABLE CASADI_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CASADI_IMPORT_RESULT)
if(NOT CASADI_IMPORT_RESULT EQUAL 0)
    message(FATAL_ERROR "Install CasADi in the selected Python environment first.")
endif()
find_library(CASADI_LIB NAMES casadi HINTS "${CASADI_ROOT}" NO_DEFAULT_PATH REQUIRED)

find_package(spdlog 1.12 CONFIG QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0)
    FetchContent_MakeAvailable(spdlog)
endif()
find_library(UMFPACK_LIBRARY NAMES umfpack REQUIRED)
find_library(AMD_LIBRARY NAMES amd REQUIRED)
find_library(SUPERLU_LIBRARY NAMES superlu REQUIRED)
find_path(SUPERLU_INCLUDE_DIR slu_Cnames.h PATH_SUFFIXES superlu REQUIRED)
find_path(SUITESPARSE_INCLUDE_DIR umfpack.h PATH_SUFFIXES suitesparse REQUIRED)

# Build only current solver-specific sources. All three games and both solvers
# are discovered, including any explicitly generated horizon/agent counts.
set(CASADI_SOURCES)
foreach(GAME car_merge_kinematic_bicycle intersection car_racing)
    file(GLOB GAME_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/../casadi_codegen/${GAME}_casadi_ipm_vne_N*_T*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../casadi_codegen/${GAME}_casadi_rd3g_N*_T*.cpp")
    list(APPEND CASADI_SOURCES ${GAME_SOURCES})
endforeach()
if(NOT CASADI_SOURCES)
    message(FATAL_ERROR "No CasADi sources. Run: uv run python -m scripts.generate_casadi_code")
endif()
foreach(SOURCE_FILE IN LISTS CASADI_SOURCES)
    get_filename_component(LIB_NAME "${SOURCE_FILE}" NAME_WE)
    add_library(${LIB_NAME} SHARED "${SOURCE_FILE}")
endforeach()

foreach(SOLVER interior_point_game rd3g_casadi)
    pybind11_add_module(${SOLVER} "solvers/${SOLVER}.cpp")
    target_include_directories(${SOLVER} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/solvers"
        "${CASADI_ROOT}/include" "${SUPERLU_INCLUDE_DIR}" "${SUITESPARSE_INCLUDE_DIR}")
    target_link_libraries(${SOLVER} PRIVATE Eigen3::Eigen spdlog::spdlog_header_only
        "${CASADI_LIB}" "${UMFPACK_LIBRARY}" "${AMD_LIBRARY}" "${SUPERLU_LIBRARY}")
    # Retain the native solver's numerical assertions in Release builds.
    target_compile_options(${SOLVER} PRIVATE -UNDEBUG)
endforeach()
