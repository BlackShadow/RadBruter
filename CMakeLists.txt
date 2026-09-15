cmake_minimum_required(VERSION 3.24)
project(RadBruter VERSION 0.2.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "RadBruter requires a 64-bit toolchain")
endif()
if(WIN32)
    if(NOT MSVC OR MSVC_VERSION LESS 1930 OR NOT CMAKE_C_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
        message(FATAL_ERROR "Use the Visual Studio 2022 or later x64 C++ toolchain")
    endif()
    add_compile_definitions(WIN32 NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00
        _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64"
       OR NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR "Supported platforms: x64 Windows/MSVC and x86-64 Linux/GCC")
endif()
include(cmake/dependencies.cmake)
find_package(Threads REQUIRED)
add_library(lm_platform STATIC src/platform.cpp)
target_include_directories(lm_platform PUBLIC src)
target_link_libraries(lm_platform PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(lm_platform PUBLIC bcrypt psapi)
else()
    find_package(OpenSSL REQUIRED)
    target_link_libraries(lm_platform PUBLIC OpenSSL::Crypto m)
endif()

set(RB_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${RB_OUTPUT_DIR}")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/symbols/$<CONFIG>")
set(QRAD "${CMAKE_SOURCE_DIR}/src/qrad")
add_library(qrad OBJECT
    ${QRAD}/engine/qrad.c ${QRAD}/engine/lightmap.c ${QRAD}/engine/trace.c ${QRAD}/engine/vismat.c
    ${QRAD}/common/bspfile.c ${QRAD}/common/cmdlib.c ${QRAD}/common/mathlib.c
    ${QRAD}/common/polylib.c ${QRAD}/common/scriplib.c ${QRAD}/common/threads.c)
target_include_directories(qrad PRIVATE src ${QRAD}/common ${QRAD}/engine)
if(MSVC)
    target_compile_options(qrad PRIVATE "$<$<NOT:$<CONFIG:Debug>>:/O2>" /fp:strict /W3 /FI${QRAD}/compat.h)
    target_compile_definitions(qrad PRIVATE LM_SERIAL_QRAD)
else()
    target_compile_options(qrad PRIVATE -O2 -fcommon -mfpmath=387 -fexcess-precision=standard
        -include ${QRAD}/compat.h -Wno-implicit-function-declaration -Wno-incompatible-pointer-types)
endif()
add_executable(radbruter src/main.cpp src/runtime.cpp $<TARGET_OBJECTS:qrad>)
target_include_directories(radbruter PRIVATE src ${QRAD}/common ${QRAD}/engine "${CMAKE_BINARY_DIR}/generated")
target_include_directories(radbruter SYSTEM PRIVATE "${DEPS}/eigen" "${DEPS}/lbfgspp/include"
    "${CUDA_DEPS}/runtime/include" "${CUDA_DEPS}/nvrtc/include")
target_compile_definitions(radbruter PRIVATE EIGEN_DONT_PARALLELIZE)
if(MSVC)
    target_compile_options(radbruter PRIVATE "$<$<NOT:$<CONFIG:Debug>>:/O2>" /fp:strict /W3 /bigobj)
else()
    target_compile_options(radbruter PRIVATE -O3 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter)
    set_target_properties(radbruter PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    target_link_options(radbruter PRIVATE LINKER:--disable-new-dtags)
endif()
target_link_libraries(radbruter PRIVATE lm_platform)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT radbruter)
include(src/compilers.cmake)

# GPU support is always included and loaded only when usable at runtime.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/kernels.cu src/gpu_types.h)
file(READ "${CMAKE_SOURCE_DIR}/src/kernels.cu" LM_CUDA_SOURCE)
file(READ "${CMAKE_SOURCE_DIR}/src/gpu_types.h" LM_GPU_TYPES)
string(PREPEND LM_CUDA_SOURCE "${LM_GPU_TYPES}\n")
if(MSVC)
    string(PREPEND LM_CUDA_SOURCE "#define LM_WINDOWS_ARITHMETIC 1\n")
endif()
configure_file(src/kernels.h.in generated/kernels.h @ONLY)
add_custom_command(TARGET radbruter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RB_NVRTC_FILES} $<TARGET_FILE_DIR:radbruter>
    COMMAND_EXPAND_LISTS VERBATIM)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME native_infrastructure COMMAND radbruter --self-test --backend cpu)
    foreach(family zhlt vhlt sdhlt)
        add_test(NAME ${family}_recovery COMMAND radbruter --self-test --backend cpu --compiler ${family})
    endforeach()
    add_test(NAME automatic_backend COMMAND radbruter --self-test)
    add_test(NAME no_visible_gpu COMMAND ${CMAKE_COMMAND}
        "-DAPP=$<TARGET_FILE:radbruter>" -DHIDE_GPU=ON
        -P "${CMAKE_SOURCE_DIR}/cmake/test_fallback.cmake")
    add_test(NAME cpu_fallback COMMAND ${CMAKE_COMMAND}
        "-DAPP=$<TARGET_FILE:radbruter>" "-DBUILD_ROOT=${CMAKE_SOURCE_DIR}/build" "-DSTAGE=${CMAKE_SOURCE_DIR}/build/test-cmake-$<CONFIG>"
        -P "${CMAKE_SOURCE_DIR}/cmake/test_fallback.cmake")
    set_tests_properties(native_infrastructure zhlt_recovery vhlt_recovery sdhlt_recovery
        automatic_backend no_visible_gpu cpu_fallback PROPERTIES TIMEOUT 300
        WORKING_DIRECTORY "${RB_OUTPUT_DIR}")
endif()
