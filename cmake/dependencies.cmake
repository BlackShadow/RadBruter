# One dependency setup for both supported platforms. Everything stays in .deps.
set(DEPS "${CMAKE_SOURCE_DIR}/.deps")
file(MAKE_DIRECTORY "${DEPS}")
file(REAL_PATH "${DEPS}" dependency_root)
file(LOCK "${DEPS}/setup.lock" GUARD FILE TIMEOUT 180)

function(rb_dependency name url checksum marker)
    cmake_parse_arguments(PACKAGE "" "REVISION" "KEEP" ${ARGN})
    set(destination "${DEPS}/${name}")
    if(EXISTS "${destination}/${marker}")
        if(PACKAGE_REVISION)
            file(READ "${destination}/.revision" revision)
            string(STRIP "${revision}" revision)
            if(NOT revision STREQUAL PACKAGE_REVISION)
                message(FATAL_ERROR "Unexpected dependency revision in ${destination}")
            endif()
        endif()
        return()
    endif()
    if(EXISTS "${destination}")
        message(FATAL_ERROR "Incomplete dependency: remove ${destination} and configure again")
    endif()
    string(REPLACE "/" "-" stage_name "${name}")
    set(stage "${dependency_root}/.unpack-${stage_name}")
    cmake_path(IS_PREFIX dependency_root "${stage}" NORMALIZE safe_stage)
    cmake_path(IS_PREFIX dependency_root "${destination}" NORMALIZE safe_destination)
    if(NOT safe_stage OR NOT safe_destination OR stage STREQUAL dependency_root)
        message(FATAL_ERROR "Dependency paths must stay inside .deps")
    endif()
    file(REMOVE_RECURSE "${stage}")
    file(MAKE_DIRECTORY "${stage}/source" "${stage}/package")
    message(STATUS "Downloading ${name}")
    file(DOWNLOAD "${url}" "${stage}/archive" EXPECTED_HASH "SHA256=${checksum}" TLS_VERIFY ON)
    file(ARCHIVE_EXTRACT INPUT "${stage}/archive" DESTINATION "${stage}/source")
    file(GLOB roots "${stage}/source/*")
    list(LENGTH roots root_count)
    if(NOT root_count EQUAL 1 OR NOT IS_DIRECTORY "${roots}")
        message(FATAL_ERROR "Unexpected archive layout for ${name}")
    endif()
    foreach(pattern IN LISTS PACKAGE_KEEP)
        file(GLOB entries RELATIVE "${roots}" "${roots}/${pattern}")
        foreach(entry IN LISTS entries)
            get_filename_component(parent "${entry}" DIRECTORY)
            file(MAKE_DIRECTORY "${stage}/package/${parent}")
            file(COPY "${roots}/${entry}" DESTINATION "${stage}/package/${parent}")
        endforeach()
    endforeach()
    if(NOT EXISTS "${stage}/package/${marker}")
        message(FATAL_ERROR "Missing ${marker} in ${name}")
    endif()
    if(PACKAGE_REVISION)
        file(WRITE "${stage}/package/.revision" "${PACKAGE_REVISION}\n")
    endif()
    get_filename_component(parent "${destination}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    file(RENAME "${stage}/package" "${destination}")
    file(REMOVE_RECURSE "${stage}")
endfunction()

rb_dependency(eigen
    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    Eigen/Core KEEP Eigen COPYING.*)
rb_dependency(lbfgspp
    https://github.com/yixuan/LBFGSpp/archive/refs/tags/v0.3.0.tar.gz
    490720b9d5acce6459cb0336ca3ae0ffc48677225f0ebfb35c9bef6baefdfc6a
    include/LBFGSB.h KEEP include LICENSE.md AUTHORS.md)

function(rb_compiler family repository revision checksum prefix rad)
    set(keep LICENSE.md "Terms of Use.txt")
    foreach(directory common template "${rad}")
        list(APPEND keep "${prefix}${directory}/*.cpp" "${prefix}${directory}/*.h")
    endforeach()
    rb_dependency("compiler-sources/${family}"
        "https://codeload.github.com/${repository}/tar.gz/${revision}" "${checksum}"
        "${prefix}${rad}/qrad.cpp" REVISION "${revision}" KEEP ${keep})
endfunction()
rb_compiler(zhlt kriswema/zhlt da4f9d76cf425b06864c29b32dab73892e36a6c6
    2ad186843264230bcd460be8ed34ad6fd7f1d8974de95696af8e7e4ff81d981e "" hlrad)
rb_compiler(vhlt FreeSlave/vhlt 13b83f91d093ef146a9a78834578994d85f99d96
    6626f23f379bd73bcb1e7b7645a0b17647bd163e59cf0fc1a8920b54c9cf8ad3 "" hlrad)
rb_compiler(sdhlt seedee/SDHLT df45198b3c03a5a09e9d1aead9c9457e51753e39
    5fca807f6da7db4dfb06744676ebdf18ffaadff820dd90d6744b8298a0cba11f "src/sdhlt/" sdHLRAD)

# Headers are used at build time; the libraries are optional at run time.
set(cuda_redist https://developer.download.nvidia.com/compute/cuda/redist)
set(CUDA_DEPS "${DEPS}/cuda/${CMAKE_SYSTEM_NAME}")
if(WIN32)
    rb_dependency(cuda/Windows/runtime
        "${cuda_redist}/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip"
        179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8
        include/cuda.h KEEP include LICENSE)
    rb_dependency(cuda/Windows/nvrtc
        "${cuda_redist}/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.9.86-archive.zip"
        1aa0644fa53c8ca34cdc73db17bcc73530557bdd3f582c7bfdbd7916c8b48f65
        include/nvrtc.h KEEP include bin/nvrtc64_120_0.dll bin/nvrtc-builtins64_129.dll LICENSE)
    set(RB_NVRTC_FILES "${CUDA_DEPS}/nvrtc/bin/nvrtc64_120_0.dll" "${CUDA_DEPS}/nvrtc/bin/nvrtc-builtins64_129.dll")
else()
    rb_dependency(cuda/Linux/runtime
        "${cuda_redist}/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz"
        1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
        include/cuda.h KEEP include LICENSE)
    rb_dependency(cuda/Linux/nvrtc
        "${cuda_redist}/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz"
        82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af
        include/nvrtc.h KEEP include lib/libnvrtc.so* lib/libnvrtc-builtins.so* LICENSE)
    set(RB_NVRTC_FILES "${CUDA_DEPS}/nvrtc/lib/libnvrtc.so.12" "${CUDA_DEPS}/nvrtc/lib/libnvrtc-builtins.so.12.9")
endif()
