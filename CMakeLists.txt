cmake_minimum_required(VERSION 3.20)
project(RealtimeObjectTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────
# Build options
# ─────────────────────────────────────────────
option(USE_CUDA       "Enable CUDA Execution Provider for ONNX Runtime" ON)
option(USE_TENSORRT   "Enable TensorRT Execution Provider for ONNX Runtime" OFF)

# ─────────────────────────────────────────────
# External dependency paths (user-configurable)
# ─────────────────────────────────────────────
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime SDK")
set(VIGEMCLIENT_ROOT "" CACHE PATH "Root directory of ViGEmClient SDK")
set(OPENCV_DIR       "" CACHE PATH "OpenCV CMake config directory")

# Detect compiler and set global encoding flags
if(MINGW)
    message(STATUS "Compiler: MinGW (GCC/G++)")
elseif(MSVC)
    message(STATUS "Compiler: MSVC")
    # Force UTF-8 source and execution character sets globally.
    # Without this, Japanese Windows (code page 932) misinterprets UTF-8
    # source files, causing cascading C2530/C2143/C4430 compile errors.
    # Use CMAKE_*_FLAGS to ensure these flags appear in every compilation
    # command, including FetchContent-built dependencies like OpenCV.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /wd4819")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /utf-8 /wd4819")
    add_compile_options(/utf-8 /wd4819)
endif()

# ─────────────────────────────────────────────
# FetchContent
# ─────────────────────────────────────────────
include(FetchContent)

# ─────────────────────────────────────────────
# ViGEmClient
# ─────────────────────────────────────────────
if(NOT VIGEMCLIENT_ROOT OR NOT EXISTS "${VIGEMCLIENT_ROOT}")
    message(STATUS "ViGEmClient not specified – fetching from GitHub …")
    FetchContent_Declare(
        vigemclient
        GIT_REPOSITORY https://github.com/ViGEm/ViGEmClient.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(vigemclient)
    set(VIGEMCLIENT_INCLUDE_DIR "${vigemclient_SOURCE_DIR}/include")
    set(VIGEMCLIENT_LIB ViGEmClient)
else()
    set(VIGEMCLIENT_INCLUDE_DIR "${VIGEMCLIENT_ROOT}/include")
    find_library(VIGEMCLIENT_LIB
        NAMES ViGEmClient
        PATHS "${VIGEMCLIENT_ROOT}/lib" "${VIGEMCLIENT_ROOT}/lib/x64"
    )
endif()

# ─────────────────────────────────────────────
# ONNX Runtime
# ─────────────────────────────────────────────
set(ONNXRUNTIME_VERSION "1.22.0" CACHE STRING "ONNX Runtime version to download")

if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}")
    message(STATUS "ONNX Runtime: using local SDK at ${ONNXRUNTIME_ROOT}")
else()
    # Auto-download pre-built ONNX Runtime GPU SDK
    message(STATUS "ONNX Runtime not specified – downloading v${ONNXRUNTIME_VERSION} GPU SDK …")
    if(WIN32)
        set(_ORT_ARCHIVE "onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
    else()
        set(_ORT_ARCHIVE "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
    endif()
    set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ORT_ARCHIVE}")

    FetchContent_Declare(
        onnxruntime_sdk
        URL      "${_ORT_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_sdk)
    set(ONNXRUNTIME_ROOT "${onnxruntime_sdk_SOURCE_DIR}")
    message(STATUS "ONNX Runtime SDK downloaded to: ${ONNXRUNTIME_ROOT}")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

# Create an imported target for ONNX Runtime that works with both MSVC and MinGW.
# The pre-built SDK ships onnxruntime.dll + onnxruntime.lib (MSVC import lib).
# MinGW can link against a .dll directly via IMPORTED_IMPLIB on the .dll file,
# or we generate a MinGW-compatible import library (.dll.a) with dlltool.
add_library(onnxruntime_imported SHARED IMPORTED)

if(WIN32)
    # Locate the DLL and the MSVC import lib
    file(GLOB _ORT_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll")
    file(GLOB _ORT_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.lib")

    if(_ORT_DLL)
        list(GET _ORT_DLL 0 _ORT_DLL_PATH)
        set_target_properties(onnxruntime_imported PROPERTIES
            IMPORTED_LOCATION "${_ORT_DLL_PATH}"
        )
    endif()

    if(MINGW)
        # MinGW: generate a .dll.a import library from the DLL using gendef+dlltool
        if(_ORT_DLL)
            set(_ORT_DEF "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.def")
            set(_ORT_DLLA "${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.dll.a")

            if(NOT EXISTS "${_ORT_DLLA}")
                # Try gendef first (from mingw-w64-tools), fall back to creating minimal def
                find_program(GENDEF_EXE gendef)
                find_program(DLLTOOL_EXE dlltool)

                if(GENDEF_EXE AND DLLTOOL_EXE)
                    execute_process(
                        COMMAND "${GENDEF_EXE}" "-" "${_ORT_DLL_PATH}"
                        OUTPUT_FILE "${_ORT_DEF}"
                        RESULT_VARIABLE _gendef_result
                    )
                    if(_gendef_result EQUAL 0)
                        execute_process(
                            COMMAND "${DLLTOOL_EXE}" -d "${_ORT_DEF}"
                                    -l "${_ORT_DLLA}" -D "onnxruntime.dll"
                            RESULT_VARIABLE _dlltool_result
                        )
                        if(_dlltool_result EQUAL 0)
                            message(STATUS "Generated MinGW import lib: ${_ORT_DLLA}")
                        endif()
                    endif()
                endif()
            endif()

            if(EXISTS "${_ORT_DLLA}")
                set_target_properties(onnxruntime_imported PROPERTIES
                    IMPORTED_IMPLIB "${_ORT_DLLA}"
                )
            else()
                # Fallback: MinGW can link directly against a DLL
                message(STATUS "MinGW: linking directly against onnxruntime.dll")
                set_target_properties(onnxruntime_imported PROPERTIES
                    IMPORTED_IMPLIB "${_ORT_DLL_PATH}"
                )
            endif()
        endif()
    else()
        # MSVC: use the .lib import library
        if(_ORT_LIB)
            list(GET _ORT_LIB 0 _ORT_LIB_PATH)
            set_target_properties(onnxruntime_imported PROPERTIES
                IMPORTED_IMPLIB "${_ORT_LIB_PATH}"
            )
        endif()
    endif()
else()
    # Linux / non-Windows
    find_library(_ORT_SO
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    if(_ORT_SO)
        set_target_properties(onnxruntime_imported PROPERTIES
            IMPORTED_LOCATION "${_ORT_SO}"
        )
    endif()
endif()

set_target_properties(onnxruntime_imported PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# ─────────────────────────────────────────────
# OpenCV - always use FetchContent unless user explicitly provides OPENCV_DIR
# ─────────────────────────────────────────────
set(_OPENCV_USE_FETCH TRUE)

if(OPENCV_DIR AND EXISTS "${OPENCV_DIR}")
    set(OpenCV_DIR "${OPENCV_DIR}")
    find_package(OpenCV QUIET COMPONENTS core imgproc)
    if(OpenCV_FOUND)
        set(_OPENCV_USE_FETCH FALSE)
        message(STATUS "OpenCV found at user-specified path: ${OpenCV_VERSION}")
    endif()
endif()

if(_OPENCV_USE_FETCH)
    message(STATUS "OpenCV: fetching via FetchContent (core + imgproc only)...")
    set(BUILD_LIST "core,imgproc" CACHE STRING "" FORCE)
    set(BUILD_opencv_apps       OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_calib3d    OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_dnn        OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_features2d OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_flann      OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_gapi       OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_highgui    OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_java       OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_js         OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_ml         OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_objdetect  OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_photo      OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3    OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_stitching  OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_ts         OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_video      OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_videoio    OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS             OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS        OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES          OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS              OFF CACHE BOOL "" FORCE)
    set(BUILD_WITH_STATIC_CRT   OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
    set(OPENCV_GENERATE_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(WITH_FFMPEG             OFF CACHE BOOL "" FORCE)
    set(WITH_GSTREAMER          OFF CACHE BOOL "" FORCE)
    set(WITH_GTK                OFF CACHE BOOL "" FORCE)
    set(WITH_VTK                OFF CACHE BOOL "" FORCE)
    set(WITH_OPENEXR            OFF CACHE BOOL "" FORCE)
    set(WITH_JASPER             OFF CACHE BOOL "" FORCE)
    set(WITH_PROTOBUF           OFF CACHE BOOL "" FORCE)
    set(WITH_QUIRC              OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG        4.10.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(opencv)

    # Include paths for FetchContent-built OpenCV.
    # IMPORTANT: OpenCV generates opencv2/opencv_modules.hpp and cvconfig.h
    # in CMAKE_BINARY_DIR (top-level build dir), NOT in opencv_BINARY_DIR.
    # This is because OpenCV's CMakeLists.txt uses CMAKE_BINARY_DIR for
    # OPENCV_CONFIG_FILE_INCLUDE_DIR.
    set(OpenCV_INCLUDE_DIRS
        "${opencv_SOURCE_DIR}/include"
        "${opencv_SOURCE_DIR}/modules/core/include"
        "${opencv_SOURCE_DIR}/modules/imgproc/include"
        "${opencv_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}"
    )
    set(OpenCV_LIBS opencv_core opencv_imgproc)

    # Debug: verify paths exist
    message(STATUS "OpenCV source dir: ${opencv_SOURCE_DIR}")
    message(STATUS "OpenCV binary dir: ${opencv_BINARY_DIR}")
    foreach(_dir ${OpenCV_INCLUDE_DIRS})
        if(EXISTS "${_dir}")
            message(STATUS "  [OK] ${_dir}")
        else()
            message(WARNING "  [MISSING] ${_dir}")
        endif()
    endforeach()
    if(EXISTS "${opencv_SOURCE_DIR}/modules/core/include/opencv2/core.hpp")
        message(STATUS "  [OK] opencv2/core.hpp found")
    else()
        message(WARNING "  [MISSING] opencv2/core.hpp NOT found!")
    endif()
    message(STATUS "OpenCV fetched and configured (core + imgproc only)")
endif()

# ─────────────────────────────────────────────
# Windows SDK (DXGI, D3D11, XInput)
# ─────────────────────────────────────────────
# These ship with the Windows SDK / MSVC.
# For MinGW, the Windows API headers and libs are provided by mingw-w64.

# ─────────────────────────────────────────────
# CUDA (optional – only for ONNX Runtime EP linkage hints)
# ─────────────────────────────────────────────
if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    else()
        message(WARNING "CUDA Toolkit not found – ONNX Runtime will still "
                        "attempt to load the CUDA EP at runtime via its DLLs.")
    endif()
endif()

# ─────────────────────────────────────────────
# Source files
# ─────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/capture/dxgi_capture.cpp
    src/inference/onnx_detector.cpp
    src/tracking/kalman_tracker.cpp
    src/input/gamepad_controller.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${VIGEMCLIENT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Debug: show all include directories for the main target
get_target_property(_ALL_INCS ${PROJECT_NAME} INCLUDE_DIRECTORIES)
message(STATUS "Target include directories: ${_ALL_INCS}")

# ─────────────────────────────────────────────
# Link libraries
# ─────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    onnxruntime_imported
    ${VIGEMCLIENT_LIB}
    ${OpenCV_LIBS}
)

# Windows system libraries (works for both MSVC and MinGW)
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        d3d11
        dxgi
        xinput
        setupapi
    )
endif()

# ─────────────────────────────────────────────
# Compile definitions
# ─────────────────────────────────────────────
if(USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
endif()
if(USE_TENSORRT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_TENSORRT=1)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE UNICODE _UNICODE)

# Prevent Windows.h from defining min/max macros that conflict with
# std::min, std::max, std::clamp (causes C2589/C2059 errors).
# WIN32_LEAN_AND_MEAN reduces Windows.h bloat and speeds compilation.
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# ─────────────────────────────────────────────
# Compiler-specific flags
# ─────────────────────────────────────────────
# Release optimization flags use generator expressions so they NEVER
# apply to Debug builds (avoids /RTC1 vs /O2 conflict).
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        $<$<CONFIG:Release>:/O2 /Oi /GL /fp:fast /arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Oi /fp:fast /arch:AVX2>
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        /SUBSYSTEM:CONSOLE
        $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
    )
elseif(MINGW)
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math -flto>
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        -static-libgcc -static-libstdc++
        $<$<CONFIG:Release>:-flto>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math -flto>
    )
endif()

# ─────────────────────────────────────────────
# Post-build: copy ALL required DLLs next to executable
# ─────────────────────────────────────────────
if(WIN32)
    # ONNX Runtime DLLs
    if(ONNXRUNTIME_ROOT)
        file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
        foreach(dll ${ORT_DLLS})
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll}"
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            )
        endforeach()
    endif()

    # OpenCV DLLs (when built as shared libs via FetchContent)
    if(_OPENCV_USE_FETCH)
        foreach(_cv_target ${OpenCV_LIBS})
            if(TARGET ${_cv_target})
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:${_cv_target}>"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                )
            endif()
        endforeach()
    endif()
endif()

message(STATUS "──────────────────────────────────────")
message(STATUS "  RealtimeObjectTracker configuration")
message(STATUS "  Compiler     : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Generator    : ${CMAKE_GENERATOR}")
message(STATUS "  CUDA EP      : ${USE_CUDA}")
message(STATUS "  TensorRT EP  : ${USE_TENSORRT}")
message(STATUS "  ONNX Runtime : ${ONNXRUNTIME_ROOT}")
message(STATUS "  OpenCV       : ${OpenCV_DIR}")
message(STATUS "──────────────────────────────────────")
