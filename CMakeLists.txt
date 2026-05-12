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

# ─────────────────────────────────────────────
# FetchContent: automatically download headers / SDKs when not provided
# ─────────────────────────────────────────────
include(FetchContent)

# --- ViGEmClient (header-only client + import lib) ---
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
if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
    )
else()
    message(FATAL_ERROR
        "ONNX Runtime SDK path not set. "
        "Please specify -DONNXRUNTIME_ROOT=<path> pointing to the extracted "
        "onnxruntime-win-x64-gpu-<version> directory."
    )
endif()

# ─────────────────────────────────────────────
# OpenCV
# ─────────────────────────────────────────────
if(OPENCV_DIR AND EXISTS "${OPENCV_DIR}")
    set(OpenCV_DIR "${OPENCV_DIR}")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ─────────────────────────────────────────────
# Windows SDK (DXGI, D3D11, XInput)
# ─────────────────────────────────────────────
# These ship with the Windows SDK / MSVC; no find_package needed.

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

# ─────────────────────────────────────────────
# Link libraries
# ─────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    # ONNX Runtime
    ${ONNXRUNTIME_LIB}
    # ViGEmClient
    ${VIGEMCLIENT_LIB}
    # OpenCV
    ${OpenCV_LIBS}
    # Windows SDK libs (only relevant on MSVC / Windows)
    $<$<PLATFORM_ID:Windows>:d3d11>
    $<$<PLATFORM_ID:Windows>:dxgi>
    $<$<PLATFORM_ID:Windows>:xinput>
    $<$<PLATFORM_ID:Windows>:setupapi>
)

# ─────────────────────────────────────────────
# Compile definitions
# ─────────────────────────────────────────────
if(USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA=1)
endif()
if(USE_TENSORRT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_TENSORRT=1)
endif()

# Unicode support
target_compile_definitions(${PROJECT_NAME} PRIVATE UNICODE _UNICODE)

# MSVC-specific flags
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /O2 /fp:fast /arch:AVX2)
    target_link_options(${PROJECT_NAME} PRIVATE /SUBSYSTEM:CONSOLE)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O3 -mavx2)
endif()

# ─────────────────────────────────────────────
# Post-build: copy ONNX Runtime DLLs next to executable
# ─────────────────────────────────────────────
if(WIN32 AND ONNXRUNTIME_ROOT)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    foreach(dll ${ORT_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach()
endif()

message(STATUS "──────────────────────────────────────")
message(STATUS "  RealtimeObjectTracker configuration")
message(STATUS "  CUDA EP      : ${USE_CUDA}")
message(STATUS "  TensorRT EP  : ${USE_TENSORRT}")
message(STATUS "  ONNX Runtime : ${ONNXRUNTIME_ROOT}")
message(STATUS "  OpenCV       : ${OpenCV_DIR}")
message(STATUS "──────────────────────────────────────")
