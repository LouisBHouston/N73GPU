cmake_minimum_required(VERSION 3.29)

# Set policies to handle MSVC/JUCE compatibility issues
if(POLICY CMP0157)
    cmake_policy(SET CMP0157 NEW)
endif()
if(POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif()

project(N73GPU VERSION 1.0.0 LANGUAGES CXX)

# Enable C++17 with MSVC compatibility - ensure it applies to ALL targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Force C++17 for all targets (including JUCE build tools)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
set(JUCE_CXX_STANDARD 17)

# Fix MSVC compiler features compatibility issues
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Disable problematic compiler features detection
    set(CMAKE_CXX_COMPILE_FEATURES "")
    
    # Strip /GL from anything that might hit nvcc (robust CUDA/MSVC compatibility fix)
    foreach (v CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE CMAKE_CUDA_FLAGS_RELEASE)
      if (DEFINED ${v})
        string(REPLACE "/GL" "" ${v} "${${v}}")
        set(${v} "${${v}}" CACHE STRING "" FORCE)
      endif()
    endforeach()
    
    # Prevent JUCE from using target_compile_features
    set(JUCE_DONT_SET_USING_NAMESPACE_JUCE ON)
    add_compile_definitions(JUCE_DONT_SET_USING_NAMESPACE_JUCE=1)
endif()

# Enable CUDA now that Milestone 3 is validated - configure for actual GPU processing
message(STATUS "Checking for CUDA toolkit at: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
if(EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
    message(STATUS "CUDA toolkit directory found!")
    
    # Set CUDA toolkit path explicitly
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
    
    # Enable CUDA language
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA architectures for modern GPUs (fix runtime device detection)
    set(CMAKE_CUDA_ARCHITECTURES 52;61;70;75;80;86;89)
    
    set(CUDA_AVAILABLE ON)
    message(STATUS "CUDA_AVAILABLE set to: ${CUDA_AVAILABLE}")
    
    # Set CUDA-specific optimization flags (after /GL removal above)
    if(MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/std:c++17")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
        set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    endif()
    
    message(STATUS "CUDA enabled for GPU processing with toolkit at ${CUDAToolkit_ROOT}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(CUDA_AVAILABLE OFF)
    message(STATUS "CUDA toolkit not found - using CPU fallback")
    message(STATUS "CUDA_AVAILABLE set to: ${CUDA_AVAILABLE}")
endif()

# Build configuration options
option(BUILD_STANDALONE "Build standalone test app" ON)
option(BUILD_VST3 "Build VST3 plugin" ON)
option(BUILD_AU "Build AU plugin" OFF)
option(GPU_FALLBACK "Include CPU fallback" ON)

# Find required packages (PkgConfig not needed on Windows)

# JUCE configuration
set(JUCE_ENABLE_MODULE_SOURCE_GROUPS ON)

# Add JUCE
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/CMakeLists.txt")
    add_subdirectory(JUCE)
    message(STATUS "JUCE found and added from root directory")
else()
    message(WARNING "JUCE not found at JUCE/ - please install JUCE")
endif()

# GPU Audio SDK temporarily disabled due to CMake compatibility issues
# TODO: Re-enable when build issues are resolved
set(GPU_AUDIO_AVAILABLE OFF)
message(STATUS "GPU Audio SDK temporarily disabled - using direct CUDA implementation")

# Add subdirectories
add_subdirectory(src)

# Configuration summary
message(STATUS "N73GPU Configuration:")
message(STATUS "  Build VST3: ${BUILD_VST3}")
message(STATUS "  Build AU: ${BUILD_AU}")
message(STATUS "  Build Standalone: ${BUILD_STANDALONE}")
message(STATUS "  GPU Fallback: ${GPU_FALLBACK}")