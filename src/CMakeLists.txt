# Source CMakeLists.txt for N73GPU

# Only proceed if JUCE is available
if(TARGET juce::juce_audio_plugin_client)
    
    # Define plugin sources
    set(PLUGIN_SOURCES
        plugin/PluginProcessor.cpp
        plugin/PluginProcessor.h
        plugin/PluginEditor.cpp  
        plugin/PluginEditor.h
        dsp/N73GPUCore.cpp
        dsp/N73GPUCore.h
        gpu/CudaProcessor.cpp
    )

    # Create the plugin target
    juce_add_plugin(N73GPU
        COMPANY_NAME "KZRACKS"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT TRUE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD FALSE
        PLUGIN_MANUFACTURER_CODE KzRk
        PLUGIN_CODE N73G
        FORMATS VST3 ${AU_FORMAT}
        PRODUCT_NAME "N73GPU"
        BUNDLE_ID com.kzracks.n73gpu
        DESCRIPTION "Neve 1073 GPU-Accelerated Emulation"
    )

    # Set AU format based on platform
    if(BUILD_AU AND APPLE)
        set(AU_FORMAT AU)
    else()
        set(AU_FORMAT "")
    endif()

    # Add source files to plugin
    target_sources(N73GPU PRIVATE ${PLUGIN_SOURCES})

    # Link JUCE modules
    target_link_libraries(N73GPU
        PRIVATE
            juce::juce_audio_utils
            juce::juce_audio_plugin_client
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags
    )

    # Preprocessor definitions
    target_compile_definitions(N73GPU
        PUBLIC
            JUCE_WEB_BROWSER=0
            JUCE_USE_CURL=0
            JUCE_VST3_CAN_REPLACE_VST2=0
            JUCE_DISPLAY_SPLASH_SCREEN=0
            JUCE_REPORT_APP_USAGE=0
    )

    # Configure CUDA processing if available
    if(CUDA_AVAILABLE)
        # Set CUDA properties for the processor file
        set_source_files_properties(gpu/CudaProcessor.cpp PROPERTIES LANGUAGE CUDA)
        
        # Find and link CUDA libraries
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(N73GPU PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
        )
        
        # CUDA compile definitions and flags - apply early and with PUBLIC scope
        target_compile_definitions(N73GPU PUBLIC GPU_AUDIO_AVAILABLE=1)
        
        # Also set as a preprocessor definition for the entire project
        add_compile_definitions(GPU_AUDIO_AVAILABLE=1)
        
        target_compile_options(N73GPU PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        
        message(STATUS "N73GPU plugin will use CUDA GPU processing")
        message(STATUS "GPU_AUDIO_AVAILABLE=1 set for all targets")
    else()
        target_compile_definitions(N73GPU PUBLIC GPU_AUDIO_AVAILABLE=0)
        add_compile_definitions(GPU_AUDIO_AVAILABLE=0)
        message(STATUS "N73GPU plugin will use CPU-only processing")
    endif()

else()
    message(WARNING "JUCE not available - skipping plugin build")
endif()

# Build standalone GPUPreAmp test application (Milestone 2)
if(BUILD_STANDALONE)
    set(GPUPREAMP_SOURCES
        standalone/GPUPreAmp.cpp
        standalone/GPUPreAmp.h
        standalone/main.cpp
    )
    
    add_executable(GPUPreAmp ${GPUPREAMP_SOURCES})
    
    # Set C++17 standard (avoiding target_compile_features MSVC compatibility issue)
    set_property(TARGET GPUPreAmp PROPERTY CXX_STANDARD 17)
    set_property(TARGET GPUPreAmp PROPERTY CXX_STANDARD_REQUIRED ON)
    
    # Include directories
    target_include_directories(GPUPreAmp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/dsp
        ${CMAKE_CURRENT_SOURCE_DIR}/standalone
    )
    
    # Preprocessor definitions
    target_compile_definitions(GPUPreAmp PRIVATE
        GPU_AUDIO_AVAILABLE=0  # Will be 1 when GPU Audio SDK is integrated
        _USE_MATH_DEFINES      # For M_PI on Windows
    )
    
    # Include GPU Audio SDK if available
    if(EXISTS "${GPU_AUDIO_SDK_PATH}")
        target_include_directories(GPUPreAmp PRIVATE "${GPU_AUDIO_SDK_PATH}/include")
        target_compile_definitions(GPUPreAmp PRIVATE GPU_AUDIO_AVAILABLE=1)
    endif()
    
    message(STATUS "Standalone GPUPreAmp test application will be built")
else()
    message(STATUS "Standalone GPUPreAmp test application disabled")
endif()