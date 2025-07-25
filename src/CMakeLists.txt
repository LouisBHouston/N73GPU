# N73GPU Plugin Source CMakeLists.txt
# Milestone 4: Includes complete Neve 1073 CPU reference implementation

# Define source files for the plugin
set(N73GPU_SOURCES
    PluginProcessor.cpp
    PluginEditor.cpp
    dsp/N73GPUCore.cpp
    dsp/Neve1073DSP.cpp
    gpu/CudaProcessor.cpp
)

# Define header files for IDE organization
set(N73GPU_HEADERS
    PluginProcessor.h
    PluginEditor.h
    dsp/N73GPUCore.h
    dsp/Neve1073DSP.h
    gpu/CudaProcessor.h
)

# Add the plugin target using JUCE
juce_add_plugin(N73GPU
    COMPANY_NAME KZRACKS
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE  # Disabled to avoid Windows path issues
    PLUGIN_MANUFACTURER_CODE KZRA
    PLUGIN_CODE N73G
    FORMATS VST3 Standalone
    PRODUCT_NAME "N73GPU"
)

# Add source files to the plugin
target_sources(N73GPU PRIVATE ${N73GPU_SOURCES} ${N73GPU_HEADERS})

# Add JUCE modules
target_link_libraries(N73GPU
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Set C++ standard
set_property(TARGET N73GPU PROPERTY CXX_STANDARD 17)

# GPU Audio SDK and CUDA support (Milestone 3+)
if(CUDA_FOUND AND GPU_AUDIO_AVAILABLE)
    message(STATUS "N73GPU: CUDA found, enabling GPU processing")
    
    # Enable CUDA language
    enable_language(CUDA)
    
    # Set CUDA architectures for modern GPUs
    set_property(TARGET N73GPU PROPERTY CUDA_ARCHITECTURES 75 80 86 89)
    
    # Add CUDA-specific flags
    target_compile_definitions(N73GPU PRIVATE GPU_AUDIO_AVAILABLE=1)
    target_link_libraries(N73GPU PRIVATE ${CUDA_LIBRARIES})
    
    # Ensure CUDA files use the CUDA compiler
    set_source_files_properties(gpu/CudaProcessor.cpp PROPERTIES LANGUAGE CUDA)
    
    message(STATUS "N73GPU: GPU processing enabled with CUDA support")
else()
    message(STATUS "N73GPU: CUDA not found or disabled, using CPU-only processing")
    target_compile_definitions(N73GPU PRIVATE GPU_AUDIO_AVAILABLE=0)
endif()

# Compiler-specific definitions
target_compile_definitions(N73GPU
    PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_REPORT_APP_USAGE=0
        JUCE_ALSA=1
        JUCE_DIRECTSOUND=1
        JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
)

# Include directories
target_include_directories(N73GPU PRIVATE .)

# GPUPreAmp standalone test harness (Milestone 2)
add_executable(GPUPreAmp
    GPUPreAmp.cpp
    main.cpp
    dsp/N73GPUCore.cpp
    dsp/Neve1073DSP.cpp
    gpu/CudaProcessor.cpp
)

# Link GPUPreAmp with JUCE audio basics
target_link_libraries(GPUPreAmp
    PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_core
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
)

# Set C++ standard for GPUPreAmp
set_property(TARGET GPUPreAmp PROPERTY CXX_STANDARD 17)

# GPU support for GPUPreAmp
if(CUDA_FOUND AND GPU_AUDIO_AVAILABLE)
    target_compile_definitions(GPUPreAmp PRIVATE GPU_AUDIO_AVAILABLE=1)
    target_link_libraries(GPUPreAmp PRIVATE ${CUDA_LIBRARIES})
    set_source_files_properties(gpu/CudaProcessor.cpp PROPERTIES LANGUAGE CUDA)
else()
    target_compile_definitions(GPUPreAmp PRIVATE GPU_AUDIO_AVAILABLE=0)
endif()

# Include directories for GPUPreAmp
target_include_directories(GPUPreAmp PRIVATE .)