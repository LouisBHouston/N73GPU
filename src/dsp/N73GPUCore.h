#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <memory>
#include <chrono>
#include "../gpu/CudaProcessor.h"

#if GPU_AUDIO_AVAILABLE
// GPU Audio SDK headers will be included here when available
// #include <gpuaudio/gpuaudio.h>
#endif

//==============================================================================
/**
    N73GPUCore - Core DSP processing class for N73GPU plugin
    
    This class manages the main signal processing for the Neve 1073 emulation.
    
    Milestone 1: Simple passthrough processing
    Milestone 2: GPU Audio SDK integration testing  
    Milestone 4: Full CPU-based 1073 implementation
    Milestone 5: GPU-accelerated processing
*/
class N73GPUCore
{
public:
    N73GPUCore();
    ~N73GPUCore();

    //==============================================================================
    // Initialization and cleanup
    void initialize();
    void prepareToPlay(double sampleRate, int samplesPerBlock);
    void releaseResources();

    //==============================================================================
    // Processing
    void processBlock(juce::AudioBuffer<float>& buffer);
    void processMidiMessage(const juce::MidiMessage& message);

    //==============================================================================
    // Parameter control (for future hardware integration)
    void setInputGain(float gainDb);
    void setHighShelfGain(float gainDb);
    void setMidFrequency(float frequency);
    void setMidGain(float gainDb);
    void setLowShelfFrequency(int freqIndex); // 0=35Hz, 1=60Hz, 2=110Hz, 3=220Hz
    void setLowShelfGain(float gainDb);
    void setHPFFrequency(int freqIndex); // 0=Off, 1=50Hz, 2=80Hz, 3=160Hz, 4=300Hz
    void setOutputLevel(float levelDb);

    //==============================================================================
    // Status and diagnostics
    bool isGPUAvailable() const { return gpuAvailable; }
    bool isUsingGPU() const { return usingGPU; }
    float getCurrentCPULoad() const { return cpuLoad; }
    double getLastProcessingTimeMs() const { return lastProcessingTimeMs; }
    
    //==============================================================================
    // GPU Performance and diagnostics (Milestone 3)
    void enableGPUProcessing(bool enable = true);
    void setGPUFallbackMode(bool enableFallback = true);
    bool runGPUDiagnostics();
    void printGPUStatus() const;
    
private:
    //==============================================================================
    // Processing state
    double sampleRate = 44100.0;
    int bufferSize = 512;
    bool isInitialized = false;
    
    // GPU/CPU processing flags (Enhanced for Milestone 3)
    bool gpuAvailable = false;
    bool usingGPU = false;
    bool gpuFallbackEnabled = true;
    float cpuLoad = 0.0f;
    double lastProcessingTimeMs = 0.0;
    
    // GPU processing performance tracking
    mutable std::chrono::high_resolution_clock::time_point processingStartTime;
    mutable std::chrono::high_resolution_clock::time_point processingEndTime;
    
    // CUDA GPU processor
    std::unique_ptr<CudaProcessor> cudaProcessor; // (will be used in Milestone 4+)
    
    // Parameter values (will be used in Milestone 4+)
    float inputGain = 0.0f;
    float highShelfGain = 0.0f;
    float midFrequency = 1600.0f;
    float midGain = 0.0f;
    int lowShelfFreqIndex = 2; // 110Hz default
    float lowShelfGain = 0.0f;
    int hpfFreqIndex = 0; // Off default
    float outputLevel = 0.0f;
    
    //==============================================================================
    // GPU processing (Milestone 2+)
    void initializeGPU();
    void processBlockGPU(juce::AudioBuffer<float>& buffer);
    
    // CPU processing (Milestone 4+)
    void processBlockCPU(juce::AudioBuffer<float>& buffer);
    
    // MIDI handling for hardware integration
    void handleControlChange(int controller, int value);
    void handleSysEx(const juce::MidiMessage& message);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (N73GPUCore)
};