#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <chrono>
#include <memory>

#include "Neve1073DSP.h"

#if GPU_AUDIO_AVAILABLE
#include "../gpu/CudaProcessor.h"
#endif

//==============================================================================
/**
    N73GPUCore - Core audio processing class for the N73GPU plugin
    
    Milestone 4: Now includes complete Neve 1073 CPU reference implementation
    with authentic EQ, filters, and saturation modeling.
    
    Features:
    - Complete Neve 1073 DSP model (CPU reference)
    - GPU acceleration via CUDA (when available)
    - Automatic CPU fallback
    - Performance monitoring
    - Parameter automation support
*/
class N73GPUCore
{
public:
    N73GPUCore();
    ~N73GPUCore();

    //==============================================================================
    // Initialization
    void initialize();
    void prepareToPlay(double sampleRate, int bufferSize);
    void releaseResources();

    //==============================================================================
    // Processing
    void processBlock(juce::AudioBuffer<float>& buffer);
    void processMidiMessage(const juce::MidiMessage& message);

    //==============================================================================
    // Neve 1073 Parameter Control (Milestone 4)
    void setInputGain(float gainDb);        // -20dB to +80dB (authentic range)
    void setHighShelfGain(float gainDb);    // ±16dB at 12kHz
    void setMidFrequency(float frequency);  // 360Hz to 7.2kHz
    void setMidGain(float gainDb);          // ±18dB
    void setMidQ(float q);                  // Q factor for mid EQ
    void setLowShelfFrequency(int index);   // 0=35Hz, 1=60Hz, 2=110Hz, 3=220Hz
    void setLowShelfGain(float gainDb);     // ±16dB
    void setHPFFrequency(int freqIndex);    // 0=off, 1=50Hz, 2=80Hz, 3=160Hz, 4=300Hz
    void setOutputLevel(float levelDb);     // -20dB to +10dB
    
    //==============================================================================
    // EQ Bypass Controls
    void setHighShelfBypass(bool bypass);
    void setMidEqBypass(bool bypass);
    void setLowShelfBypass(bool bypass);
    void setHighPassBypass(bool bypass);
    
    //==============================================================================
    // Advanced Controls (Milestone 4)
    void setSaturationAmount(float amount);     // 0.0 to 1.0
    void setTransformerModeling(bool enable);
    void setHarmonicContent(float amount);      // Harmonic distortion amount
    
    //==============================================================================
    // GPU Performance and diagnostics
    void enableGPUProcessing(bool enable);
    void setGPUFallbackMode(bool enableFallback);
    bool runGPUDiagnostics();
    void printGPUStatus() const;
    
    //==============================================================================
    // Status and analysis
    bool isGPUAvailable() const { return gpuAvailable; }
    bool isUsingGPU() const { return usingGPU; }
    float getCPULoad() const { return cpuLoad; }
    double getLastProcessingTimeMs() const { return lastProcessingTimeMs; }
    
    // Neve 1073 Analysis (Milestone 4)
    float getCurrentSaturation() const;
    float getTotalHarmonicDistortion() const;
    std::array<float, 5> getEQResponse(float frequency) const;

private:
    //==============================================================================
    // Processing state
    double sampleRate = 44100.0;
    int bufferSize = 512;
    bool isInitialized = false;
    
    // GPU/CPU processing flags
    bool gpuAvailable = false;
    bool usingGPU = false;
    bool gpuFallbackEnabled = true;
    float cpuLoad = 0.0f;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point processingStartTime;
    std::chrono::high_resolution_clock::time_point processingEndTime;
    double lastProcessingTimeMs = 0.0;
    
    //==============================================================================
    // DSP Processing Engines
    
    // Milestone 4: Neve 1073 CPU Reference Implementation
    std::unique_ptr<Neve1073DSP> neve1073Processor;
    
    // GPU Processing (when available)
#if GPU_AUDIO_AVAILABLE
    std::unique_ptr<CudaProcessor> cudaProcessor;
#else
    void* cudaProcessor = nullptr; // Placeholder when CUDA not available
#endif

    //==============================================================================
    // Legacy parameter storage (for compatibility)
    float inputGain = 0.0f;
    float highShelfGain = 0.0f;
    float midFrequency = 1500.0f;
    float midGain = 0.0f;
    int lowShelfFreqIndex = 2; // 110Hz
    float lowShelfGain = 0.0f;
    int hpfFreqIndex = 0; // Off
    float outputLevel = 0.0f;

    //==============================================================================
    // Processing methods
    void initializeGPU();
    void processBlockGPU(juce::AudioBuffer<float>& buffer);
    void processBlockCPU(juce::AudioBuffer<float>& buffer);
    
    // MIDI handling
    void handleControlChange(int controller, int value);
    void handleSysEx(const juce::MidiMessage& message);
};