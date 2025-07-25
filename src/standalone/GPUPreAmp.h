#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <functional>

#if GPU_AUDIO_AVAILABLE
// GPU Audio SDK headers will be included here when available
// #include <gpuaudio/gpuaudio.h>
#endif

//==============================================================================
/**
    GPUPreAmp - Standalone GPU Processing Test Harness
    
    This standalone application tests GPU Audio SDK integration independently
    of the JUCE plugin complexity. Used for Milestone 2 validation.
    
    Features:
    - Command-line WAV file processing
    - Simple gain processing on GPU
    - Performance benchmarking
    - Graceful CPU fallback
    - WAV output for verification
*/
class GPUPreAmp
{
public:
    GPUPreAmp();
    ~GPUPreAmp();

    //==============================================================================
    // Initialization and setup
    bool init(int sampleRate, int bufferSize);
    void shutdown();

    //==============================================================================
    // Processing
    bool processFile(const std::string& inputPath, const std::string& outputPath);
    void process(const float* input, float* output, int numSamples);
    
    //==============================================================================
    // Parameter control
    void setGain(float gainDB);
    
    //==============================================================================
    // Status and diagnostics
    bool isGPUAvailable() const { return gpuAvailable; }
    bool isUsingGPU() const { return usingGPU; }
    float getCurrentCPULoad() const { return cpuLoad; }
    double getLastProcessingTimeMs() const { return lastProcessingTimeMs; }
    
    //==============================================================================
    // Benchmarking
    void runBenchmark();
    void printSystemInfo() const;

private:
    //==============================================================================
    // Processing state
    int sampleRate = 44100;
    int bufferSize = 512;
    bool isInitialized = false;
    
    // GPU/CPU processing flags
    bool gpuAvailable = false;
    bool usingGPU = false;
    float cpuLoad = 0.0f;
    double lastProcessingTimeMs = 0.0;
    
    // Parameters
    float gainDB = 0.0f;
    float gainLinear = 1.0f;
    
    // Performance tracking
    mutable std::chrono::high_resolution_clock::time_point startTime;
    mutable std::chrono::high_resolution_clock::time_point endTime;
    
    //==============================================================================
    // GPU processing (when available)
    bool initializeGPU();
    void processBlockGPU(const float* input, float* output, int numSamples);
    void shutdownGPU();
    
    // CPU processing (fallback)
    void processBlockCPU(const float* input, float* output, int numSamples);
    
    // WAV file I/O
    bool loadWAV(const std::string& filename, std::vector<float>& audioData, int& channels);
    bool saveWAV(const std::string& filename, const std::vector<float>& audioData, 
                 int channels, int sampleRate);
    
    // Utilities
    void updateGainLinear();
    double measureProcessingTime(std::function<void()> processFunc) const;

#if GPU_AUDIO_AVAILABLE
    // GPU Audio SDK objects will go here
    // GPUDSP::Backend* gpuBackend = nullptr;
    // GPUDSP::Processor* gainProcessor = nullptr;
#endif
};