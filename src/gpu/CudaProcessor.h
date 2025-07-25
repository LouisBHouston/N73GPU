#pragma once

#include <vector>
#include <memory>
#include <string>

#if GPU_AUDIO_AVAILABLE
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

/**
 * Minimal CUDA-based GPU processor for N73GPU
 * Provides basic GPU acceleration for audio processing without external SDK dependencies
 */
class CudaProcessor
{
public:
    CudaProcessor();
    ~CudaProcessor();
    
    // Initialize CUDA context and check GPU availability
    bool initialize();
    void shutdown();
    
    // GPU processing functions
    bool processGain(float* audioData, int numSamples, float gainValue);
    bool processBuffer(float* leftChannel, float* rightChannel, int numSamples, 
                      float inputGain, float outputLevel);
    
    // GPU status and diagnostics
    bool isGPUAvailable() const { return gpuInitialized; }
    std::string getGPUInfo() const;
    double getLastProcessingTimeMs() const { return lastProcessingTimeMs; }
    
    // Performance monitoring
    void enableProfiling(bool enable) { profilingEnabled = enable; }
    
private:
    bool gpuInitialized;
    bool profilingEnabled;
    double lastProcessingTimeMs;
    
    // Device management
    int deviceCount;
    int currentDevice;
    
#if GPU_AUDIO_AVAILABLE
    // CUDA-specific members when GPU is available
    cudaDeviceProp deviceProperties;
    float* d_audioBuffer;
    size_t allocatedBufferSize;
    cudaStream_t cudaStream;
#else
    // Fallback placeholders when CUDA is disabled
    int deviceProperties;  // Placeholder
    int cudaStream;        // Placeholder  
    float* d_audioBuffer;  // Placeholder
    size_t allocatedBufferSize;  // Placeholder
#endif
    
    // Internal helper functions
    bool allocateGPUMemory(size_t bufferSize);
    void freeGPUMemory();
    bool checkCudaError(const char* operation);
};