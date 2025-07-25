#include "CudaProcessor.h"
#include <iostream>
#include <chrono>
#include <cstring>

// CUDA kernel for gain processing
#if GPU_AUDIO_AVAILABLE
__global__ void applyGainKernel(float* audioData, int numSamples, float gainValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples)
    {
        audioData[idx] *= gainValue;
    }
}

__global__ void processBufferKernel(float* leftChannel, float* rightChannel, 
                                   int numSamples, float inputGain, float outputLevel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples)
    {
        // Apply input gain
        leftChannel[idx] *= inputGain;
        rightChannel[idx] *= inputGain;
        
        // Apply basic saturation modeling (soft clipping)
        leftChannel[idx] = tanh(leftChannel[idx] * 0.8f);
        rightChannel[idx] = tanh(rightChannel[idx] * 0.8f);
        
        // Apply output level
        leftChannel[idx] *= outputLevel;
        rightChannel[idx] *= outputLevel;
    }
}
#endif

CudaProcessor::CudaProcessor()
    : gpuInitialized(false)
    , profilingEnabled(false)
    , lastProcessingTimeMs(0.0)
#ifdef GPU_AUDIO_AVAILABLE
    , d_audioBuffer(nullptr)
    , allocatedBufferSize(0)
    , cudaStream(0)
#endif
{
}

CudaProcessor::~CudaProcessor()
{
    shutdown();
}

bool CudaProcessor::initialize()
{
#if GPU_AUDIO_AVAILABLE
    std::cout << "[CudaProcessor] Initializing CUDA GPU processing..." << std::endl;
    
    // Check for CUDA-capable devices
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0)
    {
        std::cout << "[CudaProcessor] No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    // Set the device (use device 0)
    error = cudaSetDevice(0);
    if (!checkCudaError("cudaSetDevice"))
        return false;
    
    // Get device properties
    error = cudaGetDeviceProperties(&deviceProperties, 0);
    if (!checkCudaError("cudaGetDeviceProperties"))
        return false;
    
    // Create CUDA stream for async processing
    error = cudaStreamCreate(&cudaStream);
    if (!checkCudaError("cudaStreamCreate"))
        return false;
    
    gpuInitialized = true;
    std::cout << "[CudaProcessor] GPU initialized successfully: " << deviceProperties.name << std::endl;
    std::cout << "[CudaProcessor] Compute capability: " << deviceProperties.major << "." << deviceProperties.minor << std::endl;
    std::cout << "[CudaProcessor] Global memory: " << (deviceProperties.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
    
    return true;
#else
    std::cout << "[CudaProcessor] CUDA not available - GPU processing disabled" << std::endl;
    return false;
#endif
}

void CudaProcessor::shutdown()
{
#if GPU_AUDIO_AVAILABLE
    if (gpuInitialized)
    {
        freeGPUMemory();
        
        if (cudaStream)
        {
            cudaStreamDestroy(cudaStream);
            cudaStream = 0;
        }
        
        cudaDeviceReset();
        gpuInitialized = false;
        std::cout << "[CudaProcessor] GPU processing shutdown complete" << std::endl;
    }
#endif
}

bool CudaProcessor::processGain(float* audioData, int numSamples, float gainValue)
{
#if GPU_AUDIO_AVAILABLE
    if (!gpuInitialized || !audioData || numSamples <= 0)
        return false;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    size_t dataSize = numSamples * sizeof(float);
    
    // Allocate GPU memory if needed
    if (!allocateGPUMemory(dataSize))
        return false;
    
    // Copy data to GPU
    cudaError_t error = cudaMemcpyAsync(d_audioBuffer, audioData, dataSize, 
                                       cudaMemcpyHostToDevice, cudaStream);
    if (!checkCudaError("cudaMemcpyAsync H2D"))
        return false;
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;
    
    applyGainKernel<<<gridSize, blockSize, 0, cudaStream>>>(d_audioBuffer, numSamples, gainValue);
    
    if (!checkCudaError("applyGainKernel launch"))
        return false;
    
    // Copy result back to host
    error = cudaMemcpyAsync(audioData, d_audioBuffer, dataSize, 
                           cudaMemcpyDeviceToHost, cudaStream);
    if (!checkCudaError("cudaMemcpyAsync D2H"))
        return false;
    
    // Synchronize stream
    error = cudaStreamSynchronize(cudaStream);
    if (!checkCudaError("cudaStreamSynchronize"))
        return false;
    
    if (profilingEnabled)
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        lastProcessingTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    }
    
    return true;
#else
    // CPU fallback
    for (int i = 0; i < numSamples; ++i)
    {
        audioData[i] *= gainValue;
    }
    return true;
#endif
}

bool CudaProcessor::processBuffer(float* leftChannel, float* rightChannel, int numSamples, 
                                 float inputGain, float outputLevel)
{
#if GPU_AUDIO_AVAILABLE
    if (!gpuInitialized || !leftChannel || !rightChannel || numSamples <= 0)
        return false;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    size_t channelSize = numSamples * sizeof(float);
    size_t totalSize = channelSize * 2;
    
    // Allocate GPU memory if needed
    if (!allocateGPUMemory(totalSize))
        return false;
    
    float* d_rightChannel = d_audioBuffer + numSamples;
    
    // Copy data to GPU
    cudaError_t error = cudaMemcpyAsync(d_audioBuffer, leftChannel, channelSize, 
                                       cudaMemcpyHostToDevice, cudaStream);
    if (!checkCudaError("cudaMemcpyAsync left H2D"))
        return false;
    
    error = cudaMemcpyAsync(d_rightChannel, rightChannel, channelSize, 
                           cudaMemcpyHostToDevice, cudaStream);
    if (!checkCudaError("cudaMemcpyAsync right H2D"))
        return false;
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;
    
    processBufferKernel<<<gridSize, blockSize, 0, cudaStream>>>(
        d_audioBuffer, d_rightChannel, numSamples, inputGain, outputLevel);
    
    if (!checkCudaError("processBufferKernel launch"))
        return false;
    
    // Copy results back to host
    error = cudaMemcpyAsync(leftChannel, d_audioBuffer, channelSize, 
                           cudaMemcpyDeviceToHost, cudaStream);
    if (!checkCudaError("cudaMemcpyAsync left D2H"))
        return false;
    
    error = cudaMemcpyAsync(rightChannel, d_rightChannel, channelSize, 
                           cudaMemcpyDeviceToHost, cudaStream);
    if (!checkCudaError("cudaMemcpyAsync right D2H"))
        return false;
    
    // Synchronize stream
    error = cudaStreamSynchronize(cudaStream);
    if (!checkCudaError("cudaStreamSynchronize"))
        return false;
    
    if (profilingEnabled)
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        lastProcessingTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    }
    
    return true;
#else
    // CPU fallback
    for (int i = 0; i < numSamples; ++i)
    {
        leftChannel[i] *= inputGain;
        rightChannel[i] *= inputGain;
        
        // Apply basic saturation
        leftChannel[i] = std::tanh(leftChannel[i] * 0.8f);
        rightChannel[i] = std::tanh(rightChannel[i] * 0.8f);
        
        leftChannel[i] *= outputLevel;
        rightChannel[i] *= outputLevel;
    }
    return true;
#endif
}

std::string CudaProcessor::getGPUInfo() const
{
#if GPU_AUDIO_AVAILABLE
    if (!gpuInitialized)
        return "GPU not initialized";
    
    return std::string(deviceProperties.name) + " (Compute " + 
           std::to_string(deviceProperties.major) + "." + 
           std::to_string(deviceProperties.minor) + ")";
#else
    return "CUDA not available";
#endif
}

bool CudaProcessor::allocateGPUMemory(size_t bufferSize)
{
#ifdef GPU_AUDIO_AVAILABLE
    if (allocatedBufferSize >= bufferSize)
        return true;  // Already have enough memory
    
    // Free existing memory
    freeGPUMemory();
    
    // Allocate new memory with some padding
    size_t newSize = bufferSize * 2;  // 2x for safety margin
#if GPU_AUDIO_AVAILABLE
    cudaError_t error = cudaMalloc(&d_audioBuffer, newSize);
    if (error != cudaSuccess)
    {
        std::cout << "[CudaProcessor] Failed to allocate GPU memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
#else
    std::cout << "[CudaProcessor] CUDA not available - cannot allocate GPU memory" << std::endl;
    return false;
#endif
    
    allocatedBufferSize = newSize;
    return true;
#else
    return false;
#endif
}

void CudaProcessor::freeGPUMemory()
{
#if GPU_AUDIO_AVAILABLE
    if (d_audioBuffer)
    {
        cudaFree(d_audioBuffer);
        d_audioBuffer = nullptr;
        allocatedBufferSize = 0;
    }
#endif
}

bool CudaProcessor::checkCudaError(const char* operation)
{
#if GPU_AUDIO_AVAILABLE
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "[CudaProcessor] CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
#else
    return true;  // Always return success when CUDA is disabled
#endif
}