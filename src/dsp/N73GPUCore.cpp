#include "N73GPUCore.h"
#include <iostream>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
N73GPUCore::N73GPUCore()
{
    // Create CUDA processor when GPU support is available
#if GPU_AUDIO_AVAILABLE
    cudaProcessor = std::make_unique<CudaProcessor>();
    std::cout << "N73GPUCore: CUDA processor created" << std::endl;
#else
    cudaProcessor = nullptr;
    std::cout << "N73GPUCore: GPU processing not available, using CPU fallback" << std::endl;
#endif
}

N73GPUCore::~N73GPUCore()
{
    // Cleanup CUDA GPU resources
    if (cudaProcessor)
    {
        cudaProcessor->shutdown();
        std::cout << "N73GPUCore: CUDA processor shutdown complete" << std::endl;
    }
    releaseResources();
}

//==============================================================================
void N73GPUCore::initialize()
{
    // Initialize GPU processing if available
    
    // Initialize CUDA GPU processing
    if (cudaProcessor)
    {
        gpuAvailable = cudaProcessor->initialize();
        if (gpuAvailable)
        {
            cudaProcessor->enableProfiling(true);
            usingGPU = true;  // Enable GPU processing by default if available
            std::cout << "N73GPUCore: CUDA GPU processing initialized - " << cudaProcessor->getGPUInfo() << std::endl;
        }
        else
        {
            std::cout << "N73GPUCore: CUDA GPU not available, using CPU fallback" << std::endl;
        }
    }
    
    std::cout << "N73GPUCore initialized with sample rate: " << sampleRate 
              << ", buffer size: " << bufferSize 
              << ", GPU available: " << (gpuAvailable ? "YES" : "NO") << std::endl;
    
    isInitialized = true;
}

void N73GPUCore::prepareToPlay(double newSampleRate, int newBufferSize)
{
    sampleRate = newSampleRate;
    bufferSize = newBufferSize;
    
    // Reset processing state
    cpuLoad = 0.0f;
    
    // Initialize GPU processing if available and not already done
    if (!gpuAvailable)
    {
        initialize();
    }
}

void N73GPUCore::releaseResources()
{
    isInitialized = false;
    // GPU resources will be cleaned up here in future milestones
}

//==============================================================================
void N73GPUCore::processBlock(juce::AudioBuffer<float>& buffer)
{
    if (!isInitialized)
        return;
    
    // Milestone 3: Enhanced GPU processing with performance monitoring
    processingStartTime = std::chrono::high_resolution_clock::now();
    
    if (usingGPU && gpuAvailable)
    {
        processBlockGPU(buffer);
    }
    else
    {
        processBlockCPU(buffer);
    }
    
    // Update processing time measurement
    processingEndTime = std::chrono::high_resolution_clock::now();
    lastProcessingTimeMs = std::chrono::duration<double, std::milli>(processingEndTime - processingStartTime).count();
}

void N73GPUCore::processMidiMessage(const juce::MidiMessage& message)
{
    if (message.isControllerOfType(0))
    {
        // Handle MIDI CC messages for hardware integration
        handleControlChange(message.getControllerNumber(), message.getControllerValue());
    }
    else if (message.isSysEx())
    {
        // Handle SysEx messages for bidirectional communication
        handleSysEx(message);
    }
}

//==============================================================================
// Parameter setters (will be fully implemented in Milestone 4)
void N73GPUCore::setInputGain(float gainDb)
{
    inputGain = gainDb;
}

void N73GPUCore::setHighShelfGain(float gainDb)
{
    highShelfGain = gainDb;
}

void N73GPUCore::setMidFrequency(float frequency)
{
    midFrequency = frequency;
}

void N73GPUCore::setMidGain(float gainDb)
{
    midGain = gainDb;
}

void N73GPUCore::setLowShelfFrequency(int freqIndex)
{
    lowShelfFreqIndex = juce::jlimit(0, 3, freqIndex);
}

void N73GPUCore::setLowShelfGain(float gainDb)
{
    lowShelfGain = gainDb;
}

void N73GPUCore::setHPFFrequency(int freqIndex)
{
    hpfFreqIndex = juce::jlimit(0, 4, freqIndex);
}

void N73GPUCore::setOutputLevel(float levelDb)
{
    outputLevel = levelDb;
}

//==============================================================================
// GPU Performance and diagnostics (Milestone 3)

void N73GPUCore::enableGPUProcessing(bool enable)
{
    if (enable && gpuAvailable)
    {
        usingGPU = true;
        std::cout << "GPU processing enabled" << std::endl;
    }
    else if (!enable)
    {
        usingGPU = false;
        std::cout << "GPU processing disabled, using CPU" << std::endl;
    }
    else if (enable && !gpuAvailable)
    {
        std::cout << "Warning: GPU processing requested but GPU not available" << std::endl;
        if (gpuFallbackEnabled)
        {
            usingGPU = false;
            std::cout << "Falling back to CPU processing" << std::endl;
        }
    }
}

void N73GPUCore::setGPUFallbackMode(bool enableFallback)
{
    gpuFallbackEnabled = enableFallback;
    std::cout << "GPU fallback mode " << (enableFallback ? "enabled" : "disabled") << std::endl;
}

bool N73GPUCore::runGPUDiagnostics()
{
    std::cout << "\n=== N73GPU GPU Diagnostics ===" << std::endl;
    printGPUStatus();
    
    if (gpuAvailable)
    {
        // Run GPU performance test
        std::cout << "Running GPU performance test..." << std::endl;
        
        // Create test buffer
        juce::AudioBuffer<float> testBuffer(2, bufferSize);
        testBuffer.clear();
        
        // Fill with test data
        for (int channel = 0; channel < testBuffer.getNumChannels(); ++channel)
        {
            auto* channelData = testBuffer.getWritePointer(channel);
            for (int sample = 0; sample < bufferSize; ++sample)
            {
                channelData[sample] = 0.5f * std::sin(2.0f * M_PI * 440.0f * sample / sampleRate);
            }
        }
        
        // Test GPU processing
        auto startTime = std::chrono::high_resolution_clock::now();
        processBlockGPU(testBuffer);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        std::cout << "GPU test processing time: " << processingTime << " ms" << std::endl;
        
        return true;
    }
    else
    {
        std::cout << "GPU not available for diagnostics" << std::endl;
        return false;
    }
}

void N73GPUCore::printGPUStatus() const
{
    std::cout << "GPU Status:" << std::endl;
    std::cout << "  Available: " << (gpuAvailable ? "Yes" : "No") << std::endl;
    std::cout << "  Currently Using: " << (usingGPU ? "GPU" : "CPU") << std::endl;
    std::cout << "  Fallback Enabled: " << (gpuFallbackEnabled ? "Yes" : "No") << std::endl;
    std::cout << "  Sample Rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "  Buffer Size: " << bufferSize << " samples" << std::endl;
    std::cout << "  CPU Load: " << (cpuLoad * 100.0f) << "%" << std::endl;
    std::cout << "  Last Processing Time: " << lastProcessingTimeMs << " ms" << std::endl;
}

//==============================================================================
// Private methods

void N73GPUCore::initializeGPU()
{
    // Milestone 1: GPU not available yet
    // Milestone 2: Will implement GPU Audio SDK initialization
    
#if GPU_AUDIO_AVAILABLE
    // Future GPU initialization code will go here
    gpuAvailable = false; // Will be true when GPU Audio SDK is integrated
    usingGPU = false;
#else
    gpuAvailable = false;
    usingGPU = false;
#endif
}

void N73GPUCore::processBlockGPU(juce::AudioBuffer<float>& buffer)
{
    // Milestone 3: Actual CUDA GPU processing
    if (!cudaProcessor || !cudaProcessor->isGPUAvailable())
    {
        // Fallback to CPU processing if GPU not available
        processBlockCPU(buffer);
        return;
    }
    
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    
    if (numChannels >= 2)
    {
        // Stereo processing using CUDA
        float* leftChannel = buffer.getWritePointer(0);
        float* rightChannel = buffer.getWritePointer(1);
        
        float inputGainLinear = juce::Decibels::decibelsToGain(inputGain);
        float outputLevelLinear = juce::Decibels::decibelsToGain(outputLevel);
        
        bool success = cudaProcessor->processBuffer(leftChannel, rightChannel, numSamples, 
                                                   inputGainLinear, outputLevelLinear);
        
        if (!success)
        {
            std::cout << "[N73GPUCore] GPU processing failed, falling back to CPU" << std::endl;
            processBlockCPU(buffer);
            return;
        }
        
        // Copy left channel processing to additional channels if present
        for (int channel = 2; channel < numChannels; ++channel)
        {
            buffer.copyFrom(channel, 0, buffer, 0, 0, numSamples);
        }
    }
    else if (numChannels == 1)
    {
        // Mono processing using CUDA gain processor
        float* channelData = buffer.getWritePointer(0);
        float totalGain = juce::Decibels::decibelsToGain(inputGain + outputLevel);
        
        bool success = cudaProcessor->processGain(channelData, numSamples, totalGain);
        
        if (!success)
        {
            std::cout << "[N73GPUCore] GPU gain processing failed, falling back to CPU" << std::endl;
            processBlockCPU(buffer);
        }
    }
}

void N73GPUCore::processBlockCPU(juce::AudioBuffer<float>& buffer)
{
    // Milestone 1: Simple passthrough processing
    // Milestone 4: Full Neve 1073 CPU implementation
    
    auto numChannels = buffer.getNumChannels();
    auto numSamples = buffer.getNumSamples();
    
    // For Milestone 1: Apply basic input/output gain
    float inputGainLinear = juce::Decibels::decibelsToGain(inputGain);
    float outputGainLinear = juce::Decibels::decibelsToGain(outputLevel);
    
    for (int channel = 0; channel < numChannels; ++channel)
    {
        auto* channelData = buffer.getWritePointer(channel);
        
        for (int sample = 0; sample < numSamples; ++sample)
        {
            // Simple gain processing for Milestone 1
            channelData[sample] *= inputGainLinear;
            // Future: Full 1073 processing chain will go here
            channelData[sample] *= outputGainLinear;
        }
    }
    
    // Update CPU load estimate (very basic for now)
    cpuLoad = 0.05f; // 5% placeholder
}

void N73GPUCore::handleControlChange(int controller, int value)
{
    // MIDI CC mapping for hardware integration (as specified in the plan)
    switch (controller)
    {
        case 20: // INPUT_GAIN
            setInputGain(juce::jmap(float(value), 0.0f, 127.0f, -20.0f, 20.0f));
            break;
        case 21: // HIGH_SHELF_GAIN
            setHighShelfGain(juce::jmap(float(value), 0.0f, 127.0f, -16.0f, 16.0f));
            break;
        case 22: // MID_FREQUENCY
            setMidFrequency(juce::jmap(float(value), 0.0f, 127.0f, 360.0f, 7200.0f));
            break;
        case 23: // MID_GAIN
            setMidGain(juce::jmap(float(value), 0.0f, 127.0f, -18.0f, 18.0f));
            break;
        case 24: // MID_Q (placeholder for future implementation)
            break;
        case 25: // LOW_SHELF_GAIN
            setLowShelfGain(juce::jmap(float(value), 0.0f, 127.0f, -16.0f, 16.0f));
            break;
        case 26: // HPF_FREQUENCY
            setHPFFrequency(juce::jmap(value, 0, 127, 0, 4));
            break;
        case 27: // OUTPUT_LEVEL
            setOutputLevel(juce::jmap(float(value), 0.0f, 127.0f, -20.0f, 10.0f));
            break;
        case 28: // EQ_BYPASS (placeholder)
        case 29: // HPF_BYPASS (placeholder)
            break;
    }
}

void N73GPUCore::handleSysEx(const juce::MidiMessage& message)
{
    // SysEx handling for bidirectional hardware communication
    // Will be implemented in Phase 3: Hardware Integration
    
    auto sysExData = message.getSysExData();
    auto sysExSize = message.getSysExDataSize();
    
    if (sysExSize >= 3 && sysExData[0] == 0x7D) // KZRACKS manufacturer ID
    {
        switch (sysExData[1])
        {
            case 0x30: // VU Meter Feedback (future implementation)
                break;
            case 0x31: // LED Status (future implementation)
                break;
            case 0x32: // Device ID (future implementation)
                break;
        }
    }
}