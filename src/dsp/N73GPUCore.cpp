#include "N73GPUCore.h"
#include <iostream>
#include <algorithm>

//==============================================================================
// Constructor and Destructor

//==============================================================================
N73GPUCore::N73GPUCore()
{
    // Create Neve 1073 CPU processor (Milestone 4)
    neve1073Processor = std::make_unique<Neve1073DSP>();
    std::cout << "N73GPUCore: Neve 1073 CPU processor created" << std::endl;
    
    // Create CUDA processor when GPU support is available
#if GPU_AUDIO_AVAILABLE
    cudaProcessor = std::make_unique<CudaProcessor>();
    std::cout << "N73GPUCore: CUDA processor created" << std::endl;
#else
    std::cout << "N73GPUCore: CUDA support not available, using CPU-only processing" << std::endl;
#endif
    
    std::cout << "N73GPUCore: Construction complete" << std::endl;
}

N73GPUCore::~N73GPUCore()
{
    // Cleanup GPU resources
    if (cudaProcessor)
    {
#if GPU_AUDIO_AVAILABLE
        cudaProcessor->shutdown();
        std::cout << "N73GPUCore: CUDA processor shutdown complete" << std::endl;
#endif
    }
    releaseResources();
}

//==============================================================================
void N73GPUCore::initialize()
{
    // Initialize Neve 1073 CPU processor (Milestone 4)
    if (neve1073Processor)
    {
        juce::dsp::ProcessSpec spec;
        spec.sampleRate = sampleRate;
        spec.maximumBlockSize = static_cast<juce::uint32>(bufferSize);
        spec.numChannels = 2; // Stereo processing
        
        neve1073Processor->prepare(spec);
        std::cout << "N73GPUCore: Neve 1073 CPU processor initialized" << std::endl;
    }
    
    // Initialize CUDA GPU processing
    if (cudaProcessor)
    {
#if GPU_AUDIO_AVAILABLE
        gpuAvailable = cudaProcessor->initialize();
        if (gpuAvailable)
        {
            std::cout << "N73GPUCore: CUDA GPU initialized and available" << std::endl;
            usingGPU = true; // Enable GPU by default when available
        }
        else
        {
            std::cout << "N73GPUCore: CUDA GPU not available, using CPU fallback" << std::endl;
        }
#endif
    }
    
    std::cout << "N73GPUCore initialized with sample rate: " << sampleRate 
              << "Hz, buffer size: " << bufferSize << " samples" << std::endl;
    
    isInitialized = true;
}

void N73GPUCore::prepareToPlay(double sampleRate, int bufferSize)
{
    this->sampleRate = sampleRate;
    this->bufferSize = bufferSize;
    
    std::cout << "N73GPUCore: Preparing to play at " << sampleRate << "Hz with " 
              << bufferSize << " samples per buffer" << std::endl;
    
    // Reset processing state
    cpuLoad = 0.0f;
    
    // Prepare Neve 1073 processor with new specs
    if (neve1073Processor)
    {
        juce::dsp::ProcessSpec spec;
        spec.sampleRate = sampleRate;
        spec.maximumBlockSize = static_cast<juce::uint32>(bufferSize);
        spec.numChannels = 2;
        
        neve1073Processor->prepare(spec);
        std::cout << "N73GPUCore: Neve 1073 processor prepared for " 
                  << sampleRate << "Hz, " << bufferSize << " samples" << std::endl;
    }
    
    // Initialize GPU processing if available and not already done
    if (!gpuAvailable)
    {
        initializeGPU();
    }
    
    isInitialized = true;
}

void N73GPUCore::releaseResources()
{
    isInitialized = false;
    
    // Reset Neve 1073 processor
    if (neve1073Processor)
    {
        neve1073Processor->reset();
    }
}

//==============================================================================
void N73GPUCore::processBlock(juce::AudioBuffer<float>& buffer)
{
    if (!isInitialized)
        return;
    
    // Milestone 4: Enhanced processing with complete Neve 1073 model
    processingStartTime = std::chrono::high_resolution_clock::now();
    
    if (usingGPU && gpuAvailable)
    {
        processBlockGPU(buffer);
    }
    else
    {
        processBlockCPU(buffer);
    }
    
    processingEndTime = std::chrono::high_resolution_clock::now();
    lastProcessingTimeMs = std::chrono::duration<double, std::milli>(processingEndTime - processingStartTime).count();
}

void N73GPUCore::processMidiMessage(const juce::MidiMessage& message)
{
    if (message.isControllerOfType(0x07)) // Volume CC
    {
        auto value = message.getControllerValue();
        setOutputLevel(juce::jmap(float(value), 0.0f, 127.0f, -20.0f, 10.0f));
    }
    else if (message.isControllerOfType(0x01)) // Modulation wheel for input gain
    {
        auto value = message.getControllerValue();
        setInputGain(juce::jmap(float(value), 0.0f, 127.0f, -20.0f, 80.0f));
    }
    
    // Handle other MIDI messages
    if (message.isControllerOfType(juce::MidiMessage::controllerEvent))
    {
        handleControlChange(message.getControllerNumber(), message.getControllerValue());
    }
    else if (message.isSysEx())
    {
        handleSysEx(message);
    }
}

//==============================================================================
// Neve 1073 Parameter Control (Milestone 4)

void N73GPUCore::setInputGain(float gainDb)
{
    inputGain = gainDb;
    if (neve1073Processor)
        neve1073Processor->setInputGain(gainDb);
}

void N73GPUCore::setHighShelfGain(float gainDb)
{
    highShelfGain = gainDb;
    if (neve1073Processor)
        neve1073Processor->setHighShelfGain(gainDb);
}

void N73GPUCore::setMidFrequency(float frequency)
{
    midFrequency = frequency;
    if (neve1073Processor)
        neve1073Processor->setMidFrequency(frequency);
}

void N73GPUCore::setMidGain(float gainDb)
{
    midGain = gainDb;
    if (neve1073Processor)
        neve1073Processor->setMidGain(gainDb);
}

void N73GPUCore::setMidQ(float q)
{
    if (neve1073Processor)
        neve1073Processor->setMidQ(q);
}

void N73GPUCore::setLowShelfFrequency(int index)
{
    lowShelfFreqIndex = index;
    if (neve1073Processor)
        neve1073Processor->setLowShelfFrequency(index);
}

void N73GPUCore::setLowShelfGain(float gainDb)
{
    lowShelfGain = gainDb;
    if (neve1073Processor)
        neve1073Processor->setLowShelfGain(gainDb);
}

void N73GPUCore::setHPFFrequency(int freqIndex)
{
    hpfFreqIndex = freqIndex;
    if (neve1073Processor)
        neve1073Processor->setHighPassFrequency(freqIndex);
}

void N73GPUCore::setOutputLevel(float levelDb)
{
    outputLevel = levelDb;
    if (neve1073Processor)
        neve1073Processor->setOutputLevel(levelDb);
}

//==============================================================================
// EQ Bypass Controls

void N73GPUCore::setHighShelfBypass(bool bypass)
{
    if (neve1073Processor)
        neve1073Processor->setHighShelfBypass(bypass);
}

void N73GPUCore::setMidEqBypass(bool bypass)
{
    if (neve1073Processor)
        neve1073Processor->setMidEqBypass(bypass);
}

void N73GPUCore::setLowShelfBypass(bool bypass)
{
    if (neve1073Processor)
        neve1073Processor->setLowShelfBypass(bypass);
}

void N73GPUCore::setHighPassBypass(bool bypass)
{
    if (neve1073Processor)
        neve1073Processor->setHighPassBypass(bypass);
}

//==============================================================================
// Advanced Controls (Milestone 4)

void N73GPUCore::setSaturationAmount(float amount)
{
    if (neve1073Processor)
        neve1073Processor->setSaturationAmount(amount);
}

void N73GPUCore::setTransformerModeling(bool enable)
{
    if (neve1073Processor)
        neve1073Processor->setTransformerModeling(enable);
}

void N73GPUCore::setHarmonicContent(float amount)
{
    if (neve1073Processor)
        neve1073Processor->setHarmonicContent(amount);
}

//==============================================================================
// GPU Performance and diagnostics

void N73GPUCore::enableGPUProcessing(bool enable)
{
    usingGPU = enable && gpuAvailable;
    std::cout << "N73GPUCore: GPU processing " << (usingGPU ? "enabled" : "disabled") << std::endl;
}

void N73GPUCore::setGPUFallbackMode(bool enableFallback)
{
    gpuFallbackEnabled = enableFallback;
    std::cout << "N73GPUCore: GPU fallback mode " << (enableFallback ? "enabled" : "disabled") << std::endl;
}

bool N73GPUCore::runGPUDiagnostics()
{
    std::cout << "N73GPUCore: Running GPU diagnostics..." << std::endl;
    
#if GPU_AUDIO_AVAILABLE
    if (cudaProcessor)
    {
        bool result = cudaProcessor->runDiagnostics();
        std::cout << "N73GPUCore: GPU diagnostics " << (result ? "passed" : "failed") << std::endl;
        return result;
    }
#endif
    
    std::cout << "N73GPUCore: GPU diagnostics not available (CUDA not compiled)" << std::endl;
    return false;
}

void N73GPUCore::printGPUStatus() const
{
    std::cout << "=== N73GPUCore GPU Status ===" << std::endl;
    std::cout << "GPU Available: " << (gpuAvailable ? "YES" : "NO") << std::endl;
    std::cout << "GPU In Use: " << (usingGPU ? "YES" : "NO") << std::endl;
    std::cout << "GPU Fallback Enabled: " << (gpuFallbackEnabled ? "YES" : "NO") << std::endl;
    std::cout << "Last Processing Time: " << lastProcessingTimeMs << "ms" << std::endl;
    std::cout << "CPU Load: " << (cpuLoad * 100.0f) << "%" << std::endl;
    
#if GPU_AUDIO_AVAILABLE
    if (cudaProcessor)
    {
        cudaProcessor->printStatus();
    }
    else
    {
        std::cout << "CUDA Processor: Not initialized" << std::endl;
    }
#else
    std::cout << "CUDA Support: Not compiled" << std::endl;
#endif
    
    std::cout << "============================" << std::endl;
}

//==============================================================================
// Neve 1073 Analysis (Milestone 4)

float N73GPUCore::getCurrentSaturation() const
{
    if (neve1073Processor)
        return neve1073Processor->getCurrentSaturation();
    return 0.0f;
}

float N73GPUCore::getTotalHarmonicDistortion() const
{
    if (neve1073Processor)
        return neve1073Processor->getTotalHarmonicDistortion();
    return 0.0f;
}

std::array<float, 5> N73GPUCore::getEQResponse(float frequency) const
{
    if (neve1073Processor)
        return neve1073Processor->getEQResponse(frequency);
    return {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
}

//==============================================================================
// Private methods

void N73GPUCore::initializeGPU()
{
    // GPU initialization handled in initialize() method
}

void N73GPUCore::processBlockGPU(juce::AudioBuffer<float>& buffer)
{
    // Milestone 3: Actual CUDA GPU processing
    if (!cudaProcessor)
    {
        // Fallback to CPU processing if GPU not available
        processBlockCPU(buffer);
        return;
    }

#if GPU_AUDIO_AVAILABLE
    if (!cudaProcessor->isGPUAvailable())
    {
        // Fallback to CPU processing if GPU not available
        processBlockCPU(buffer);
        return;
    }
    
    // Attempt GPU processing
    try
    {
        auto numChannels = buffer.getNumChannels();
        auto numSamples = buffer.getNumSamples();
        
        // Simple GPU gain processing for now (Milestone 3)
        // Future: Full Neve 1073 GPU implementation will go here (Milestone 5)
        
        for (int channel = 0; channel < numChannels; ++channel)
        {
            auto* channelData = buffer.getWritePointer(channel);
            
            // Process this channel on GPU
            bool success = cudaProcessor->processAudioChannel(
                channelData, 
                numSamples, 
                juce::Decibels::decibelsToGain(inputGain)
            );
            
            if (!success)
            {
                std::cout << "N73GPUCore: GPU processing failed for channel " << channel 
                         << ", using CPU fallback" << std::endl;
                
                if (gpuFallbackEnabled)
                {
                    processBlockCPU(buffer);
                    return;
                }
            }
        }
        
        // Update CPU load estimate for GPU processing
        cpuLoad = 0.02f; // 2% CPU when using GPU
    }
    catch (const std::exception& e)
    {
        std::cout << "N73GPUCore: GPU processing exception: " << e.what() << std::endl;
        
        if (gpuFallbackEnabled)
        {
            processBlockCPU(buffer);
        }
    }
#else
    // No CUDA support, fallback to CPU
    processBlockCPU(buffer);
#endif
}

void N73GPUCore::processBlockCPU(juce::AudioBuffer<float>& buffer)
{
    // Milestone 4: Use complete Neve 1073 CPU implementation
    if (neve1073Processor)
    {
        // Process through authentic Neve 1073 model
        neve1073Processor->processBlock(buffer);
        
        // Update CPU load estimate (more accurate for complex processing)
        cpuLoad = 0.15f; // 15% estimate for full 1073 processing
    }
    else
    {
        // Fallback: Simple gain processing (legacy)
        auto numChannels = buffer.getNumChannels();
        auto numSamples = buffer.getNumSamples();
        
        float inputGainLinear = juce::Decibels::decibelsToGain(inputGain);
        float outputGainLinear = juce::Decibels::decibelsToGain(outputLevel);
        
        for (int channel = 0; channel < numChannels; ++channel)
        {
            auto* channelData = buffer.getWritePointer(channel);
            
            for (int sample = 0; sample < numSamples; ++sample)
            {
                channelData[sample] *= inputGainLinear;
                channelData[sample] *= outputGainLinear;
            }
        }
        
        cpuLoad = 0.05f; // 5% for simple processing
    }
}

void N73GPUCore::handleControlChange(int controller, int value)
{
    // MIDI CC mapping for Neve 1073 parameters (Milestone 4)
    switch (controller)
    {
        case 20: // INPUT_GAIN
            setInputGain(juce::jmap(float(value), 0.0f, 127.0f, -20.0f, 80.0f));
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
        case 24: // MID_Q
            setMidQ(juce::jmap(float(value), 0.0f, 127.0f, 0.3f, 4.0f));
            break;
        case 25: // LOW_SHELF_GAIN
            setLowShelfGain(juce::jmap(float(value), 0.0f, 127.0f, -16.0f, 16.0f));
            break;
        case 26: // LOW_SHELF_FREQUENCY (0-3 mapped to 0-127)
            setLowShelfFrequency(juce::jmap(value, 0, 127, 0, 3));
            break;
        case 27: // HPF_FREQUENCY (0-4 mapped to 0-127)
            setHPFFrequency(juce::jmap(value, 0, 127, 0, 4));
            break;
        case 28: // OUTPUT_LEVEL
            setOutputLevel(juce::jmap(float(value), 0.0f, 127.0f, -20.0f, 10.0f));
            break;
        case 29: // SATURATION_AMOUNT
            setSaturationAmount(juce::jmap(float(value), 0.0f, 127.0f, 0.0f, 1.0f));
            break;
        case 30: // HARMONIC_CONTENT
            setHarmonicContent(juce::jmap(float(value), 0.0f, 127.0f, 0.0f, 1.0f));
            break;
        
        // GPU control (Milestone 3)
        case 100: // GPU_ENABLE
            enableGPUProcessing(value >= 64); // On/off at mid-point
            break;
        case 101: // GPU_FALLBACK
            setGPUFallbackMode(value >= 64);
            break;
            
        default:
            // Unknown controller
            break;
    }
}

void N73GPUCore::handleSysEx(const juce::MidiMessage& message)
{
    // Future: Handle SysEx messages for bulk parameter updates
    // This will be important for hardware controller integration
    auto data = message.getSysExData();
    auto dataSize = message.getSysExDataSize();
    
    if (dataSize >= 3 && data[0] == 0x7D) // Custom manufacturer ID for KZRACKS
    {
        // Handle KZRACKS-specific SysEx commands
        switch (data[1])
        {
            case 0x01: // Bulk parameter dump
                // Future implementation
                break;
            case 0x02: // Preset recall
                // Future implementation
                break;
            default:
                break;
        }
    }
}