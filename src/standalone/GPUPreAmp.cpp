#include "GPUPreAmp.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <functional>

//==============================================================================
GPUPreAmp::GPUPreAmp()
{
    updateGainLinear();
}

GPUPreAmp::~GPUPreAmp()
{
    shutdown();
}

//==============================================================================
bool GPUPreAmp::init(int newSampleRate, int newBufferSize)
{
    sampleRate = newSampleRate;
    bufferSize = newBufferSize;
    
    std::cout << "Initializing GPUPreAmp..." << std::endl;
    std::cout << "Sample Rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "Buffer Size: " << bufferSize << " samples" << std::endl;
    
    // Try to initialize GPU processing
    if (initializeGPU())
    {
        std::cout << "GPU processing initialized successfully!" << std::endl;
        usingGPU = true;
    }
    else
    {
        std::cout << "GPU not available, using CPU processing" << std::endl;
        usingGPU = false;
    }
    
    isInitialized = true;
    return true;
}

void GPUPreAmp::shutdown()
{
    if (!isInitialized)
        return;
        
    std::cout << "Shutting down GPUPreAmp..." << std::endl;
    
    if (usingGPU)
    {
        shutdownGPU();
    }
    
    isInitialized = false;
}

//==============================================================================
bool GPUPreAmp::processFile(const std::string& inputPath, const std::string& outputPath)
{
    std::cout << "Processing file: " << inputPath << std::endl;
    
    // Load input WAV file
    std::vector<float> audioData;
    int channels = 0;
    
    if (!loadWAV(inputPath, audioData, channels))
    {
        std::cerr << "Error: Could not load input file: " << inputPath << std::endl;
        return false;
    }
    
    std::cout << "Loaded " << audioData.size() << " samples, " << channels << " channels" << std::endl;
    
    // Process audio in chunks
    std::vector<float> outputData = audioData; // Copy input to output
    int totalSamples = static_cast<int>(audioData.size());
    
    auto processStart = std::chrono::high_resolution_clock::now();
    
    // Process in buffer-sized chunks
    for (int offset = 0; offset < totalSamples; offset += bufferSize)
    {
        int samplesToProcess = std::min(bufferSize, totalSamples - offset);
        
        // Process this chunk
        process(audioData.data() + offset, outputData.data() + offset, samplesToProcess);
    }
    
    auto processEnd = std::chrono::high_resolution_clock::now();
    auto processingTime = std::chrono::duration<double, std::milli>(processEnd - processStart).count();
    
    std::cout << "Processing completed in " << processingTime << " ms" << std::endl;
    std::cout << "Real-time factor: " << (totalSamples / double(sampleRate)) / (processingTime / 1000.0) << "x" << std::endl;
    
    // Save output WAV file
    if (!saveWAV(outputPath, outputData, channels, sampleRate))
    {
        std::cerr << "Error: Could not save output file: " << outputPath << std::endl;
        return false;
    }
    
    std::cout << "Output saved to: " << outputPath << std::endl;
    return true;
}

void GPUPreAmp::process(const float* input, float* output, int numSamples)
{
    if (!isInitialized)
        return;
        
    if (usingGPU && gpuAvailable)
    {
        processBlockGPU(input, output, numSamples);
    }
    else
    {
        processBlockCPU(input, output, numSamples);
    }
}

//==============================================================================
void GPUPreAmp::setGain(float gainDB)
{
    this->gainDB = gainDB;
    updateGainLinear();
    
    std::cout << "Gain set to " << gainDB << " dB (" << gainLinear << " linear)" << std::endl;
}

//==============================================================================
void GPUPreAmp::runBenchmark()
{
    std::cout << "\n=== GPU PreAmp Benchmark ===" << std::endl;
    printSystemInfo();
    
    // Generate test audio
    const int testSamples = sampleRate * 5; // 5 seconds of audio
    std::vector<float> testInput(testSamples);
    std::vector<float> testOutput(testSamples);
    
    // Generate sine wave test signal
    for (int i = 0; i < testSamples; ++i)
    {
        testInput[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / sampleRate);
    }
    
    std::cout << "\nBenchmarking " << testSamples << " samples (" << testSamples / double(sampleRate) << " seconds)..." << std::endl;
    
    // Benchmark CPU processing
    usingGPU = false;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    
    for (int offset = 0; offset < testSamples; offset += bufferSize)
    {
        int samplesToProcess = std::min(bufferSize, testSamples - offset);
        processBlockCPU(testInput.data() + offset, testOutput.data() + offset, samplesToProcess);
    }
    
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    
    std::cout << "CPU Processing: " << cpuTime << " ms" << std::endl;
    std::cout << "CPU Real-time factor: " << (testSamples / double(sampleRate)) / (cpuTime / 1000.0) << "x" << std::endl;
    
    // Benchmark GPU processing (if available)
    if (gpuAvailable)
    {
        usingGPU = true;
        auto gpuStart = std::chrono::high_resolution_clock::now();
        
        for (int offset = 0; offset < testSamples; offset += bufferSize)
        {
            int samplesToProcess = std::min(bufferSize, testSamples - offset);
            processBlockGPU(testInput.data() + offset, testOutput.data() + offset, samplesToProcess);
        }
        
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        auto gpuTime = std::chrono::duration<double, std::milli>(gpuEnd - gpuStart).count();
        
        std::cout << "GPU Processing: " << gpuTime << " ms" << std::endl;
        std::cout << "GPU Real-time factor: " << (testSamples / double(sampleRate)) / (gpuTime / 1000.0) << "x" << std::endl;
        std::cout << "GPU Speedup: " << cpuTime / gpuTime << "x" << std::endl;
    }
    else
    {
        std::cout << "GPU not available for benchmarking" << std::endl;
    }
}

void GPUPreAmp::printSystemInfo() const
{
    std::cout << "System Information:" << std::endl;
    std::cout << "  Sample Rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "  Buffer Size: " << bufferSize << " samples" << std::endl;
    std::cout << "  GPU Available: " << (gpuAvailable ? "Yes" : "No") << std::endl;
    std::cout << "  Currently Using: " << (usingGPU ? "GPU" : "CPU") << std::endl;
    std::cout << "  Current Gain: " << gainDB << " dB" << std::endl;
}

//==============================================================================
// Private methods

bool GPUPreAmp::initializeGPU()
{
    // Milestone 2: GPU not available yet
    // This will be implemented when GPU Audio SDK is integrated
    
#if GPU_AUDIO_AVAILABLE
    // Future GPU Audio SDK initialization code
    /*
    try {
        gpuBackend = new GPUDSP::Backend();
        if (gpuBackend->initialize()) {
            gainProcessor = gpuBackend->createGainProcessor();
            gpuAvailable = true;
            return true;
        }
    } catch (...) {
        // GPU initialization failed
    }
    */
#endif
    
    gpuAvailable = false;
    return false;
}

void GPUPreAmp::processBlockGPU(const float* input, float* output, int numSamples)
{
    // Milestone 2: GPU processing placeholder
    // For now, fall back to CPU processing
    processBlockCPU(input, output, numSamples);
}

void GPUPreAmp::shutdownGPU()
{
#if GPU_AUDIO_AVAILABLE
    // Future GPU cleanup code
    /*
    if (gainProcessor) {
        delete gainProcessor;
        gainProcessor = nullptr;
    }
    if (gpuBackend) {
        gpuBackend->shutdown();
        delete gpuBackend;
        gpuBackend = nullptr;
    }
    */
#endif
}

void GPUPreAmp::processBlockCPU(const float* input, float* output, int numSamples)
{
    // Simple gain processing on CPU
    for (int i = 0; i < numSamples; ++i)
    {
        output[i] = input[i] * gainLinear;
    }
    
    // Update CPU load estimate
    cpuLoad = 0.02f; // 2% placeholder for simple gain processing
}

//==============================================================================
// WAV file I/O (simplified for testing)

bool GPUPreAmp::loadWAV(const std::string& filename, std::vector<float>& audioData, int& channels)
{
    // For Milestone 2, we'll create a simple test tone instead of loading WAV files
    // This avoids dependencies on external WAV libraries
    
    std::cout << "Generating test tone (simulating WAV load)..." << std::endl;
    
    channels = 1; // Mono
    int testDuration = 2; // 2 seconds
    int totalSamples = sampleRate * testDuration;
    
    audioData.resize(totalSamples);
    
    // Generate 440Hz sine wave
    for (int i = 0; i < totalSamples; ++i)
    {
        audioData[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / sampleRate);
    }
    
    return true;
}

bool GPUPreAmp::saveWAV(const std::string& filename, const std::vector<float>& audioData, 
                        int channels, int sampleRate)
{
    // For Milestone 2, we'll just write a simple text file with sample values
    // This allows us to verify processing without WAV library dependencies
    
    std::ofstream file(filename + ".txt");
    if (!file.is_open())
        return false;
        
    file << "# N73GPU GPUPreAmp Test Output" << std::endl;
    file << "# Sample Rate: " << sampleRate << std::endl;
    file << "# Channels: " << channels << std::endl;
    file << "# Samples: " << audioData.size() << std::endl;
    file << "# Gain Applied: " << gainDB << " dB" << std::endl;
    file << "#" << std::endl;
    
    // Write first 100 samples for verification
    int samplesToWrite = std::min(100, static_cast<int>(audioData.size()));
    for (int i = 0; i < samplesToWrite; ++i)
    {
        file << i << "\t" << audioData[i] << std::endl;
    }
    
    file.close();
    std::cout << "Test data written to: " << filename << ".txt" << std::endl;
    return true;
}

//==============================================================================
// Utilities

void GPUPreAmp::updateGainLinear()
{
    gainLinear = std::pow(10.0f, gainDB / 20.0f);
}

double GPUPreAmp::measureProcessingTime(std::function<void()> processFunc) const
{
    auto start = std::chrono::high_resolution_clock::now();
    processFunc();
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}