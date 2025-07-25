#include "GPUPreAmp.h"
#include <iostream>
#include <string>

void printUsage(const char* programName)
{
    std::cout << "N73GPU GPUPreAmp - Standalone GPU Processing Test" << std::endl;
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  -i, --input <file>      Input WAV file (or generate test tone)" << std::endl;
    std::cout << "  -o, --output <file>     Output file prefix" << std::endl;
    std::cout << "  -g, --gain <db>         Gain in dB (default: 0)" << std::endl;
    std::cout << "  -s, --samplerate <hz>   Sample rate (default: 44100)" << std::endl;
    std::cout << "  -b, --buffersize <n>    Buffer size (default: 512)" << std::endl;
    std::cout << "  --benchmark             Run performance benchmark" << std::endl;
    std::cout << "  --info                  Show system information" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " --benchmark" << std::endl;
    std::cout << "  " << programName << " -g 6 -o processed_audio" << std::endl;
    std::cout << "  " << programName << " --info" << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "=== N73GPU GPUPreAmp Test Application ===" << std::endl;
    std::cout << "Milestone 2: Standalone GPU Prototype" << std::endl;
    std::cout << "KZRACKS - Neve 1073 GPU Emulation" << std::endl;
    std::cout << std::endl;

    // Default parameters
    std::string inputFile = "";
    std::string outputFile = "gpupreamp_output";
    float gainDB = 0.0f;
    int sampleRate = 44100;
    int bufferSize = 512;
    bool runBenchmark = false;
    bool showInfo = false;
    bool showHelp = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help")
        {
            showHelp = true;
        }
        else if (arg == "-i" || arg == "--input")
        {
            if (i + 1 < argc)
                inputFile = argv[++i];
        }
        else if (arg == "-o" || arg == "--output")
        {
            if (i + 1 < argc)
                outputFile = argv[++i];
        }
        else if (arg == "-g" || arg == "--gain")
        {
            if (i + 1 < argc)
                gainDB = std::stof(argv[++i]);
        }
        else if (arg == "-s" || arg == "--samplerate")
        {
            if (i + 1 < argc)
                sampleRate = std::stoi(argv[++i]);
        }
        else if (arg == "-b" || arg == "--buffersize")
        {
            if (i + 1 < argc)
                bufferSize = std::stoi(argv[++i]);
        }
        else if (arg == "--benchmark")
        {
            runBenchmark = true;
        }
        else if (arg == "--info")
        {
            showInfo = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (showHelp)
    {
        printUsage(argv[0]);
        return 0;
    }

    // Create and initialize GPUPreAmp
    GPUPreAmp preamp;
    if (!preamp.init(sampleRate, bufferSize))
    {
        std::cerr << "Error: Failed to initialize GPUPreAmp" << std::endl;
        return 1;
    }

    // Set gain parameter
    preamp.setGain(gainDB);

    try
    {
        if (showInfo)
        {
            preamp.printSystemInfo();
        }
        else if (runBenchmark)
        {
            preamp.runBenchmark();
        }
        else
        {
            // Process audio file (or generate test tone)
            std::cout << "Processing audio..." << std::endl;
            
            std::string input = inputFile.empty() ? "test_tone" : inputFile;
            if (!preamp.processFile(input, outputFile))
            {
                std::cerr << "Error: Processing failed" << std::endl;
                return 1;
            }
            
            std::cout << "Processing completed successfully!" << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Show final status
    std::cout << std::endl;
    std::cout << "=== Final Status ===" << std::endl;
    std::cout << "GPU Available: " << (preamp.isGPUAvailable() ? "Yes" : "No") << std::endl;
    std::cout << "Used Processing: " << (preamp.isUsingGPU() ? "GPU" : "CPU") << std::endl;
    std::cout << "CPU Load: " << (preamp.getCurrentCPULoad() * 100.0f) << "%" << std::endl;
    std::cout << "Last Processing Time: " << preamp.getLastProcessingTimeMs() << " ms" << std::endl;

    std::cout << std::endl;
    std::cout << "GPUPreAmp test completed!" << std::endl;
    return 0;
}