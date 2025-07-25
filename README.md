# N73GPU - Neve 1073 GPU-Accelerated Emulation Plugin

A high-performance Neve 1073 preamp emulation VST3 plugin with CUDA GPU acceleration for real-time audio processing.

## 🚀 Features

- **Authentic Neve 1073 Emulation**: Classic British console preamp character
- **CUDA GPU Acceleration**: Real-time GPU processing for enhanced performance
- **CPU Fallback Mode**: Automatic fallback for systems without CUDA support
- **VST3 Plugin**: Professional DAW integration
- **Cross-Platform**: Windows (current), macOS support planned
- **MIDI Ready**: Architecture prepared for hardware controller integration

## 🏗️ Architecture

- **Framework**: JUCE 7.x
- **Build System**: CMake with Visual Studio 2022
- **GPU Processing**: Direct CUDA integration (CUDA Toolkit 12.9)
- **Target Platform**: Windows VST3, macOS AU (planned)
- **Language**: C++17
- **GPU Support**: NVIDIA GeForce RTX series (compute capability 5.2+)

## 🔧 Prerequisites

### Required Tools
- **Visual Studio 2022** (MSVC 19.44+)
- **CMake 3.20+**
- **CUDA Toolkit 12.9** (for GPU acceleration)
- **Git** (for submodule management)

### Hardware Requirements
- **GPU**: NVIDIA GeForce GTX 10xx series or newer
- **Driver**: NVIDIA Driver 576.88+ (for CUDA 12.9 support)
- **Memory**: 4GB+ GPU memory recommended

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone --recursive https://github.com/LouisBHouston/N73GPU.git
cd N73GPU
```

### 2. Configure Build
```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
```

### 3. Build Plugin
```bash
cmake --build . --config Release
```

### 4. Install Plugin
The VST3 plugin will be built to:
```
build/src/N73GPU_artefacts/Release/VST3/N73GPU.vst3
```

Copy to your VST3 plugins directory or use directly in your DAW.

## 🎛️ Development Status

### ✅ Completed Milestones

- **Milestone 1**: Skeleton Plugin Validation
  - ✅ JUCE VST3 integration
  - ✅ Basic audio passthrough
  - ✅ Parameter controls
  - ✅ DAW compatibility validated

- **Milestone 2**: Standalone GPU Prototype  
  - ✅ GPUPreAmp test harness
  - ✅ CUDA gain processing
  - ✅ WAV file benchmarking

- **Milestone 3**: Plugin GPU Integration
  - ✅ CUDA processor integration
  - ✅ GPU/CPU fallback architecture
  - ✅ CMake CUDA configuration
  - ✅ MSVC/CUDA compatibility fixes
  - ✅ Build system validation

### 🔄 In Progress

- **Milestone 4**: CPU Reference Implementation
- **Milestone 5**: GPU DSP Conversion  
- **Milestone 6**: Professional GUI & UX
- **Milestone 7**: Advanced DSP Features
- **Milestone 8**: Cross-Platform & Packaging

## 🔧 Build Configuration

### CUDA Support
The plugin automatically detects CUDA availability:
- **GPU Available**: Uses CUDA acceleration for real-time processing
- **GPU Unavailable**: Falls back to optimized CPU processing

### CMake Options
```cmake
option(BUILD_STANDALONE "Build standalone test app" ON)
option(BUILD_VST3 "Build VST3 plugin" ON)  
option(BUILD_AU "Build AU plugin" OFF)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Not Detected**
   - Verify NVIDIA driver version (576.88+)
   - Check CUDA Toolkit installation
   - Ensure GPU compute capability compatibility

2. **Build Errors**
   - Use Visual Studio 2022 (17.4+)
   - Verify CMake version (3.20+)
   - Check all prerequisites installed

3. **Plugin Not Loading**
   - Verify VST3 path in DAW settings
   - Check plugin architecture (x64)
   - Validate audio driver configuration

## 🤝 Contributing

This project follows a milestone-driven development approach:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** changes following project architecture
4. **Test** with both GPU and CPU modes
5. **Submit** pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **JUCE Framework**: Cross-platform audio application framework
- **NVIDIA CUDA**: GPU acceleration platform
- **Neve Electronics**: Original 1073 preamp inspiration

## 📞 Contact

**Developer**: Louis Houston  
**Repository**: https://github.com/LouisBHouston/N73GPU  
**Issues**: https://github.com/LouisBHouston/N73GPU/issues

---

*Built with ❤️ and CUDA acceleration*