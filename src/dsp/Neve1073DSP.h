#pragma once

#ifndef JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED
#define JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED 1
#endif

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <memory>

//==============================================================================
/**
    Neve1073DSP - Authentic Neve 1073 Preamp/EQ CPU Reference Implementation
    
    This class implements a detailed software model of the legendary Neve 1073 
    preamp and EQ section, including:
    
    - Input transformer and gain staging (-20dB to +80dB)
    - 3-band EQ with authentic frequency responses:
      * High shelf: ±16dB at 12kHz
      * Mid bell: ±18dB, 360Hz-7.2kHz, variable Q
      * Low shelf: ±16dB at 35/60/110/220Hz
    - High-pass filter: 50/80/160/300Hz
    - Output level control with saturation
    - Analog-style harmonic distortion and transformer modeling
    
    Milestone 4: CPU Reference Implementation for N73GPU plugin
*/
class Neve1073DSP
{
public:
    //==============================================================================
    Neve1073DSP();
    ~Neve1073DSP();

    //==============================================================================
    // DSP Lifecycle
    void prepare(const juce::dsp::ProcessSpec& spec);
    void processBlock(juce::AudioBuffer<float>& buffer);
    void reset();

    //==============================================================================
    // Neve 1073 Parameter Control (Authentic Ranges)
    void setInputGain(float gainDb);            // -20dB to +80dB (authentic range)
    void setHighShelfGain(float gainDb);        // ±16dB at 12kHz
    void setMidFrequency(float frequency);      // 360Hz to 7.2kHz (11 positions)
    void setMidGain(float gainDb);              // ±18dB
    void setMidQ(float q);                      // Q factor (0.3 to 4.0)
    void setLowShelfFrequency(int index);       // 0=35Hz, 1=60Hz, 2=110Hz, 3=220Hz
    void setLowShelfGain(float gainDb);         // ±16dB
    void setHighPassFrequency(int freqIndex);   // 0=off, 1=50Hz, 2=80Hz, 3=160Hz, 4=300Hz
    void setOutputLevel(float levelDb);         // -20dB to +10dB

    //==============================================================================
    // EQ Section Bypass Controls
    void setHighShelfBypass(bool bypass);
    void setMidEqBypass(bool bypass);
    void setLowShelfBypass(bool bypass);
    void setHighPassBypass(bool bypass);

    //==============================================================================
    // Advanced Analog Modeling Controls
    void setSaturationAmount(float amount);     // 0.0 to 1.0 (tube-like saturation)
    void setTransformerModeling(bool enable);   // Marinair transformer emulation
    void setHarmonicContent(float amount);      // Harmonic enhancement (0.0 to 1.0)

    //==============================================================================
    // Real-time Analysis and Metering
    float getCurrentSaturation() const { return currentSaturation; }
    float getTotalHarmonicDistortion() const { return currentTHD; }
    std::array<float, 5> getEQResponse(float frequency) const;

private:
    //==============================================================================
    // DSP State
    double sampleRate = 44100.0;
    int maximumBlockSize = 512;
    bool isInitialized = false;

    //==============================================================================
    // Neve 1073 Parameters
    float inputGain = 0.0f;         // -20 to +80 dB
    float highShelfGain = 0.0f;     // ±16 dB at 12kHz
    float midFrequency = 1500.0f;   // 360Hz to 7.2kHz
    float midGain = 0.0f;           // ±18 dB
    float midQ = 0.7f;              // Q factor
    int lowShelfFreqIndex = 2;      // 110Hz default
    float lowShelfGain = 0.0f;      // ±16 dB
    int highPassFreqIndex = 0;      // Off default
    float outputLevel = 0.0f;       // -20 to +10 dB

    //==============================================================================
    // Bypass States
    bool highShelfBypassed = false;
    bool midEqBypassed = false;
    bool lowShelfBypassed = false;
    bool highPassBypassed = true;   // HPF off by default

    //==============================================================================
    // Advanced Controls
    float saturationAmount = 0.3f;      // Default mild saturation
    bool transformerModelingEnabled = true;
    float harmonicContent = 0.2f;       // Default harmonic enhancement

    //==============================================================================
    // Analysis and Metering
    float currentSaturation = 0.0f;
    float currentTHD = 0.0f;
    std::array<float, 5> lastEQResponse = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    //==============================================================================
    // DSP Filter Objects (JUCE DSP)
    juce::dsp::IIR::Filter<float> inputStageFilter;
    juce::dsp::IIR::Filter<float> highShelfFilter;
    juce::dsp::IIR::Filter<float> midEqFilter;
    juce::dsp::IIR::Filter<float> lowShelfFilter;
    juce::dsp::IIR::Filter<float> highPassFilter;
    juce::dsp::IIR::Filter<float> outputStageFilter;

    //==============================================================================
    // Analog Modeling State
    float inputTransformerState = 0.0f;     // For transformer memory
    float outputTransformerState = 0.0f;
    float saturationMemory = 0.0f;          // For hysteresis-like effects
    
    // Authentic Neve 1073 frequency tables
    static constexpr std::array<float, 4> lowShelfFrequencies = {35.0f, 60.0f, 110.0f, 220.0f};
    static constexpr std::array<float, 5> highPassFrequencies = {0.0f, 50.0f, 80.0f, 160.0f, 300.0f}; // 0 = off

    //==============================================================================
    // Internal methods
    void updateInputStage();
    void updateHighShelfEQ();
    void updateMidEQ();
    void updateLowShelfEQ();
    void updateHighPassFilter();
    void updateOutputStage();
    void updateSaturationCurves();
    void updateTransformerModels();
    void calculateCurrentMetrics(const juce::AudioBuffer<float>& buffer);
    
    // Transformer modeling functions
    float inputTransformerCurve(float sample);
    float outputTransformerCurve(float sample);
    float saturationCurve(float sample, float drive);
    float harmonicEnhancement(float sample);
    
    // EQ coefficient calculation
    juce::dsp::IIR::Coefficients<float>::Ptr calculateHighShelfCoefficients();
    juce::dsp::IIR::Coefficients<float>::Ptr calculateMidEqCoefficients();
    juce::dsp::IIR::Coefficients<float>::Ptr calculateLowShelfCoefficients();
    juce::dsp::IIR::Coefficients<float>::Ptr calculateHighPassCoefficients();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Neve1073DSP)
};