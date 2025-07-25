#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "../dsp/N73GPUCore.h"

//==============================================================================
/**
    N73GPU - Neve 1073 GPU-Accelerated Emulation
    
    This AudioProcessor implementation provides the main plugin interface
    for the N73GPU Neve 1073 emulation plugin.
*/
class N73GPUAudioProcessor : public juce::AudioProcessor
{
public:
    N73GPUAudioProcessor();
    ~N73GPUAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // Parameter management
    juce::AudioProcessorValueTreeState parameters;
    
private:
    //==============================================================================
    // DSP Core
    N73GPUCore dspCore;
    
    // Plugin state
    double currentSampleRate = 44100.0;
    int currentBufferSize = 512;
    bool isPluginInitialized = false;
    
    // Create parameter layout
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (N73GPUAudioProcessor)
};