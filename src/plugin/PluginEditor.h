#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

//==============================================================================
/**
    N73GPU Editor - Basic GUI for Milestone 1
    
    This AudioProcessorEditor provides a simple interface for the N73GPU plugin.
    In Milestone 6, this will be replaced with the full professional GUI.
*/
class N73GPUAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    N73GPUAudioProcessorEditor (N73GPUAudioProcessor&);
    ~N73GPUAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // Reference to the processor
    N73GPUAudioProcessor& audioProcessor;
    
    // Basic GUI components for Milestone 1
    juce::Label titleLabel;
    juce::Label versionLabel;
    juce::Label statusLabel;
    
    // Parameter controls (basic sliders for now)
    juce::Slider inputGainSlider;
    juce::Label inputGainLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> inputGainAttachment;
    
    juce::Slider outputLevelSlider;
    juce::Label outputLevelLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> outputLevelAttachment;
    
    // Bypass button
    juce::TextButton bypassButton;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (N73GPUAudioProcessorEditor)
};