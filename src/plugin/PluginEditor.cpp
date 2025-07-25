#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
N73GPUAudioProcessorEditor::N73GPUAudioProcessorEditor (N73GPUAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Set up the main window
    setSize (400, 300);
    
    // Configure title label
    titleLabel.setText ("N73GPU", juce::dontSendNotification);
    titleLabel.setFont (juce::Font (24.0f, juce::Font::bold));
    titleLabel.setJustificationType (juce::Justification::centred);
    titleLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible (titleLabel);
    
    // Configure version label
    versionLabel.setText ("Neve 1073 GPU Emulation v1.0", juce::dontSendNotification);
    versionLabel.setFont (juce::Font (12.0f));
    versionLabel.setJustificationType (juce::Justification::centred);
    versionLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (versionLabel);
    
    // Configure status label
    statusLabel.setText ("Milestone 1: Basic Plugin - Audio Passthrough", juce::dontSendNotification);
    statusLabel.setFont (juce::Font (10.0f));
    statusLabel.setJustificationType (juce::Justification::centred);
    statusLabel.setColour (juce::Label::textColourId, juce::Colours::yellow);
    addAndMakeVisible (statusLabel);
    
    // Configure Input Gain slider
    inputGainSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    inputGainSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 20);
    inputGainSlider.setColour (juce::Slider::rotarySliderFillColourId, juce::Colours::orange);
    addAndMakeVisible (inputGainSlider);
    
    inputGainLabel.setText ("Input Gain", juce::dontSendNotification);
    inputGainLabel.setFont (juce::Font (11.0f));
    inputGainLabel.setJustificationType (juce::Justification::centred);
    inputGainLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible (inputGainLabel);
    
    inputGainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.parameters, "inputGain", inputGainSlider);
    
    // Configure Output Level slider
    outputLevelSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    outputLevelSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 60, 20);
    outputLevelSlider.setColour (juce::Slider::rotarySliderFillColourId, juce::Colours::green);
    addAndMakeVisible (outputLevelSlider);
    
    outputLevelLabel.setText ("Output Level", juce::dontSendNotification);
    outputLevelLabel.setFont (juce::Font (11.0f));
    outputLevelLabel.setJustificationType (juce::Justification::centred);
    outputLevelLabel.setColour (juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible (outputLevelLabel);
    
    outputLevelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.parameters, "outputLevel", outputLevelSlider);
    
    // Configure bypass button
    bypassButton.setButtonText ("Bypass");
    bypassButton.setColour (juce::TextButton::buttonColourId, juce::Colours::darkred);
    bypassButton.setColour (juce::TextButton::textColourOnId, juce::Colours::white);
    addAndMakeVisible (bypassButton);
}

N73GPUAudioProcessorEditor::~N73GPUAudioProcessorEditor()
{
}

//==============================================================================
void N73GPUAudioProcessorEditor::paint (juce::Graphics& g)
{
    // Create Neve-inspired color scheme
    juce::Colour neveOrange (0xff, 0x8c, 0x00);
    juce::Colour neveGrey (0x2a, 0x2a, 0x2a);
    juce::Colour neveDarkGrey (0x1a, 0x1a, 0x1a);
    
    // Fill background with gradient
    g.fillAll (neveDarkGrey);
    
    juce::ColourGradient gradient (neveGrey, 0, 0, neveDarkGrey, 0, getHeight(), false);
    g.setGradientFill (gradient);
    g.fillRect (getLocalBounds());
    
    // Draw border
    g.setColour (neveOrange);
    g.drawRect (getLocalBounds(), 2);
    
    // Draw KZRACKS branding
    g.setColour (juce::Colours::lightgrey);
    g.setFont (juce::Font (8.0f));
    g.drawText ("KZRACKS", getWidth() - 50, getHeight() - 15, 45, 12, juce::Justification::right);
}

void N73GPUAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();
    auto margin = 10;
    
    // Title area
    titleLabel.setBounds (bounds.removeFromTop (40).reduced (margin));
    versionLabel.setBounds (bounds.removeFromTop (20).reduced (margin));
    statusLabel.setBounds (bounds.removeFromTop (15).reduced (margin));
    
    bounds.removeFromTop (10); // Spacer
    
    // Control area
    auto controlArea = bounds.removeFromTop (120);
    auto sliderWidth = 80;
    auto sliderHeight = 80;
    
    // Input Gain
    auto inputArea = controlArea.removeFromLeft (sliderWidth + 20);
    inputGainLabel.setBounds (inputArea.removeFromBottom (15));
    inputGainSlider.setBounds (inputArea.reduced (5));
    
    // Output Level  
    auto outputArea = controlArea.removeFromRight (sliderWidth + 20);
    outputLevelLabel.setBounds (outputArea.removeFromBottom (15));
    outputLevelSlider.setBounds (outputArea.reduced (5));
    
    // Bypass button
    bounds.removeFromTop (10); // Spacer
    bypassButton.setBounds (bounds.removeFromTop (30).reduced (margin * 4));
}