#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
N73GPUAudioProcessor::N73GPUAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
#endif
      parameters (*this, nullptr, juce::Identifier ("N73GPU"), createParameterLayout())
{
    // Initialize DSP core
    dspCore.initialize();
}

N73GPUAudioProcessor::~N73GPUAudioProcessor()
{
}

//==============================================================================
const juce::String N73GPUAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool N73GPUAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool N73GPUAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool N73GPUAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double N73GPUAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int N73GPUAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int N73GPUAudioProcessor::getCurrentProgram()
{
    return 0;
}

void N73GPUAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String N73GPUAudioProcessor::getProgramName (int index)
{
    return {};
}

void N73GPUAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void N73GPUAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBufferSize = samplesPerBlock;
    
    // Prepare DSP core
    dspCore.prepareToPlay(sampleRate, samplesPerBlock);
    
    isPluginInitialized = true;
}

void N73GPUAudioProcessor::releaseResources()
{
    dspCore.releaseResources();
    isPluginInitialized = false;
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool N73GPUAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void N73GPUAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear any output channels that don't contain input data
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // For Milestone 1: Simple passthrough processing with parameter control
    // Later this will be replaced with full N73GPU DSP processing
    if (isPluginInitialized)
    {
        // Update DSP core with current parameter values
        dspCore.setInputGain(*parameters.getRawParameterValue("inputGain"));
        dspCore.setOutputLevel(*parameters.getRawParameterValue("outputLevel"));
        dspCore.setHighShelfGain(*parameters.getRawParameterValue("highShelfGain"));
        dspCore.setMidFrequency(*parameters.getRawParameterValue("midFreq"));
        dspCore.setMidGain(*parameters.getRawParameterValue("midGain"));
        dspCore.setLowShelfGain(*parameters.getRawParameterValue("lowShelfGain"));
        
        // Process MIDI messages for future hardware integration
        for (const auto metadata : midiMessages)
        {
            auto message = metadata.getMessage();
            dspCore.processMidiMessage(message);
        }
        
        // Process audio (now with parameter control!)
        dspCore.processBlock(buffer);
    }
}

//==============================================================================
bool N73GPUAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* N73GPUAudioProcessor::createEditor()
{
    return new N73GPUAudioProcessorEditor (*this);
}

//==============================================================================
void N73GPUAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void N73GPUAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xmlState));
}

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout N73GPUAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Input Gain: -20dB to +20dB
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "inputGain",
        "Input Gain",
        juce::NormalisableRange<float> (-20.0f, 20.0f, 0.1f),
        0.0f));

    // High Shelf: ±16dB at 12kHz
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "highShelfGain",
        "High Shelf Gain",
        juce::NormalisableRange<float> (-16.0f, 16.0f, 0.1f),
        0.0f));

    // Mid EQ Frequency: 360Hz - 7.2kHz
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "midFreq",
        "Mid Frequency",
        juce::NormalisableRange<float> (360.0f, 7200.0f, 1.0f, 0.3f),
        1600.0f));

    // Mid EQ Gain: ±18dB
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "midGain",
        "Mid Gain",
        juce::NormalisableRange<float> (-18.0f, 18.0f, 0.1f),
        0.0f));

    // Low Shelf Frequency: 35/60/110/220 Hz
    layout.add (std::make_unique<juce::AudioParameterChoice> (
        "lowShelfFreq",
        "Low Shelf Frequency",
        juce::StringArray { "35Hz", "60Hz", "110Hz", "220Hz" },
        2));

    // Low Shelf Gain: ±16dB
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "lowShelfGain",
        "Low Shelf Gain",
        juce::NormalisableRange<float> (-16.0f, 16.0f, 0.1f),
        0.0f));

    // High Pass Filter: 50/80/160/300 Hz + Off
    layout.add (std::make_unique<juce::AudioParameterChoice> (
        "hpfFreq",
        "HPF Frequency",
        juce::StringArray { "Off", "50Hz", "80Hz", "160Hz", "300Hz" },
        0));

    // Output Level: -20dB to +10dB
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        "outputLevel",
        "Output Level",
        juce::NormalisableRange<float> (-20.0f, 10.0f, 0.1f),
        0.0f));

    return layout;
}

//==============================================================================
// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new N73GPUAudioProcessor();
}