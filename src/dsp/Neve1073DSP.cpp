#include "Neve1073DSP.h"
#include <cmath>

Neve1073DSP::Neve1073DSP() = default;
Neve1073DSP::~Neve1073DSP() = default;

void Neve1073DSP::prepare(const juce::dsp::ProcessSpec& spec)
{
    sampleRate = spec.sampleRate;
    maximumBlockSize = static_cast<int>(spec.maximumBlockSize);
    isInitialized = true;

    reset();

    updateInputStage();
    updateHighShelfEQ();
    updateMidEQ();
    updateLowShelfEQ();
    updateHighPassFilter();
    updateOutputStage();
}

void Neve1073DSP::processBlock(juce::AudioBuffer<float>& buffer)
{
    if (!isInitialized)
        return;

    auto numChannels = buffer.getNumChannels();
    auto numSamples  = buffer.getNumSamples();

    float inGain  = juce::Decibels::decibelsToGain(inputGain);
    float outGain = juce::Decibels::decibelsToGain(outputLevel);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* data = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i)
        {
            float sample = data[i] * inGain;

            if (!highShelfBypassed)
                sample = highShelfFilter.processSample(sample);
            if (!midEqBypassed)
                sample = midEqFilter.processSample(sample);
            if (!lowShelfBypassed)
                sample = lowShelfFilter.processSample(sample);
            if (!highPassBypassed)
                sample = highPassFilter.processSample(sample);

            data[i] = sample * outGain;
        }
    }

    // Simple placeholder metrics
    currentSaturation = 0.0f;
    currentTHD = 0.0f;
}

void Neve1073DSP::reset()
{
    inputStageFilter.reset();
    highShelfFilter.reset();
    midEqFilter.reset();
    lowShelfFilter.reset();
    highPassFilter.reset();
    outputStageFilter.reset();
}

void Neve1073DSP::setInputGain(float gainDb)
{
    inputGain = gainDb;
}

void Neve1073DSP::setHighShelfGain(float gainDb)
{
    highShelfGain = gainDb;
    updateHighShelfEQ();
}

void Neve1073DSP::setMidFrequency(float frequency)
{
    midFrequency = frequency;
    updateMidEQ();
}

void Neve1073DSP::setMidGain(float gainDb)
{
    midGain = gainDb;
    updateMidEQ();
}

void Neve1073DSP::setMidQ(float q)
{
    midQ = q;
    updateMidEQ();
}

void Neve1073DSP::setLowShelfFrequency(int index)
{
    lowShelfFreqIndex = juce::jlimit(0, (int)lowShelfFrequencies.size() - 1, index);
    updateLowShelfEQ();
}

void Neve1073DSP::setLowShelfGain(float gainDb)
{
    lowShelfGain = gainDb;
    updateLowShelfEQ();
}

void Neve1073DSP::setHighPassFrequency(int freqIndex)
{
    highPassFreqIndex = juce::jlimit(0, (int)highPassFrequencies.size() - 1, freqIndex);
    updateHighPassFilter();
}

void Neve1073DSP::setOutputLevel(float levelDb)
{
    outputLevel = levelDb;
}

void Neve1073DSP::setHighShelfBypass(bool bypass)
{
    highShelfBypassed = bypass;
}

void Neve1073DSP::setMidEqBypass(bool bypass)
{
    midEqBypassed = bypass;
}

void Neve1073DSP::setLowShelfBypass(bool bypass)
{
    lowShelfBypassed = bypass;
}

void Neve1073DSP::setHighPassBypass(bool bypass)
{
    highPassBypassed = bypass;
}

void Neve1073DSP::setSaturationAmount(float amount)
{
    saturationAmount = amount;
}

void Neve1073DSP::setTransformerModeling(bool enable)
{
    transformerModelingEnabled = enable;
}

void Neve1073DSP::setHarmonicContent(float amount)
{
    harmonicContent = amount;
}

std::array<float,5> Neve1073DSP::getEQResponse(float /*frequency*/) const
{
    return lastEQResponse;
}

juce::dsp::IIR::Coefficients<float>::Ptr Neve1073DSP::calculateHighShelfCoefficients()
{
    return juce::dsp::IIR::Coefficients<float>::makeHighShelf(sampleRate, 12000.0f, 0.707f,
                                                               juce::Decibels::decibelsToGain(highShelfGain));
}

juce::dsp::IIR::Coefficients<float>::Ptr Neve1073DSP::calculateMidEqCoefficients()
{
    return juce::dsp::IIR::Coefficients<float>::makePeakFilter(sampleRate, midFrequency, midQ,
                                                               juce::Decibels::decibelsToGain(midGain));
}

juce::dsp::IIR::Coefficients<float>::Ptr Neve1073DSP::calculateLowShelfCoefficients()
{
    float freq = lowShelfFrequencies[lowShelfFreqIndex];
    return juce::dsp::IIR::Coefficients<float>::makeLowShelf(sampleRate, freq, 0.707f,
                                                             juce::Decibels::decibelsToGain(lowShelfGain));
}

juce::dsp::IIR::Coefficients<float>::Ptr Neve1073DSP::calculateHighPassCoefficients()
{
    float freq = highPassFrequencies[highPassFreqIndex];
    if (freq <= 0.0f)
        freq = 20.0f;
    return juce::dsp::IIR::Coefficients<float>::makeHighPass(sampleRate, freq);
}

void Neve1073DSP::updateInputStage()
{
    juce::ignoreUnused(inputStageFilter);
}

void Neve1073DSP::updateHighShelfEQ()
{
    highShelfFilter.coefficients = calculateHighShelfCoefficients();
}

void Neve1073DSP::updateMidEQ()
{
    midEqFilter.coefficients = calculateMidEqCoefficients();
}

void Neve1073DSP::updateLowShelfEQ()
{
    lowShelfFilter.coefficients = calculateLowShelfCoefficients();
}

void Neve1073DSP::updateHighPassFilter()
{
    highPassFilter.coefficients = calculateHighPassCoefficients();
}

void Neve1073DSP::updateOutputStage()
{
    juce::ignoreUnused(outputStageFilter);
}

void Neve1073DSP::updateSaturationCurves() {}
void Neve1073DSP::updateTransformerModels() {}
void Neve1073DSP::calculateCurrentMetrics(const juce::AudioBuffer<float>&) {}

float Neve1073DSP::inputTransformerCurve(float sample) { return sample; }
float Neve1073DSP::outputTransformerCurve(float sample) { return sample; }
float Neve1073DSP::saturationCurve(float sample, float /*drive*/) { return sample; }
float Neve1073DSP::harmonicEnhancement(float sample) { return sample; }

