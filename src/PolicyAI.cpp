#include "PolicyAI.h"

#include "BattleFieldUtils.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <span>

namespace
{
    constexpr std::array<char, 8> ModelMagic{'T', 'A', 'C', 'A', 'I', 'P', 'O', 'L'};
    constexpr std::uint32_t ModelFormatVersion = 1;
    constexpr std::uint32_t MaximumHiddenDim = 4096;
    constexpr double DistanceNormalizationFactor = 1.0 / 500.0;
    constexpr double SpeedNormalizationFactor = 1.0 / 500.0;

    [[noreturn]] void ThrowModelError(FilePathView path, StringView reason)
    {
        throw Error{U"Failed to load AI policy '" + String{path} + U"': " + String{reason}};
    }

    void ReadExact(BinaryReader& reader, void* destination, std::size_t byteCount, FilePathView path)
    {
        if (reader.read(destination, static_cast<int64>(byteCount)) != static_cast<int64>(byteCount))
        {
            ThrowModelError(path, U"file is truncated");
        }
    }

    template <class Value>
    Value ReadValue(BinaryReader& reader, FilePathView path)
    {
        Value value{};
        ReadExact(reader, &value, sizeof(value), path);
        return value;
    }

    std::uint64_t ExpectedFloatCount(std::uint32_t observationDim, std::uint32_t hiddenDim)
    {
        const auto obs = static_cast<std::uint64_t>(observationDim);
        const auto hidden = static_cast<std::uint64_t>(hiddenDim);
        return hidden * obs + hidden + hidden * hidden + hidden
            + toy_acai::PolicyActionDim * hidden + toy_acai::PolicyActionDim
            + toy_acai::PolicyActionDim + hidden + 1;
    }

    float Sigmoid(float value)
    {
        if (value >= 0.0f)
        {
            return 1.0f / (1.0f + std::exp(-value));
        }
        const float expValue = std::exp(value);
        return expValue / (1.0f + expValue);
    }

    struct RelativeCandidate
    {
        double distanceSq;
        int index;
        toy_acai::RelativePose pose;
    };

    void SortPolicyCandidates(std::vector<RelativeCandidate>& candidates)
    {
        std::stable_sort(candidates.begin(), candidates.end(), [](const RelativeCandidate& left, const RelativeCandidate& right)
        {
            return left.distanceSq < right.distanceSq;
        });
    }
}

namespace toy_acai
{
    PolicyNetwork PolicyNetwork::Load(FilePathView path)
    {
        static_assert(std::endian::native == std::endian::little, "Policy files require a little-endian CPU");

        BinaryReader reader{path};
        if (!reader)
        {
            ThrowModelError(path, U"file could not be opened");
        }

        std::array<char, ModelMagic.size()> magic{};
        ReadExact(reader, magic.data(), magic.size(), path);
        if (magic != ModelMagic)
        {
            ThrowModelError(path, U"invalid magic");
        }

        const std::uint32_t version = ReadValue<std::uint32_t>(reader, path);
        const std::uint32_t observationDim = ReadValue<std::uint32_t>(reader, path);
        const std::uint32_t hiddenDim = ReadValue<std::uint32_t>(reader, path);
        const std::uint32_t actionDim = ReadValue<std::uint32_t>(reader, path);
        const std::uint32_t fireDim = ReadValue<std::uint32_t>(reader, path);
        const std::uint64_t floatCount = ReadValue<std::uint64_t>(reader, path);

        if (version != ModelFormatVersion)
        {
            ThrowModelError(path, U"unsupported format version " + Format(version));
        }
        if (observationDim != PolicyObservationDim)
        {
            ThrowModelError(path, U"expected 29 observation values, got " + Format(observationDim));
        }
        if (hiddenDim == 0 || hiddenDim > MaximumHiddenDim)
        {
            ThrowModelError(path, U"invalid hidden dimension " + Format(hiddenDim));
        }
        if (actionDim != PolicyActionDim || fireDim != 1)
        {
            ThrowModelError(path, U"expected 2 continuous actions and 1 fire logit");
        }

        const std::uint64_t expectedFloatCount = ExpectedFloatCount(observationDim, hiddenDim);
        if (floatCount != expectedFloatCount)
        {
            ThrowModelError(path, U"parameter count does not match the network dimensions");
        }
        if (floatCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float)))
        {
            ThrowModelError(path, U"parameter data is too large");
        }

        PolicyNetwork network;
        network.m_hiddenDim = hiddenDim;
        network.m_parameters.resize(static_cast<std::size_t>(floatCount));
        ReadExact(reader, network.m_parameters.data(), network.m_parameters.size() * sizeof(float), path);
        if (reader.getPos() != reader.size())
        {
            ThrowModelError(path, U"file contains trailing data");
        }
        if (!std::all_of(network.m_parameters.begin(), network.m_parameters.end(), [](float value)
        {
            return std::isfinite(value);
        }))
        {
            ThrowModelError(path, U"parameters contain NaN or infinity");
        }

        return network;
    }

    PolicyOutput PolicyNetwork::Forward(const PolicyObservation& observation) const
    {
        if (m_hiddenDim == 0 || m_parameters.empty())
        {
            throw Error{U"AI policy has not been loaded"};
        }
        if (!std::all_of(observation.begin(), observation.end(), [](float value)
        {
            return std::isfinite(value);
        }))
        {
            throw Error{U"AI policy observation contains NaN or infinity"};
        }

        const std::size_t hidden = m_hiddenDim;
        std::size_t offset = 0;
        const std::span<const float> firstWeights{m_parameters.data() + offset, hidden * PolicyObservationDim};
        offset += firstWeights.size();
        const std::span<const float> firstBias{m_parameters.data() + offset, hidden};
        offset += firstBias.size();
        const std::span<const float> secondWeights{m_parameters.data() + offset, hidden * hidden};
        offset += secondWeights.size();
        const std::span<const float> secondBias{m_parameters.data() + offset, hidden};
        offset += secondBias.size();
        const std::span<const float> actionWeights{m_parameters.data() + offset, PolicyActionDim * hidden};
        offset += actionWeights.size();
        const std::span<const float> actionBias{m_parameters.data() + offset, PolicyActionDim};
        offset += actionBias.size();
        const std::span<const float> logStd{m_parameters.data() + offset, PolicyActionDim};
        offset += logStd.size();
        const std::span<const float> fireWeights{m_parameters.data() + offset, hidden};
        offset += fireWeights.size();
        const float fireBias = m_parameters[offset];

        std::vector<float> firstHidden(hidden);
        std::vector<float> secondHidden(hidden);
        for (std::size_t output = 0; output < hidden; ++output)
        {
            float value = firstBias[output];
            const std::size_t row = output * PolicyObservationDim;
            for (std::size_t input = 0; input < PolicyObservationDim; ++input)
            {
                value += firstWeights[row + input] * observation[input];
            }
            firstHidden[output] = std::tanh(value);
        }
        for (std::size_t output = 0; output < hidden; ++output)
        {
            float value = secondBias[output];
            const std::size_t row = output * hidden;
            for (std::size_t input = 0; input < hidden; ++input)
            {
                value += secondWeights[row + input] * firstHidden[input];
            }
            secondHidden[output] = std::tanh(value);
        }

        PolicyOutput result{};
        for (std::size_t action = 0; action < PolicyActionDim; ++action)
        {
            float mean = actionBias[action];
            const std::size_t row = action * hidden;
            for (std::size_t input = 0; input < hidden; ++input)
            {
                mean += actionWeights[row + input] * secondHidden[input];
            }
            result.actionMean[action] = mean;
            result.actionStd[action] = std::exp(logStd[action]);
        }

        result.fireLogit = fireBias;
        for (std::size_t input = 0; input < hidden; ++input)
        {
            result.fireLogit += fireWeights[input] * secondHidden[input];
        }
        return result;
    }

    FighterInput PolicyNetwork::SampleAction(const PolicyObservation& observation, std::mt19937& randomEngine) const
    {
        const PolicyOutput output = Forward(observation);
        std::array<float, PolicyActionDim> rawAction{};
        for (std::size_t action = 0; action < PolicyActionDim; ++action)
        {
            std::normal_distribution<float> distribution{output.actionMean[action], output.actionStd[action]};
            rawAction[action] = distribution(randomEngine);
        }
        std::bernoulli_distribution fireDistribution{static_cast<double>(Sigmoid(output.fireLogit))};
        return FighterInput{
            static_cast<double>(std::tanh(rawAction[0])),
            static_cast<double>(std::tanh(rawAction[1])),
            fireDistribution(randomEngine),
        };
    }

    PolicyObservation BuildPolicyObservation(const BattlefieldContext& context, int fighterIndex)
    {
        if (fighterIndex < 0 || FighterCount <= fighterIndex)
        {
            throw Error{U"fighter index is outside the observation range"};
        }

        PolicyObservation observation{};
        std::size_t featureIndex = 0;
        const auto add = [&](float value)
        {
            observation[featureIndex++] = value;
        };

        const FighterState& fighter = context.fighters[fighterIndex];
        add(static_cast<float>(fighter.speed * SpeedNormalizationFactor));
        add(fighter.missileCooldown > 0.0 ? 1.0f : 0.0f);

        const DistanceFromBoundary boundary = ComputeDistanceFromBoundary(context, fighterIndex);
        add(static_cast<float>(boundary.distance * DistanceNormalizationFactor));
        add(static_cast<float>(std::cos(boundary.relativeAngle)));
        add(static_cast<float>(std::sin(boundary.relativeAngle)));

        const AbsolutePose fighterPose{fighter};
        std::vector<RelativeCandidate> opponents;
        opponents.reserve(FighterCount);
        for (int index = 0; index < FighterCount; ++index)
        {
            const FighterState& opponent = context.fighters[index];
            if (opponent.health <= 0.0 || opponent.teamId == fighter.teamId)
            {
                continue;
            }
            RelativePose pose = ComputeRelativePose(fighterPose, AbsolutePose{opponent});
            opponents.push_back(RelativeCandidate{pose.relativePosition.lengthSq(), index, pose});
        }
        SortPolicyCandidates(opponents);

        for (std::size_t slot = 0; slot < 2; ++slot)
        {
            if (slot < opponents.size())
            {
                const RelativeCandidate& candidate = opponents[slot];
                const FighterState& opponent = context.fighters[candidate.index];
                add(1.0f);
                add(static_cast<float>(candidate.pose.relativePosition.x * DistanceNormalizationFactor));
                add(static_cast<float>(candidate.pose.relativePosition.y * DistanceNormalizationFactor));
                add(static_cast<float>(std::cos(candidate.pose.relativeBearing)));
                add(static_cast<float>(std::sin(candidate.pose.relativeBearing)));
                add(static_cast<float>(opponent.speed * SpeedNormalizationFactor));
            }
            else
            {
                for (int feature = 0; feature < 6; ++feature)
                {
                    add(0.0f);
                }
            }
        }

        std::vector<RelativeCandidate> missiles;
        missiles.reserve(context.missiles.size());
        for (int index = 0; index < static_cast<int>(context.missiles.size()); ++index)
        {
            const MissileState& missile = context.missiles[index];
            if (missile.teamId == fighter.teamId)
            {
                continue;
            }
            RelativePose pose = ComputeRelativePose(fighterPose, AbsolutePose{missile});
            missiles.push_back(RelativeCandidate{pose.relativePosition.lengthSq(), index, pose});
        }
        SortPolicyCandidates(missiles);

        for (std::size_t slot = 0; slot < 2; ++slot)
        {
            if (slot < missiles.size())
            {
                const RelativeCandidate& candidate = missiles[slot];
                const MissileState& missile = context.missiles[candidate.index];
                add(1.0f);
                add(static_cast<float>(candidate.pose.relativePosition.x * DistanceNormalizationFactor));
                add(static_cast<float>(candidate.pose.relativePosition.y * DistanceNormalizationFactor));
                add(static_cast<float>(std::cos(candidate.pose.relativeBearing)));
                add(static_cast<float>(std::sin(candidate.pose.relativeBearing)));
                add(static_cast<float>(missile.speed * SpeedNormalizationFactor));
            }
            else
            {
                for (int feature = 0; feature < 6; ++feature)
                {
                    add(0.0f);
                }
            }
        }

        if (featureIndex != observation.size())
        {
            throw Error{U"internal error while building AI policy observation"};
        }
        return observation;
    }
}
