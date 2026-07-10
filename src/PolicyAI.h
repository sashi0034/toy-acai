#pragma once

#include <Siv3D.hpp>

#include "BattlefieldContext.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace toy_acai
{
    inline constexpr std::size_t PolicyObservationDim = 29;
    inline constexpr std::size_t PolicyActionDim = 2;

    using PolicyObservation = std::array<float, PolicyObservationDim>;

    struct PolicyOutput
    {
        std::array<float, PolicyActionDim> actionMean;
        std::array<float, PolicyActionDim> actionStd;
        float fireLogit;
    };

    class PolicyNetwork
    {
    public:
        [[nodiscard]] static PolicyNetwork Load(FilePathView path);

        [[nodiscard]] PolicyOutput Forward(const PolicyObservation& observation) const;

        [[nodiscard]] FighterInput SampleAction(const PolicyObservation& observation, std::mt19937& randomEngine) const;

        [[nodiscard]] std::size_t hiddenDim() const noexcept
        {
            return m_hiddenDim;
        }

    private:
        std::size_t m_hiddenDim{};
        std::vector<float> m_parameters;
    };

    [[nodiscard]] PolicyObservation BuildPolicyObservation(const BattlefieldContext& context, int fighterIndex);
}
