#include "PolicyAI.h"

#include <Siv3D.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace toy_acai;

namespace
{
    void Check(bool condition, const char* message)
    {
        if (!condition)
        {
            throw std::runtime_error{message};
        }
    }

    void CheckClose(float actual, float expected, float tolerance, const char* message)
    {
        Check(std::abs(actual - expected) <= tolerance, message);
    }

    std::vector<char> ReadFile(const std::string& path)
    {
        std::ifstream input{path, std::ios::binary};
        Check(static_cast<bool>(input), "could not open test policy file");
        return std::vector<char>{std::istreambuf_iterator<char>{input}, {}};
    }

    void WriteFile(const char* path, const std::vector<char>& bytes)
    {
        std::ofstream output{path, std::ios::binary | std::ios::trunc};
        Check(static_cast<bool>(output), "could not create malformed policy file");
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        Check(static_cast<bool>(output), "could not write malformed policy file");
    }

    template <class Value>
    void SetValue(std::vector<char>& bytes, std::size_t offset, Value value)
    {
        Check(offset + sizeof(value) <= bytes.size(), "test binary offset is invalid");
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    void ExpectLoadFailure(const std::vector<char>& bytes)
    {
        constexpr const char* NativePath = "/tmp/toy_acai_invalid_policy.bin";
        constexpr FilePathView Siv3DPath = U"/tmp/toy_acai_invalid_policy.bin";
        WriteFile(NativePath, bytes);
        bool failed = false;
        try
        {
            static_cast<void>(PolicyNetwork::Load(Siv3DPath));
        }
        catch (const Error&)
        {
            failed = true;
        }
        std::remove(NativePath);
        Check(failed, "malformed policy was accepted");
    }

    void TestForward(const PolicyNetwork& policy)
    {
        Check(policy.hiddenDim() == 256, "hidden dimension was not loaded");
        PolicyObservation observation{};
        for (std::size_t i = 0; i < observation.size(); ++i)
        {
            observation[i] = (static_cast<float>(i) - 14.0f) / 10.0f;
        }

        const PolicyOutput output = policy.Forward(observation);
        CheckClose(output.actionMean[0], -2.47135973f, 3e-5f, "first action mean differs from PyTorch");
        CheckClose(output.actionMean[1], 1.07498229f, 3e-5f, "second action mean differs from PyTorch");
        CheckClose(output.actionStd[0], 0.59353286f, 1e-6f, "first action std differs from PyTorch");
        CheckClose(output.actionStd[1], 0.50985479f, 1e-6f, "second action std differs from PyTorch");
        CheckClose(output.fireLogit, -4.37250233f, 3e-5f, "fire logit differs from PyTorch");

        std::mt19937 randomEngine{12345};
        const FighterInput input = policy.SampleAction(observation, randomEngine);
        Check(std::isfinite(input.acceleration) && -1.0 <= input.acceleration && input.acceleration <= 1.0, "sampled acceleration is invalid");
        Check(std::isfinite(input.turn) && -1.0 <= input.turn && input.turn <= 1.0, "sampled turn is invalid");
    }

    void TestObservation()
    {
        BattlefieldContext battlefield{};
        InitBattlefield(battlefield);
        for (auto& fighter : battlefield.fighters)
        {
            fighter.health = 0.0;
        }

        FighterState& observer = battlefield.fighters[1];
        observer.health = 1.0;
        observer.position = Vec2{100.0, 200.0};
        observer.yaw = 0.0;
        observer.speed = 250.0;
        observer.missileCooldown = 1.0;

        FighterState& nearest = battlefield.fighters[4];
        nearest.health = 1.0;
        nearest.position = Vec2{100.0, 300.0};
        nearest.yaw = 0.0;
        nearest.speed = 100.0;

        FighterState& second = battlefield.fighters[5];
        second.health = 1.0;
        second.position = Vec2{300.0, 200.0};
        second.yaw = Math::HalfPi;
        second.speed = 200.0;

        battlefield.missiles.clear();
        battlefield.missiles.push_back(MissileState{
            1, 1, 0, 4, 1, Vec2{100.0, 250.0}, 0.0, 300.0, 0.0});
        battlefield.missiles.push_back(MissileState{
            2, 0, 0, 1, 4, Vec2{100.0, 210.0}, 0.0, 300.0, 0.0});

        const PolicyObservation observation = BuildPolicyObservation(battlefield, 1);
        CheckClose(observation[0], 0.5f, 1e-6f, "speed normalization is wrong");
        CheckClose(observation[1], 1.0f, 0.0f, "cooldown compatibility feature is wrong");
        CheckClose(observation[5], 1.0f, 0.0f, "nearest opponent is missing");
        CheckClose(observation[6], 0.2f, 1e-6f, "nearest opponent relative x is wrong");
        CheckClose(observation[7], 0.0f, 1e-6f, "nearest opponent relative y is wrong");
        CheckClose(observation[10], 0.2f, 1e-6f, "nearest opponent speed is wrong");
        CheckClose(observation[12], 0.0f, 1e-6f, "second opponent relative x is wrong");
        CheckClose(observation[13], 0.4f, 1e-6f, "second opponent relative y is wrong");
        CheckClose(observation[15], 1.0f, 1e-6f, "second opponent bearing sine is wrong");
        CheckClose(observation[17], 1.0f, 0.0f, "hostile missile is missing");
        CheckClose(observation[18], 0.1f, 1e-6f, "hostile missile relative x is wrong");
        CheckClose(observation[23], 0.0f, 0.0f, "friendly missile was not excluded");

        const PolicyObservation otherTeam = BuildPolicyObservation(battlefield, 4);
        CheckClose(otherTeam[5], 1.0f, 0.0f, "team-relative opponent filtering is wrong");
    }

    void TestMalformedModels(const std::vector<char>& original)
    {
        Check(original.size() > 40, "test policy file is unexpectedly small");

        auto invalidMagic = original;
        invalidMagic[0] = 'X';
        ExpectLoadFailure(invalidMagic);

        auto invalidVersion = original;
        SetValue<std::uint32_t>(invalidVersion, 8, 2);
        ExpectLoadFailure(invalidVersion);

        auto invalidDimension = original;
        SetValue<std::uint32_t>(invalidDimension, 12, 28);
        ExpectLoadFailure(invalidDimension);

        auto truncated = original;
        truncated.pop_back();
        ExpectLoadFailure(truncated);

        auto trailing = original;
        trailing.push_back(0);
        ExpectLoadFailure(trailing);

        auto nonFinite = original;
        SetValue<float>(nonFinite, 36, std::numeric_limits<float>::quiet_NaN());
        ExpectLoadFailure(nonFinite);
    }
}

int main(int argc, char** argv)
{
    try
    {
        Check(argc == 2, "usage: policy-tests MODEL_PATH");
        const FilePath modelPath = Unicode::FromUTF8(argv[1]);
        const PolicyNetwork policy = PolicyNetwork::Load(modelPath);
        TestForward(policy);
        TestObservation();
        TestMalformedModels(ReadFile(argv[1]));
        std::cout << "Policy AI tests passed\n";
        return 0;
    }
    catch (const Error& error)
    {
        std::cerr << error << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
    }
    return 1;
}
