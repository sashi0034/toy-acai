#include "BattlefieldContext.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
    using ActionArray = nb::ndarray<const double, nb::shape<toy_acai::FighterCount, 3>, nb::device::cpu, nb::c_contig>;
    using Matrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

    constexpr size_t FighterColumnCount = 9;
    constexpr size_t MissileColumnCount = 8;

    Matrix MakeMatrix(std::vector<double>* values, size_t rows, size_t cols)
    {
        nb::capsule owner(values, [](void* ptr) noexcept
        {
            delete static_cast<std::vector<double>*>(ptr);
        });
        return Matrix(values->data(), {rows, cols}, owner);
    }

    Matrix MakeFighterMatrix(const toy_acai::BattlefieldContext& context)
    {
        auto* values = new std::vector<double>(toy_acai::FighterCount * FighterColumnCount);
        for (size_t i = 0; i < context.fighters.size(); ++i)
        {
            const auto& fighter = context.fighters[i];
            const size_t offset = i * FighterColumnCount;
            (*values)[offset + 0] = static_cast<double>(fighter.teamId);
            (*values)[offset + 1] = static_cast<double>(fighter.memberId);
            (*values)[offset + 2] = fighter.position.x;
            (*values)[offset + 3] = fighter.position.y;
            (*values)[offset + 4] = fighter.yaw;
            (*values)[offset + 5] = fighter.speed;
            (*values)[offset + 6] = fighter.health;
            (*values)[offset + 7] = fighter.missileCooldown;
            (*values)[offset + 8] = fighter.outOfBoundsTime;
        }
        return MakeMatrix(values, toy_acai::FighterCount, FighterColumnCount);
    }

    Matrix MakeMissileMatrix(const toy_acai::BattlefieldContext& context)
    {
        const size_t rows = context.missiles.size();
        auto* values = new std::vector<double>(std::max<size_t>(1, rows * MissileColumnCount));
        for (size_t i = 0; i < rows; ++i)
        {
            const auto& missile = context.missiles[i];
            const size_t offset = i * MissileColumnCount;
            (*values)[offset + 0] = missile.position.x;
            (*values)[offset + 1] = missile.position.y;
            (*values)[offset + 2] = missile.yaw;
            (*values)[offset + 3] = missile.speed;
            (*values)[offset + 4] = missile.age;
            (*values)[offset + 5] = missile.lockLostTime;
            (*values)[offset + 6] = static_cast<double>(missile.teamId);
            (*values)[offset + 7] = static_cast<double>(missile.targetFighterIndex);
        }
        return MakeMatrix(values, rows, MissileColumnCount);
    }

    class BattlefieldEnv
    {
    public:
        BattlefieldEnv()
        {
            toy_acai::InitBattlefield(m_context);
        }

        nb::dict reset()
        {
            toy_acai::InitBattlefield(m_context);
            return observation();
        }

        nb::dict step(ActionArray actions, double deltaTime)
        {
            if (!(deltaTime > 0.0))
            {
                throw std::invalid_argument("deltaTime must be positive");
            }

            std::array<toy_acai::FighterInput, toy_acai::FighterCount> inputs{};
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                inputs[i] = toy_acai::FighterInput{
                    std::clamp(actions(i, 0), -1.0, 1.0),
                    std::clamp(actions(i, 1), -1.0, 1.0),
                    actions(i, 2) >= 0.5,
                };
            }

            toy_acai::UpdateBattlefield(m_context, inputs, deltaTime);
            return observation();
        }

    private:
        nb::dict observation() const
        {
            nb::dict result;
            result["fighters"] = MakeFighterMatrix(m_context);
            result["missiles"] = MakeMissileMatrix(m_context);
            result["screen_size"] = nb::make_tuple(m_context.screenSize.x, m_context.screenSize.y);
            result["battlefield"] = nb::make_tuple(
                m_context.battlefieldArea.x,
                m_context.battlefieldArea.y,
                m_context.battlefieldArea.w,
                m_context.battlefieldArea.h);
            result["fighter_count"] = toy_acai::FighterCount;
            result["fighter_columns"] = FighterColumnCount;
            result["missile_columns"] = MissileColumnCount;
            return result;
        }

        toy_acai::BattlefieldContext m_context{};
    };
}

NB_MODULE(toy_acai_core, m)
{
    m.doc() = "Headless Python bindings for the toy-acai air combat simulator.";
    m.attr("FIGHTER_COUNT") = toy_acai::FighterCount;
    m.attr("TEAM_COUNT") = toy_acai::TeamCount;
    m.attr("TEAM_FIGHTER_COUNT") = toy_acai::TeamFighterCount;
    m.attr("FIGHTER_COLUMNS") = FighterColumnCount;
    m.attr("MISSILE_COLUMNS") = MissileColumnCount;

    nb::class_<BattlefieldEnv>(m, "BattlefieldEnv")
        .def(nb::init<>())
        .def("reset", &BattlefieldEnv::reset)
        .def("step", &BattlefieldEnv::step, "actions"_a, "dt"_a = 0.1);
}
