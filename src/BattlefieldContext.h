#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "BattlefieldParameters.h"

namespace toy_acai
{
    struct FighterState
    {
        int teamId;
        int memberId; // 同じチーム内での識別 ID
        Vec2 position; // battlefieldArea.pos からの相対座標
        double yaw;
        double speed;
        double health;
        double missileCooldown;
        double outOfBoundsTime;
    };

    struct MissileState
    {
        std::uint64_t id = -1;
        int teamId;
        int firedFrame;
        int shooterFighterIndex;
        int targetFighterIndex;
        Vec2 position; // battlefieldArea.pos からの相対座標
        double yaw;
        double speed;
        double age;
        double lockLostTime;
    };

    struct DeathEvent
    {
        enum class Reason
        {
            OutOfBounds,
            HitByMissile,
        };

        Reason reason;
        int fighterIndex;
        MissileState killerMissile;
    };

    struct BattlefieldContext
    {
        int frameCount;
        std::array<FighterState, TeamCount * TeamFighterCount> fighters;
        std::vector<MissileState> missiles;
        std::vector<DeathEvent> deathEvents;
        std::uint64_t nextMissileId = 0;
        Vec2 screenSize;
        RectF battlefieldArea;
        double battlefieldDiagonalLength;
    };

    struct FighterInput
    {
        double acceleration; // [-1.0, 1.0]
        double turn; // [-1.0, 1.0]
        bool fire;
    };

    void InitBattlefield(BattlefieldContext& context);

    void UpdateBattlefield(BattlefieldContext& context, const std::array<FighterInput, FighterCount>& inputs, double deltaTime);
}
