#pragma once
#include <array>
#include <vector>

#include "BattlefieldParameters.h"

namespace toy_acai
{
    struct FighterState
    {
        int teamId;
        Vec2 position; // battlefieldArea.pos からの相対座標
        double yaw;
        double speed;
        double health;
        double missileCooldown;
        double outOfBoundsTime;
    };

    struct MissileState
    {
        Vec2 position; // battlefieldArea.pos からの相対座標
        double yaw;
        double speed;
        double age;
        int teamId;
        int targetFighterIndex;
    };

    struct BattlefieldContext
    {
        std::array<FighterState, TeamCount * TeamFighterCount> fighters;
        std::vector<MissileState> missiles;
        Float2 screenSize;
        RectF battlefieldArea;
    };

    struct FighterInput
    {
        double acceleration; // [-1.0, 1.0]
        double turn; // [-1.0, 1.0]
        bool fire;
    };

    void InitBattlefield(BattlefieldContext& context);

    void UpdateBattlefield(BattlefieldContext& context, const FighterInput& input, double deltaTime);
}
