#pragma once

namespace toy_acai
{
    constexpr int TeamCount = 2;

    constexpr int TeamFighterCount = 4;

    constexpr int MaxMissileCount = 64;

    constexpr double FighterSize = 32.0;

    constexpr double FighterMaxSpeed = 360.0;

    constexpr double FighterAcceleration = 260.0;

    constexpr double FighterDrag = 0.985;

    constexpr double FighterTurnRate = 2.0;

    constexpr double FighterInitialHealth = 1.0;

    constexpr double OutOfBoundsDeathTime = 3.0;

    constexpr double TrailDuration = 1.5;

    constexpr double MissileSize = 16.0;

    constexpr double MissileSpeed = 500.0;

    constexpr double MissileTurnRate = 1.5;

    constexpr double MissileLifetime = 6.0;

    constexpr double MissileFireCooldown = 1.0;

    constexpr double MissileHitRadius = 28.0;
}
