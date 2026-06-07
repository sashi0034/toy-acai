#include "BattlefieldContext.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using namespace toy_acai;

namespace
{
    constexpr int FighterCount = TeamCount * TeamFighterCount;
    constexpr double TwoPi = 6.28318530717958647692;

    Vec2 Forward(double yaw)
    {
        return Vec2{std::cos(yaw), std::sin(yaw)};
    }

    double DistanceSq(const Vec2& a, const Vec2& b)
    {
        const Vec2 d = a - b;
        return d.x * d.x + d.y * d.y;
    }

    double NormalizeAngle(double angle)
    {
        return std::remainder(angle, TwoPi);
    }

    bool IsAlive(const FighterState& fighter)
    {
        return fighter.health > 0.0;
    }

    bool IsInBattlefield(const BattlefieldContext& context, const Vec2& position)
    {
        return 0.0 <= position.x && position.x <= context.battlefieldArea.w && 0.0 <= position.y && position.y <= context.battlefieldArea.h;
    }

    int FindMissileTarget(const BattlefieldContext& context, int shooterIndex)
    {
        const auto& shooter = context.fighters[shooterIndex];
        const Vec2 forward = Forward(shooter.yaw);
        int nearestIndex = -1;
        int nearestFrontIndex = -1;
        double nearestDistanceSq = std::numeric_limits<double>::max();
        double nearestFrontDistanceSq = std::numeric_limits<double>::max();

        for (int i = 0; i < FighterCount; ++i)
        {
            const auto& target = context.fighters[i];
            if (i == shooterIndex || !IsAlive(target) || target.teamId == shooter.teamId)
            {
                continue;
            }

            const Vec2 toTarget = target.position - shooter.position;
            const double distanceSq = toTarget.x * toTarget.x + toTarget.y * toTarget.y;
            if (distanceSq < nearestDistanceSq)
            {
                nearestDistanceSq = distanceSq;
                nearestIndex = i;
            }

            const double distance = std::sqrt(distanceSq);
            if (distance <= 0.0)
            {
                continue;
            }

            const double dot = (toTarget.x * forward.x + toTarget.y * forward.y) / distance;
            if (dot >= 0.35 && distanceSq < nearestFrontDistanceSq)
            {
                nearestFrontDistanceSq = distanceSq;
                nearestFrontIndex = i;
            }
        }

        return nearestFrontIndex != -1 ? nearestFrontIndex : nearestIndex;
    }

    void FireMissile(BattlefieldContext& context, int shooterIndex)
    {
        if (context.missiles.size() >= MaxMissileCount)
        {
            return;
        }

        auto& shooter = context.fighters[shooterIndex];
        if (!IsAlive(shooter) || shooter.missileCooldown > 0.0)
        {
            return;
        }

        const int targetIndex = FindMissileTarget(context, shooterIndex);
        if (targetIndex == -1)
        {
            return;
        }

        const Vec2 forward = Forward(shooter.yaw);
        context.missiles.push_back(MissileState{
            shooter.position + forward * (FighterSize * 0.75),
            shooter.yaw,
            MissileSpeed,
            0.0,
            shooter.teamId,
            targetIndex,
        });
        shooter.missileCooldown = MissileFireCooldown;
    }

    void UpdateFighters(BattlefieldContext& context, const FighterInput& input, double deltaTime)
    {
        for (int i = 0; i < FighterCount; ++i)
        {
            auto& fighter = context.fighters[i];
            if (!IsAlive(fighter))
            {
                continue;
            }

            fighter.missileCooldown = std::max(0.0, fighter.missileCooldown - deltaTime);
            fighter.yaw = NormalizeAngle(fighter.yaw + input.turn * FighterTurnRate * deltaTime);

            constexpr double minimumSpeed = 50.0;
            fighter.speed *= std::pow(FighterDrag, deltaTime * 60.0);
            fighter.speed = std::clamp(fighter.speed + input.acceleration * FighterAcceleration * deltaTime, minimumSpeed, FighterMaxSpeed);

            fighter.position += Forward(fighter.yaw) * fighter.speed * deltaTime;

            if (IsInBattlefield(context, fighter.position))
            {
                fighter.outOfBoundsTime = 0.0;
            }
            else
            {
                fighter.outOfBoundsTime += deltaTime;
                if (OutOfBoundsDeathTime <= fighter.outOfBoundsTime)
                {
                    fighter.health = 0.0;
                    continue;
                }
            }

            if (input.fire)
            {
                FireMissile(context, i);
            }
        }
    }

    void UpdateMissiles(BattlefieldContext& context, double deltaTime)
    {
        std::vector<MissileState> missiles;
        missiles.reserve(context.missiles.size());

        for (auto missile : context.missiles)
        {
            missile.age += deltaTime;
            if (MissileLifetime < missile.age || missile.targetFighterIndex < 0 || FighterCount <= missile.targetFighterIndex)
            {
                continue;
            }

            auto& target = context.fighters[missile.targetFighterIndex];
            if (!IsAlive(target))
            {
                continue;
            }

            const Vec2 toTarget = target.position - missile.position;
            const double desiredYaw = std::atan2(toTarget.y, toTarget.x);
            const double yawDelta = NormalizeAngle(desiredYaw - missile.yaw);
            const double maxTurn = MissileTurnRate * deltaTime;
            missile.yaw += std::clamp(yawDelta, -maxTurn, maxTurn);

            missile.position += Forward(missile.yaw) * missile.speed * deltaTime;

            if (DistanceSq(missile.position, target.position) <= MissileHitRadius * MissileHitRadius)
            {
                target.health = 0.0;
                continue;
            }

            missiles.push_back(missile);
        }

        context.missiles = std::move(missiles);
    }
}

namespace toy_acai
{
    void InitBattlefield(BattlefieldContext& context)
    {
        context.screenSize = {1920, 1080};
        context.battlefieldArea = RectF{Arg::center = context.screenSize * 0.5f, Vec2{1600, 900}};
        context.missiles.clear();

        for (int team = 0; team < TeamCount; ++team)
        {
            for (int member = 0; member < TeamFighterCount; ++member)
            {
                const int index = team * TeamFighterCount + member;
                const double x = team == 0 ? context.battlefieldArea.w * 0.22 : context.battlefieldArea.w * 0.78;
                const double y = context.battlefieldArea.h * (member + 1.0) / (TeamFighterCount + 1.0);

                context.fighters[index] = FighterState{
                    team,
                    Vec2{x, y},
                    team == 0 ? 0.0 : 3.14159265358979323846,
                    FighterMaxSpeed * 0.35,
                    FighterInitialHealth,
                    0.0,
                    0.0,
                };
            }
        }
    }

    void UpdateBattlefield(BattlefieldContext& context, const FighterInput& input, double deltaTime)
    {
        UpdateFighters(context, input, deltaTime);

        UpdateMissiles(context, deltaTime);
    }
}
