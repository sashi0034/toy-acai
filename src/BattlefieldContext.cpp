#include "BattlefieldContext.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using namespace toy_acai;

namespace
{
    constexpr double fighterMaxSpeed = 360.0;

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
        int nearestFrontIndex = -1;
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
            const double distance = std::sqrt(distanceSq);
            if (distance <= 0.0)
            {
                continue;
            }

            const double dot = (toTarget.x * forward.x + toTarget.y * forward.y) / distance;
            constexpr double seekerHalfAngle = 0.85;
            if (dot >= std::cos(seekerHalfAngle) && distanceSq < nearestFrontDistanceSq)
            {
                nearestFrontDistanceSq = distanceSq;
                nearestFrontIndex = i;
            }
        }

        return nearestFrontIndex;
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
        constexpr double initialSpeed = 150.0;
        context.missiles.push_back(MissileState{
            shooter.position + forward * (FighterSize * 0.75),
            shooter.yaw,
            initialSpeed,
            0.0,
            0.0,
            shooter.teamId,
            targetIndex,
        });

        constexpr double cooldown = 3.5;
        shooter.missileCooldown = cooldown;
    }

    void UpdateFighters(BattlefieldContext& context, const std::array<FighterInput, FighterCount>& inputs, double deltaTime)
    {
        for (int i = 0; i < FighterCount; ++i)
        {
            auto& fighter = context.fighters[i];
            if (!IsAlive(fighter))
            {
                continue;
            }

            fighter.missileCooldown = std::max(0.0, fighter.missileCooldown - deltaTime);

            const FighterInput& input = inputs[i];

            constexpr double turnRate = 2.0;
            fighter.yaw = NormalizeAngle(fighter.yaw + input.turn * turnRate * deltaTime);

            constexpr double minimumSpeed = 25.0;
            constexpr double fighterDrag = 0.985;
            fighter.speed *= std::pow(fighterDrag, deltaTime * 60.0);

            constexpr double acceleration = 200.0;
            fighter.speed = std::clamp(fighter.speed + input.acceleration * acceleration * deltaTime, minimumSpeed, fighterMaxSpeed);

            fighter.position += Forward(fighter.yaw) * fighter.speed * deltaTime;

            if (IsInBattlefield(context, fighter.position))
            {
                fighter.outOfBoundsTime = 0.0;
            }
            else
            {
                fighter.outOfBoundsTime += deltaTime;
                constexpr double outOfBoundsDeathTime = 3.0;
                if (outOfBoundsDeathTime <= fighter.outOfBoundsTime)
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
            constexpr double lifetime = 6.0;
            if (lifetime < missile.age)
            {
                continue;
            }

            const bool hasTarget = 0 <= missile.targetFighterIndex && missile.targetFighterIndex < FighterCount;
            FighterState* target = hasTarget ? &context.fighters[missile.targetFighterIndex] : nullptr;
            const bool targetAlive = target != nullptr && IsAlive(*target);

            if (targetAlive)
            {
                const Vec2 toTarget = target->position - missile.position;
                const double desiredYaw = std::atan2(toTarget.y, toTarget.x);
                const double yawDelta = NormalizeAngle(desiredYaw - missile.yaw);
                constexpr double seekerHalfAngle = 0.85;
                if (std::abs(yawDelta) <= seekerHalfAngle)
                {
                    constexpr double turnRate = 1.5;
                    const double maxTurn = turnRate * deltaTime;
                    missile.yaw = NormalizeAngle(missile.yaw + std::clamp(yawDelta, -maxTurn, maxTurn));
                    missile.lockLostTime = 0.0;
                }
                else
                {
                    missile.lockLostTime += deltaTime;
                }
            }
            else
            {
                missile.lockLostTime += deltaTime;
            }

            constexpr double lockLostLifetime = 1.1;
            if (lockLostLifetime < missile.lockLostTime)
            {
                continue;
            }

            constexpr double boostDuration = 0.5;
            if (missile.age <= boostDuration)
            {
                constexpr double boostAcceleration = 100.0;
                missile.speed += boostAcceleration * deltaTime;
            }

            missile.position += Forward(missile.yaw) * missile.speed * deltaTime;

            constexpr double hitRadius = MissileSize;
            if (targetAlive && DistanceSq(missile.position, target->position) <= hitRadius * hitRadius)
            {
                target->health = 0.0;
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
                const double x = team == 0 ? context.battlefieldArea.w * 0.1 : context.battlefieldArea.w * 0.9;
                const double y = context.battlefieldArea.h * (team == 0 ? member + 1.0 : 3.0 * TeamFighterCount - member) / (3.0 * TeamFighterCount + 1.0);

                constexpr double fighterInitialHealth = 1.0;
                context.fighters[index] = FighterState{
                    team,
                    member,
                    Vec2{x, y},
                    team == 0 ? 0.0 : Math::Pi,
                    fighterMaxSpeed * 0.35,
                    fighterInitialHealth,
                    0.0,
                    0.0,
                };
            }
        }
    }

    void UpdateBattlefield(BattlefieldContext& context, const std::array<FighterInput, FighterCount>& inputs, double deltaTime)
    {
        UpdateFighters(context, inputs, deltaTime);

        UpdateMissiles(context, deltaTime);
    }
}
