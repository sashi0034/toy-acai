#include "BattlefieldRenderer.h"

#include <algorithm>
#include <array>
#include <vector>

using namespace toy_acai;

namespace
{
    constexpr int FighterCount = TeamCount * TeamFighterCount;

    struct TrailPoint
    {
        Vec2 position; // battlefieldArea.pos からの相対座標
        double age;
    };

    bool IsAlive(const FighterState& fighter)
    {
        return fighter.health > 0.0;
    }

    ColorF GetTeamColor(int teamId)
    {
        switch (teamId)
        {
        case 0: return ColorF{Palette::Dodgerblue};
        case 1: return ColorF{Palette::Red};
        default: return ColorF{Palette::Gray};
        }
    }

    constexpr double trailDuration = 1.5;
}

struct BattlefieldRenderer::Impl
{
    Image fighterImage = Icon::CreateImage(0xF0390, 256); // https://pictogrammers.com/library/mdi/icon/navigation/
    Texture fighterTexture{fighterImage};

    Image missileImage = Icon::CreateImage(0xF0079, 256); // https://pictogrammers.com/library/mdi/icon/battery/
    Texture missileTexture{missileImage};

    std::array<std::vector<TrailPoint>, FighterCount> fighterTrails;

    void updateTrails(const BattlefieldContext& context, double deltaTime)
    {
        for (int i = 0; i < FighterCount; ++i)
        {
            auto& trail = fighterTrails[i];
            for (auto& point : trail)
            {
                point.age += deltaTime;
            }

            trail.erase(
                std::remove_if(
                    trail.begin(),
                    trail.end(),
                    [](const TrailPoint& point)
                    {
                        return trailDuration < point.age;
                    }),
                trail.end());

            if (IsAlive(context.fighters[i]))
            {
                trail.push_back(TrailPoint{context.fighters[i].position, 0.0});
            }
        }
    }
};

namespace toy_acai
{
    BattlefieldRenderer::BattlefieldRenderer() : p_impl(std::make_shared<Impl>()) {}

    void BattlefieldRenderer::render(const BattlefieldContext& context, double deltaTime)
    {
        p_impl->updateTrails(context, deltaTime);

        // 背景を描画
        RectF{context.screenSize}.draw(ColorF{1.0f});

        // グリッドを描画
        const Vec2 gridCenter = context.battlefieldArea.pos + Vec2{context.battlefieldArea.w * 0.5, context.battlefieldArea.h * 0.5};
        for (double x = gridCenter.x; x <= context.screenSize.x; x += 16.0)
        {
            Line{Vec2{x, 0.0}, Vec2{x, context.screenSize.y}}.draw(1.0, ColorF{0.92});
        }
        for (double x = gridCenter.x - 16.0; 0.0 <= x; x -= 16.0)
        {
            Line{Vec2{x, 0.0}, Vec2{x, context.screenSize.y}}.draw(1.0, ColorF{0.92});
        }
        for (double y = gridCenter.y; y <= context.screenSize.y; y += 16.0)
        {
            Line{Vec2{0.0, y}, Vec2{context.screenSize.x, y}}.draw(1.0, ColorF{0.92});
        }
        for (double y = gridCenter.y - 16.0; 0.0 <= y; y -= 16.0)
        {
            Line{Vec2{0.0, y}, Vec2{context.screenSize.x, y}}.draw(1.0, ColorF{0.92});
        }

        context.battlefieldArea.drawFrame(8.0, ColorF{0.1});

        // 軌跡を描画
        for (size_t fighterIndex = 0; fighterIndex < p_impl->fighterTrails.size(); ++fighterIndex)
        {
            const auto& trail = p_impl->fighterTrails[fighterIndex];
            if (trail.size() < 2)
            {
                continue;
            }

            const int teamId = context.fighters[fighterIndex].teamId;
            for (size_t i = 1; i < trail.size(); ++i)
            {
                const Vec2 from = context.battlefieldArea.pos + trail[i - 1].position;
                const Vec2 to = context.battlefieldArea.pos + trail[i].position;
                const double alpha = 0.35 * (1.0 - std::min(trail[i].age / trailDuration, 1.0));
                (void)Line{from, to}.draw(2.0, GetTeamColor(teamId).withAlpha(alpha));
            }
        }

        // 戦闘機を描画
        for (const auto& fighter : context.fighters)
        {
            if (!IsAlive(fighter))
            {
                continue;
            }

            const Vec2 pos = context.battlefieldArea.pos + fighter.position;
            const double yaw = fighter.yaw;
            (void)p_impl->fighterTexture.resized(FighterSize).rotated(yaw + Math::HalfPi).drawAt(pos, GetTeamColor(fighter.teamId));
        }

        // ミサイルを描画
        for (const auto& missile : context.missiles)
        {
            const Vec2 pos = context.battlefieldArea.pos + missile.position;
            (void)p_impl->missileTexture.resized(MissileSize).rotated(missile.yaw + Math::HalfPi).drawAt(pos, GetTeamColor(missile.teamId));
        }
    }
}
