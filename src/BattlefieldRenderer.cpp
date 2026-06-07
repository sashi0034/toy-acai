#include "BattlefieldRenderer.h"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

using namespace toy_acai;

namespace
{
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

    template <class...>
    constexpr bool DependentFalse = false;

    class PaintableTexture
    {
    public:
        explicit PaintableTexture(Image image)
            : m_image(std::move(image)), m_texture(m_image) {}

        void drawAt(const Vec2& pos, double angle, double size, const ColorF& color) const
        {
            (void)m_texture.resized(size).rotated(angle).drawAt(pos, color);
        }

        void paintAt(Image& dst, const Vec2& pos, double angle, double size, const ColorF& color) const
        {
            const Size imageSize{static_cast<int32>(size), static_cast<int32>(size)};
            m_image.scaled(imageSize).rotated(angle).paintAt(dst, pos.asPoint(), color.toColor());
        }

    private:
        Image m_image;
        Texture m_texture;
    };
}

struct BattlefieldRenderer::Impl
{
    Image m_imageBuffer{1920, 1080};
    bool m_renderToImageBuffer{};

    PaintableTexture fighterTexture{Icon::CreateImage(0xF0390, 256)}; // https://pictogrammers.com/library/mdi/icon/navigation/
    PaintableTexture missileTexture{Icon::CreateImage(0xF0079, 256)}; // https://pictogrammers.com/library/mdi/icon/battery/

    Font font{24};

    std::array<std::vector<TrailPoint>, FighterCount> fighterTrails;

    template <class Drawable, class... Args>
    void render(const Drawable& drawable, Args&&... args)
    {
        if (m_renderToImageBuffer)
        {
            if constexpr (requires { drawable.paint(m_imageBuffer, std::forward<Args>(args)...); })
            {
                (void)drawable.paint(m_imageBuffer, std::forward<Args>(args)...);
            }
            else
            {
                static_assert(DependentFalse<Drawable, Args...>, "Drawable must provide paint(Image&, ...) to render to the image buffer.");
            }
        }
        else
        {
            (void)drawable.draw(std::forward<Args>(args)...);
        }
    }

    template <class Drawable, class... Args>
    void renderAt(const Drawable& drawable, Args&&... args)
    {
        if (m_renderToImageBuffer)
        {
            if constexpr (requires { drawable.paintAt(m_imageBuffer, std::forward<Args>(args)...); })
            {
                (void)drawable.paintAt(m_imageBuffer, std::forward<Args>(args)...);
            }
            else
            {
                static_assert(DependentFalse<Drawable, Args...>, "Drawable must provide paintAt(Image&, ...) to render to the image buffer.");
            }
        }
        else
        {
            (void)drawable.drawAt(std::forward<Args>(args)...);
        }
    }

    template <class Drawable, class... Args>
    void renderFrame(const Drawable& drawable, Args&&... args)
    {
        if (m_renderToImageBuffer)
        {
            if constexpr (requires { drawable.paintFrame(m_imageBuffer, std::forward<Args>(args)...); })
            {
                (void)drawable.paintFrame(m_imageBuffer, std::forward<Args>(args)...);
            }
            else
            {
                static_assert(DependentFalse<Drawable, Args...>, "Drawable must provide paintFrame(Image&, ...) to render to the image buffer.");
            }
        }
        else
        {
            (void)drawable.drawFrame(std::forward<Args>(args)...);
        }
    }

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

    void BattlefieldRenderer::setRenderToImageBuffer(bool enabled)
    {
        p_impl->m_renderToImageBuffer = enabled;
    }

    const Image& BattlefieldRenderer::imageBuffer() const
    {
        return p_impl->m_imageBuffer;
    }

    void BattlefieldRenderer::render(const BattlefieldContext& context, double deltaTime)
    {
        p_impl->updateTrails(context, deltaTime);

        // 背景を描画
        p_impl->render(RectF{context.screenSize}, ColorF{1.0f}.toColor());

        // グリッドを描画
        const Vec2 gridCenter = context.battlefieldArea.pos + Vec2{context.battlefieldArea.w * 0.5, context.battlefieldArea.h * 0.5};
        for (double x = gridCenter.x; x <= context.screenSize.x; x += 16.0)
        {
            p_impl->render(Line{Vec2{x, 0.0}, Vec2{x, context.screenSize.y}}, 1, ColorF{0.92}.toColor());
        }
        for (double x = gridCenter.x - 16.0; 0.0 <= x; x -= 16.0)
        {
            p_impl->render(Line{Vec2{x, 0.0}, Vec2{x, context.screenSize.y}}, 1, ColorF{0.92}.toColor());
        }
        for (double y = gridCenter.y; y <= context.screenSize.y; y += 16.0)
        {
            p_impl->render(Line{Vec2{0.0, y}, Vec2{context.screenSize.x, y}}, 1, ColorF{0.92}.toColor());
        }
        for (double y = gridCenter.y - 16.0; 0.0 <= y; y -= 16.0)
        {
            p_impl->render(Line{Vec2{0.0, y}, Vec2{context.screenSize.x, y}}, 1, ColorF{0.92}.toColor());
        }

        p_impl->renderFrame(context.battlefieldArea, 4, 4, ColorF{0.1}.toColor());

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
                p_impl->render(Line{from, to}, 2, GetTeamColor(teamId).withAlpha(alpha).toColor());
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
            p_impl->renderAt(p_impl->fighterTexture, pos, yaw + Math::HalfPi, FighterSize, GetTeamColor(fighter.teamId));
        }

        // ミサイルを描画
        for (const auto& missile : context.missiles)
        {
            const Vec2 pos = context.battlefieldArea.pos + missile.position;
            p_impl->renderAt(p_impl->missileTexture, pos, missile.yaw + Math::HalfPi, MissileSize, GetTeamColor(missile.teamId));
        }

        // ファイター識別用の文字列を描画
        for (const auto& fighter : context.fighters)
        {
            if (!IsAlive(fighter))
            {
                continue;
            }

            String name = (fighter.teamId == 0 ? U"B" : U"R") + Format(fighter.memberId);
            const Vec2 pos = context.battlefieldArea.pos + fighter.position;
            p_impl->renderAt(p_impl->font(name), pos.movedBy(0, -FighterSize * 0.75), ColorF{0.1}.toColor());
        }

        // 各情報を描画
        {
            // 生存しているファイターの数
            int alive0 = 0;
            int alive1 = 0;
            for (const auto& fighter : context.fighters)
            {
                if (IsAlive(fighter))
                {
                    if (fighter.teamId == 0)
                    {
                        alive0++;
                    }
                    else
                    {
                        alive1++;
                    }
                }
            }

            String desc = U"Alive: " + Format(alive0) + U"-" + Format(alive1);
            p_impl->renderAt(p_impl->font(desc), (context.screenSize * 0.5).withY(64.0), ColorF{0.1}.toColor());
        }
    }
}
