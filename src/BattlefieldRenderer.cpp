#include "BattlefieldRenderer.h"
#include "BattlefieldRenderer.h"

using namespace toy_acai;

namespace
{
    ColorF GetTeamColor(int teamId)
    {
        switch (teamId)
        {
        case 0: return Palette::Dodgerblue;
        case 1: return Palette::Red;
        default: return Palette::Gray;
        }
    }
}

struct BattlefieldRenderer::Impl
{
    Image fighterImage = Icon::CreateImage(0xF0390, 256); // https://pictogrammers.com/library/mdi/icon/navigation/
    Texture fighterTexture{fighterImage};

    Image missileImage = Icon::CreateImage(0xF0079, 256); // https://pictogrammers.com/library/mdi/icon/battery/
    Texture missileTexture{missileImage};
};

namespace toy_acai
{
    BattlefieldRenderer::BattlefieldRenderer() : p_impl(std::make_shared<Impl>()) {}

    void BattlefieldRenderer::render(const BattlefieldContext& context)
    {
        // 背景を描画
        RectF{context.screenSize}.draw(ColorF{1.0f});

        // TODO: 中心基準で罫線を 16px 間隔で描画 (幅 1px)

        context.battlefieldArea.drawFrame(8.0, ColorF{0.1});

        // 戦闘機を描画
        for (const auto& fighter : context.fighters)
        {
            const Vec2 pos = context.battlefieldArea.pos + fighter.position;
            const double yaw = fighter.yaw;
            (void)p_impl->fighterTexture.resized(FighterSize).rotated(yaw).drawAt(pos, GetTeamColor(fighter.teamId));
        }

        // ミサイルを描画
        // TODO
    }
}
