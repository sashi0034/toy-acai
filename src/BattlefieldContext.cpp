#include "BattlefieldContext.h"

using namespace toy_acai;

namespace
{
    // TODO: 基本的に static 関数は使わずに namespace 内で定義
}

namespace toy_acai
{
    void InitBattlefield(BattlefieldContext& context)
    {
        context.screenSize = {1920, 1080};
        context.battlefieldArea =
            RectF{Arg::center = context.screenSize * 0.5f, Vec2{1600, 900}};;

        // TODO: チームで向かい合うようにしてプレイヤーを配置, yaw も
        context.fighters[0].position = context.battlefieldArea.getRelativePoint(0.25, 0.5);
    }

    void UpdateBattlefield(BattlefieldContext& context, const FighterInput& input, double deltaTime)
    {
        // todo
    }
}
