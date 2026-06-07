#pragma once
#include "BattlefieldContext.h"

namespace toy_acai
{
    class BattlefieldRenderer
    {
    public:
        BattlefieldRenderer();

        void render(const BattlefieldContext& context, double deltaTime);

        void EnableRenderToImageBuffer(Size size);

        const Image& imageBuffer() const;

    private:
        double v1(double value) const;

        Vec2 v2(const Vec2& pos) const;
        Vec2 v2(double x, double y) const;

        struct Impl;
        std::shared_ptr<Impl> p_impl;
    };
}
