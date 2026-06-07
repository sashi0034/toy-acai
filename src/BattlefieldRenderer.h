#pragma once
#include "BattlefieldContext.h"

namespace toy_acai
{
    class BattlefieldRenderer
    {
    public:
        BattlefieldRenderer();

        void render(const BattlefieldContext& context, double deltaTime);

    private:
        struct Impl;
        std::shared_ptr<Impl> p_impl;
    };
}
