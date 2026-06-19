#pragma once
#include "BattlefieldContext.h"

namespace toy_acai
{
    struct DistanceFromBoundary
    {
        double distance; // 境界内部なら正、境界外なら負
        double relativeAngle; // [-Pi, Pi] (は該当する境界辺の外向き法線を 0 とする相対角。例えば、上辺から見たとき上ベクトルの相対角度は 0 になる)
    };

    DistanceFromBoundary ComputeForwardDistanceFromBoundary(const BattlefieldContext& context, int fighterIndex);
}
