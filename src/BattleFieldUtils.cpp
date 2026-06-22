#include "BattleFieldUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{
    constexpr double Epsilon = 1e-9;

    struct BoundaryEdge
    {
        Vec2 start;
        Vec2 end;
        Vec2 outwardNormal;
    };

    bool IsInside(const Vec2& point, double width, double height)
    {
        return 0.0 <= point.x && point.x <= width && 0.0 <= point.y && point.y <= height;
    }

    double NormalizeAngle(double angle)
    {
        return Math::NormalizeAngle(angle, 0);
    }

    double DistancePointToSegment(const Vec2& point, const Vec2& segmentStart, const Vec2& segmentEnd)
    {
        const Vec2 segment = segmentEnd - segmentStart;
        const double segmentLengthSq = segment.dot(segment);
        if (segmentLengthSq <= Epsilon)
        {
            return point.distanceFrom(segmentStart);
        }

        const double t = std::clamp((point - segmentStart).dot(segment) / segmentLengthSq, 0.0, 1.0);
        return point.distanceFrom(segmentStart + segment * t);
    }

    double RelativeAngleFromEdge(const BoundaryEdge& edge, double yaw)
    {
        const double normalAngle = std::atan2(edge.outwardNormal.y, edge.outwardNormal.x);
        return NormalizeAngle(yaw - normalAngle);
    }
}

namespace toy_acai
{
    DistanceFromBoundary ComputeDistanceFromBoundary(const BattlefieldContext& context, int fighterIndex)
    {
        const auto& area = context.battlefieldArea;
        const auto& fighter = context.fighters[fighterIndex];

        const std::array<BoundaryEdge, 4> edges{
            BoundaryEdge{Vec2{0.0, 0.0}, Vec2{area.w, 0.0}, Vec2{0.0, -1.0}},
            BoundaryEdge{Vec2{area.w, 0.0}, Vec2{area.w, area.h}, Vec2{1.0, 0.0}},
            BoundaryEdge{Vec2{area.w, area.h}, Vec2{0.0, area.h}, Vec2{0.0, 1.0}},
            BoundaryEdge{Vec2{0.0, area.h}, Vec2{0.0, 0.0}, Vec2{-1.0, 0.0}},
        };

        const BoundaryEdge* nearestEdge = nullptr;
        double distance = std::numeric_limits<double>::max();
        for (const auto& edge : edges)
        {
            const double edgeDistance = DistancePointToSegment(fighter.position, edge.start, edge.end);
            if (edgeDistance < distance)
            {
                distance = edgeDistance;
                nearestEdge = &edge;
            }
        }

        const bool inside = IsInside(fighter.position, area.w, area.h);
        return DistanceFromBoundary{
            inside ? distance : -distance,
            nearestEdge != nullptr ? RelativeAngleFromEdge(*nearestEdge, fighter.yaw) : 0.0,
        };
    }

    RelativePose ComputeRelativePose(const AbsolutePose& fromPose, const AbsolutePose& toPose)
    {
        Vec2 relativePosition = (toPose.position - fromPose.position).rotated(-fromPose.yaw);
        relativePosition = relativePosition.yx();

        const double relativeBearing = toPose.yaw - fromPose.yaw;
        return RelativePose{
            relativePosition,
            relativeBearing
        };
    }
}
