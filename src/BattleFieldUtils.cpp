#include "BattleFieldUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace
{
    constexpr double Epsilon = 1e-9;

    struct HalfLine
    {
        Vec2 origin;
        Vec2 direction;
    };

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

    double DistancePointToHalfLine(const Vec2& point, const HalfLine& halfLine)
    {
        const double directionLengthSq = halfLine.direction.dot(halfLine.direction);
        if (directionLengthSq <= Epsilon)
        {
            return point.distanceFrom(halfLine.origin);
        }

        const double t = (point - halfLine.origin).dot(halfLine.direction) / directionLengthSq;
        if (t <= 0.0)
        {
            return point.distanceFrom(halfLine.origin);
        }

        return point.distanceFrom(halfLine.origin + halfLine.direction * t);
    }

    double HalfLineSegmentIntersectionDistance(const HalfLine& halfLine, const BoundaryEdge& edge)
    {
        const Vec2 segment = edge.end - edge.start;
        const double directionLength = halfLine.direction.length();
        if (directionLength <= Epsilon)
        {
            return std::numeric_limits<double>::max();
        }

        const double denominator = halfLine.direction.cross(segment);
        const Vec2 toSegmentStart = edge.start - halfLine.origin;
        if (std::abs(denominator) <= Epsilon)
        {
            if (std::abs(toSegmentStart.cross(halfLine.direction)) > Epsilon)
            {
                return std::numeric_limits<double>::max();
            }

            const double directionLengthSq = halfLine.direction.dot(halfLine.direction);
            const double t0 = (edge.start - halfLine.origin).dot(halfLine.direction) / directionLengthSq;
            const double t1 = (edge.end - halfLine.origin).dot(halfLine.direction) / directionLengthSq;
            const double maxT = std::max(t0, t1);
            if (maxT < -Epsilon)
            {
                return std::numeric_limits<double>::max();
            }

            return std::max(0.0, std::min(t0, t1)) * directionLength;
        }

        const double halfLineT = toSegmentStart.cross(segment) / denominator;
        const double segmentT = toSegmentStart.cross(halfLine.direction) / denominator;
        if (halfLineT < -Epsilon || segmentT < -Epsilon || 1.0 + Epsilon < segmentT)
        {
            return std::numeric_limits<double>::max();
        }

        return std::max(0.0, halfLineT) * directionLength;
    }

    double DistanceHalfLineToSegment(const HalfLine& halfLine, const BoundaryEdge& edge)
    {
        return std::min({
            DistancePointToHalfLine(edge.start, halfLine),
            DistancePointToHalfLine(edge.end, halfLine),
            DistancePointToSegment(halfLine.origin, edge.start, edge.end),
        });
    }

    double RelativeAngleFromEdge(const BoundaryEdge& edge, double yaw)
    {
        const double normalAngle = std::atan2(edge.outwardNormal.y, edge.outwardNormal.x);
        return NormalizeAngle(yaw - normalAngle);
    }
}

namespace toy_acai
{
    DistanceFromBoundary ComputeForwardDistanceFromBoundary(const BattlefieldContext& context, int fighterIndex)
    {
        const auto& area = context.battlefieldArea;
        const auto& fighter = context.fighters[fighterIndex];
        const HalfLine forward{
            fighter.position,
            Vec2{std::cos(fighter.yaw), std::sin(fighter.yaw)},
        };

        const std::array<BoundaryEdge, 4> edges{
            BoundaryEdge{Vec2{0.0, 0.0}, Vec2{area.w, 0.0}, Vec2{0.0, -1.0}},
            BoundaryEdge{Vec2{area.w, 0.0}, Vec2{area.w, area.h}, Vec2{1.0, 0.0}},
            BoundaryEdge{Vec2{area.w, area.h}, Vec2{0.0, area.h}, Vec2{0.0, 1.0}},
            BoundaryEdge{Vec2{0.0, area.h}, Vec2{0.0, 0.0}, Vec2{-1.0, 0.0}},
        };

        const BoundaryEdge* nearestEdge = nullptr;
        double nearestIntersectionDistance = std::numeric_limits<double>::max();
        for (const auto& edge : edges)
        {
            const double distance = HalfLineSegmentIntersectionDistance(forward, edge);
            if (distance < nearestIntersectionDistance)
            {
                nearestIntersectionDistance = distance;
                nearestEdge = &edge;
            }
        }

        double distance = nearestIntersectionDistance;
        if (nearestEdge == nullptr || nearestIntersectionDistance == std::numeric_limits<double>::max())
        {
            for (const auto& edge : edges)
            {
                const double edgeDistance = DistanceHalfLineToSegment(forward, edge);
                if (edgeDistance < distance)
                {
                    distance = edgeDistance;
                    nearestEdge = &edge;
                }
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

        const double relativeYaw = toPose.yaw - fromPose.yaw;
        return RelativePose{
            relativePosition,
            relativeYaw
        };
    }
}
