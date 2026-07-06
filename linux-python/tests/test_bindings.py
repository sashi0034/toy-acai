#!/usr/bin/env python3
"""Smoke and behavior tests for the direct toy_acai_core bindings."""

import importlib
import math
import sys
import unittest


if len(sys.argv) != 2:
    raise SystemExit("usage: test_bindings.py <extension-module-directory>")

sys.path.insert(0, sys.argv.pop(1))
toy_acai_core = importlib.import_module("toy_acai_core")


class ToyAcaiBindingsTest(unittest.TestCase):
    def setUp(self):
        self.context = toy_acai_core.BattlefieldContext()
        toy_acai_core.init_battlefield(self.context)

    def test_context_state_is_mutable_in_place(self):
        for fighter in self.context.fighters:
            fighter.health = 0.0
        self.assertEqual(
            [fighter.health for fighter in self.context.fighters],
            [0.0] * toy_acai_core.FIGHTER_COUNT,
        )

        toy_acai_core.init_battlefield(self.context)
        fighter = self.context.fighters[0]
        fighter.position.x = 120.0
        fighter.position.y = 240.0
        fighter.yaw = 0.0
        fighter.speed = 100.0

        self.assertEqual(self.context.fighters[0].position.x, 120.0)
        self.assertEqual(self.context.fighters[0].position.y, 240.0)
        self.assertEqual(len(list(self.context.fighters)), toy_acai_core.FIGHTER_COUNT)

        inputs = [toy_acai_core.FighterInput(0.0, 0.0, False) for _ in range(toy_acai_core.FIGHTER_COUNT)]
        toy_acai_core.update_battlefield(self.context, inputs, 1.0 / 60.0)
        self.assertGreater(self.context.fighters[0].position.x, 120.0)

        missile = toy_acai_core.MissileState()
        missile.id = 42
        missile.position = toy_acai_core.Vec2(5.0, 10.0)
        self.context.missiles.append(missile)
        self.assertEqual(self.context.missiles[0].id, 42)
        self.context.missiles[0].speed = 75.0
        self.assertEqual(self.context.missiles[0].speed, 75.0)
        self.context.missiles.clear()

        death_event = toy_acai_core.DeathEvent()
        death_event.reason = toy_acai_core.DeathEvent.Reason.HitByMissile
        death_event.fighter_index = 5
        death_event.killer_missile = missile
        self.context.death_events.append(death_event)
        self.context.death_events[0].fighter_index = 6
        self.assertEqual(self.context.death_events[0].fighter_index, 6)

    def test_input_count_and_fighter_index_are_checked(self):
        with self.assertRaises(ValueError):
            toy_acai_core.update_battlefield(self.context, [toy_acai_core.FighterInput()], 1.0 / 60.0)

        with self.assertRaises(IndexError):
            toy_acai_core.compute_distance_from_boundary(self.context, toy_acai_core.FIGHTER_COUNT)

    def test_battlefield_utils(self):
        origin = toy_acai_core.Vec2()
        point = toy_acai_core.Vec2(3.0, 4.0)
        self.assertEqual(point.length(), 5.0)
        self.assertEqual(point.length_sq(), 25.0)
        self.assertEqual(point.distance_sq(), 25.0)
        self.assertEqual(origin.distance_from(point), 5.0)
        self.assertEqual(origin.distance_from_sq(point), 25.0)
        self.assertEqual(point.dot(point), 25.0)
        self.assertEqual(origin.cross(point), 0.0)
        self.assertEqual(point.normalized().length(), 1.0)
        self.assertTrue(math.isclose(point.rotated(math.pi).x, -3.0))
        self.assertTrue(math.isclose(point.angle(), math.atan2(3.0, -4.0)))
        self.assertTrue(
            math.isclose(
                point.angle_to(toy_acai_core.Vec2(4.0, 3.0)),
                math.atan2(-7.0, 24.0),
            )
        )

        self.assertTrue(math.isclose(
            self.context.battlefield_diagonal_length,
            math.hypot(
                self.context.battlefield_area.w,
                self.context.battlefield_area.h,
            ),
        ))

        boundary = toy_acai_core.compute_distance_from_boundary(self.context, 0)
        self.assertGreater(boundary.distance, 0.0)

        from_pose = toy_acai_core.AbsolutePose(self.context.fighters[0])
        to_pose = toy_acai_core.AbsolutePose(self.context.fighters[1])
        relative = toy_acai_core.compute_relative_pose(from_pose, to_pose)
        self.assertIsInstance(relative.relative_position, toy_acai_core.Vec2)
        self.assertIsInstance(relative.relative_bearing, float)

    def test_headless_renderer_returns_rgba_frame(self):
        renderer = toy_acai_core.BattlefieldRenderer()
        renderer.enable_render_to_image_buffer(toy_acai_core.Size(64, 36))
        renderer.update(self.context, 1.0 / 60.0)
        renderer.render(self.context)

        frame = renderer.image_buffer()
        self.assertEqual(frame.shape, (36, 64, 4))
        self.assertEqual(frame.dtype.name, "uint8")


if __name__ == "__main__":
    unittest.main()
