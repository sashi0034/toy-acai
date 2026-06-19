#!/usr/bin/env python3
"""Smoke and behavior tests for the direct toy_acai_core bindings."""

import importlib
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

        hit_event = toy_acai_core.HitEvent()
        hit_event.shooter_fighter_index = 1
        hit_event.target_fighter_index = 5
        self.context.hit_events.append(hit_event)
        self.context.hit_events[0].target_fighter_index = 6
        self.assertEqual(self.context.hit_events[0].target_fighter_index, 6)

    def test_input_count_and_fighter_index_are_checked(self):
        with self.assertRaises(ValueError):
            toy_acai_core.update_battlefield(self.context, [toy_acai_core.FighterInput()], 1.0 / 60.0)

        with self.assertRaises(IndexError):
            toy_acai_core.compute_forward_distance_from_boundary(self.context, toy_acai_core.FIGHTER_COUNT)

    def test_battlefield_utils(self):
        boundary = toy_acai_core.compute_forward_distance_from_boundary(self.context, 0)
        self.assertGreater(boundary.distance, 0.0)

        from_pose = toy_acai_core.AbsolutePose(self.context.fighters[0])
        to_pose = toy_acai_core.AbsolutePose(self.context.fighters[1])
        relative = toy_acai_core.compute_relative_pose(from_pose, to_pose)
        self.assertIsInstance(relative.relative_position, toy_acai_core.Vec2)
        self.assertIsInstance(relative.relative_yaw, float)

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
