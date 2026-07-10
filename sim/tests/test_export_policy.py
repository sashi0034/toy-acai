from pathlib import Path
import tempfile
import unittest

import torch

from sim.export_policy import (
    FORMAT_VERSION,
    HEADER,
    MAGIC,
    PARAMETER_NAMES,
    export_policy,
    load_policy_parameters,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "windows-gui"
    / "App"
    / "model"
    / "p1783401529686949_6447.pt"
)


class ExportPolicyTests(unittest.TestCase):
    def test_export_real_policy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "policy.bin"
            parameters = export_policy(CHECKPOINT_PATH, output_path)

            self.assertEqual(parameters.observation_dim, 29)
            self.assertEqual(parameters.hidden_dim, 256)
            self.assertEqual(parameters.action_dim, 2)
            self.assertEqual(parameters.fire_dim, 1)
            self.assertEqual(parameters.float_count, 74245)

            data = output_path.read_bytes()
            header = HEADER.unpack_from(data)
            self.assertEqual(
                header, (MAGIC, FORMAT_VERSION, 29, 256, 2, 1, 74245)
            )
            self.assertEqual(len(data), HEADER.size + 74245 * 4)

            checkpoint = torch.load(
                CHECKPOINT_PATH, map_location="cpu", weights_only=True
            )
            offset = HEADER.size
            for name in PARAMETER_NAMES:
                expected = checkpoint["policy_state_dict"][name].float().contiguous()
                byte_count = expected.numel() * 4
                actual = torch.frombuffer(
                    bytearray(data[offset : offset + byte_count]),
                    dtype=torch.float32,
                ).reshape(expected.shape)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                offset += byte_count
            self.assertEqual(offset, len(data))

    def test_rejects_non_finite_parameter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = torch.load(
                CHECKPOINT_PATH, map_location="cpu", weights_only=True
            )
            checkpoint["policy_state_dict"]["net.0.bias"] = checkpoint[
                "policy_state_dict"
            ]["net.0.bias"].clone()
            checkpoint["policy_state_dict"]["net.0.bias"][0] = torch.nan
            path = Path(temporary_directory) / "nan.pt"
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "contains NaN or infinity"):
                load_policy_parameters(path)

    def test_rejects_wrong_layer_shape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = torch.load(
                CHECKPOINT_PATH, map_location="cpu", weights_only=True
            )
            checkpoint["policy_state_dict"]["net.2.bias"] = torch.zeros(255)
            path = Path(temporary_directory) / "wrong-shape.pt"
            torch.save(checkpoint, path)

            with self.assertRaisesRegex(ValueError, "net.2.bias has shape"):
                load_policy_parameters(path)


if __name__ == "__main__":
    unittest.main()
