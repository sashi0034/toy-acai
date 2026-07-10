#!/usr/bin/env python3
"""Export a PyTorch policy checkpoint to the small C++ runtime format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import struct

import torch


MAGIC = b"TACAIPOL"
FORMAT_VERSION = 1
HEADER = struct.Struct("<8sIIIIIQ")

PARAMETER_NAMES = (
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
    "action_mean.weight",
    "action_mean.bias",
    "log_std",
    "fire_logit.weight",
    "fire_logit.bias",
)


@dataclass(frozen=True)
class PolicyParameters:
    observation_dim: int
    hidden_dim: int
    action_dim: int
    fire_dim: int
    tensors: tuple[torch.Tensor, ...]

    @property
    def float_count(self) -> int:
        return sum(tensor.numel() for tensor in self.tensors)


def _require_shape(
    state_dict: dict[str, torch.Tensor], name: str, expected: tuple[int, ...]
) -> torch.Tensor:
    tensor = state_dict[name]
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}, expected {expected}"
        )
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return tensor


def load_policy_parameters(checkpoint_path: Path) -> PolicyParameters:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")

    state_dict = checkpoint.get("policy_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain policy_state_dict")

    missing = set(PARAMETER_NAMES) - set(state_dict)
    unexpected = set(state_dict) - set(PARAMETER_NAMES)
    if missing:
        raise ValueError(f"policy_state_dict is missing keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(
            f"policy_state_dict contains unsupported keys: {sorted(unexpected)}"
        )

    first_weight = state_dict["net.0.weight"]
    if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 2:
        raise ValueError("net.0.weight must be a rank-2 tensor")
    hidden_dim, observation_dim = first_weight.shape
    if observation_dim <= 0 or hidden_dim <= 0:
        raise ValueError("policy dimensions must be positive")

    action_weight = state_dict["action_mean.weight"]
    fire_weight = state_dict["fire_logit.weight"]
    if not isinstance(action_weight, torch.Tensor) or action_weight.ndim != 2:
        raise ValueError("action_mean.weight must be a rank-2 tensor")
    if not isinstance(fire_weight, torch.Tensor) or fire_weight.ndim != 2:
        raise ValueError("fire_logit.weight must be a rank-2 tensor")
    action_dim = int(action_weight.shape[0])
    fire_dim = int(fire_weight.shape[0])
    if action_dim != 2 or fire_dim != 1:
        raise ValueError(
            f"only 2 continuous actions and 1 fire logit are supported; got "
            f"{action_dim} and {fire_dim}"
        )

    shapes = {
        "net.0.weight": (hidden_dim, observation_dim),
        "net.0.bias": (hidden_dim,),
        "net.2.weight": (hidden_dim, hidden_dim),
        "net.2.bias": (hidden_dim,),
        "action_mean.weight": (action_dim, hidden_dim),
        "action_mean.bias": (action_dim,),
        "log_std": (action_dim,),
        "fire_logit.weight": (fire_dim, hidden_dim),
        "fire_logit.bias": (fire_dim,),
    }
    tensors = tuple(
        _require_shape(state_dict, name, tuple(int(v) for v in shapes[name]))
        for name in PARAMETER_NAMES
    )
    return PolicyParameters(
        observation_dim=int(observation_dim),
        hidden_dim=int(hidden_dim),
        action_dim=action_dim,
        fire_dim=fire_dim,
        tensors=tensors,
    )


def export_policy(checkpoint_path: Path, output_path: Path) -> PolicyParameters:
    parameters = load_policy_parameters(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    try:
        with temporary_path.open("wb") as output:
            output.write(
                HEADER.pack(
                    MAGIC,
                    FORMAT_VERSION,
                    parameters.observation_dim,
                    parameters.hidden_dim,
                    parameters.action_dim,
                    parameters.fire_dim,
                    parameters.float_count,
                )
            )
            for tensor in parameters.tensors:
                output.write(tensor.numpy().tobytes(order="C"))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export policy_state_dict from a PyTorch checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    parameters = export_policy(args.checkpoint, args.output)
    print(
        f"Exported {args.output}: obs={parameters.observation_dim}, "
        f"hidden={parameters.hidden_dim}, actions={parameters.action_dim}, "
        f"floats={parameters.float_count}"
    )


if __name__ == "__main__":
    main()
