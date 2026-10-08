"""Kandinsky 6 Pro MagCache state for ComfyUI's native sampler.

The cache keeps the canonical K6 residual in embedding space.  It supports
both ways ComfyUI evaluates classifier-free guidance: separate cond/uncond
calls and a combined batch containing both branches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


def _nearest(values: Sequence[float], length: int) -> tuple[float, ...]:
    if length < 1:
        raise ValueError("MagCache requires at least one sampling step.")
    if not values:
        raise ValueError("MagCache ratio tables must not be empty.")
    if length == 1:
        return (float(values[-1]),)
    scale = (len(values) - 1) / (length - 1)
    return tuple(float(values[round(index * scale)]) for index in range(length))


def prepare_lane_ratios(
    ratios: Sequence[float], num_steps: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Split canonical interleaved cond/uncond ratios and resize per lane."""
    if len(ratios) < 2:
        raise ValueError("MagCache needs interleaved cond/uncond ratio tables.")
    return _nearest(ratios[0::2], num_steps), _nearest(ratios[1::2], num_steps)


@dataclass
class _LaneState:
    step: int = 0
    accumulated_error: float = 0.0
    accumulated_steps: int = 0
    accumulated_ratio: float = 1.0
    visual_residual: torch.Tensor | None = None
    audio_residual: torch.Tensor | None = None

    def reset_accumulator(self) -> None:
        self.accumulated_error = 0.0
        self.accumulated_steps = 0
        self.accumulated_ratio = 1.0


@dataclass(frozen=True)
class MagCacheDecision:
    lanes: tuple[int, ...]
    chunk_size: int
    skip: bool


class K6MagCacheState:
    """Mutable per-MODEL-clone MagCache controller.

    This deliberately is not a dict: ComfyUI recursively copies nested dicts
    in ``transformer_options`` on every invocation, while this object must keep
    state across denoising calls.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        threshold: float,
        max_skip_steps: int,
        retention_ratio: float,
        mag_ratios: Mapping[str, Sequence[float]],
    ) -> None:
        if num_steps < 1:
            raise ValueError("MagCache num_steps must be positive.")
        if threshold <= 0:
            raise ValueError("MagCache threshold must be positive.")
        if max_skip_steps < 1:
            raise ValueError("MagCache max_skip_steps must be positive.")
        if not 0.0 <= retention_ratio < 1.0:
            raise ValueError("MagCache retention_ratio must be in [0, 1).")

        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.max_skip_steps = int(max_skip_steps)
        self.retention_ratio = float(retention_ratio)
        self._ratios = {
            str(mode): prepare_lane_ratios(values, self.num_steps)
            for mode, values in mag_ratios.items()
        }
        if not {"t2va", "i2va"}.issubset(self._ratios):
            raise ValueError("MagCache requires calibrated t2va and i2va ratios.")

        self.last_generation_stats: dict[str, object] | None = None
        self._generation = 0
        self._profile: str | None = None
        self._shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._lanes: dict[int, _LaneState] = {}
        self._seen_lanes: set[int] = set()
        self._last_timestep: float | None = None
        self._finished = False
        self._computed_forwards = 0
        self._skipped_forwards = 0
        self._computed_model_calls = 0
        self._skipped_model_calls = 0

    def _reset(self, profile: str, visual: torch.Tensor, audio: torch.Tensor) -> None:
        self._generation += 1
        self._profile = profile
        self._shape = (tuple(visual.shape[1:]), tuple(audio.shape[1:]))
        self._lanes = {}
        self._seen_lanes = set()
        self._last_timestep = None
        self._finished = False
        self._computed_forwards = 0
        self._skipped_forwards = 0
        self._computed_model_calls = 0
        self._skipped_model_calls = 0

    def reset_generation(self) -> None:
        """Discard residuals on native pre-run/cleanup, including interrupted runs."""
        self._profile = None
        self._shape = None
        self._lanes = {}
        self._seen_lanes = set()
        self._last_timestep = None
        self._finished = False
        self._computed_forwards = self._skipped_forwards = 0
        self._computed_model_calls = self._skipped_model_calls = 0

    @staticmethod
    def _normalise_lanes(cond_or_uncond: object, batch: int) -> tuple[int, ...]:
        if isinstance(cond_or_uncond, (list, tuple)) and cond_or_uncond:
            lanes = tuple(int(value) for value in cond_or_uncond)
        else:
            lanes = (0,)
        if len(set(lanes)) != len(lanes) or any(lane not in (0, 1) for lane in lanes):
            raise ValueError(f"Unexpected ComfyUI CFG branch layout: {lanes!r}.")
        if batch % len(lanes):
            raise ValueError(
                f"MagCache cannot split batch {batch} across CFG branches {lanes!r}."
            )
        return lanes

    def _lane_wants_skip(self, lane_id: int) -> bool:
        lane = self._lanes.setdefault(lane_id, _LaneState())
        retention_steps = int(self.num_steps * self.retention_ratio)
        if lane.step < retention_steps:
            return False

        ratios = self._ratios[self._profile][lane_id]
        ratio = ratios[min(lane.step, len(ratios) - 1)]
        lane.accumulated_ratio *= ratio
        lane.accumulated_steps += 1
        lane.accumulated_error += abs(1.0 - lane.accumulated_ratio)
        has_residual = lane.visual_residual is not None and lane.audio_residual is not None
        if (
            has_residual
            and lane.accumulated_error < self.threshold
            and lane.accumulated_steps <= self.max_skip_steps
        ):
            return True
        lane.reset_accumulator()
        return False

    def begin(
        self,
        *,
        profile: str,
        timestep: torch.Tensor | Sequence[torch.Tensor],
        cond_or_uncond: object,
        visual: torch.Tensor,
        audio: torch.Tensor,
    ) -> MagCacheDecision:
        if profile not in self._ratios:
            raise ValueError(f"No MagCache calibration for {profile!r}.")
        lanes = self._normalise_lanes(cond_or_uncond, visual.shape[0])
        chunk_size = visual.shape[0] // len(lanes)
        time_tensor = timestep[0] if isinstance(timestep, (list, tuple)) else timestep
        current_timestep = float(time_tensor.detach().flatten()[0].item())
        shape = (tuple(visual.shape[1:]), tuple(audio.shape[1:]))
        new_generation = (
            self._profile != profile
            or self._shape != shape
            or self._finished
            or (
                self._last_timestep is not None
                and current_timestep > self._last_timestep + 1e-6
            )
        )
        if new_generation:
            self._reset(profile, visual, audio)

        self._last_timestep = current_timestep
        self._seen_lanes.update(lanes)
        wants_skip = {lane: self._lane_wants_skip(lane) for lane in lanes}
        skip = all(wants_skip.values())

        # A combined CFG batch cannot cheaply compute only one branch.  If the
        # calibrated decisions disagree, refresh both residuals and restart the
        # accumulator for the branch that would otherwise have skipped.
        if not skip:
            for lane, wanted in wants_skip.items():
                if wanted:
                    self._lanes[lane].reset_accumulator()
        return MagCacheDecision(lanes=lanes, chunk_size=chunk_size, skip=skip)

    def apply_cached(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        decision: MagCacheDecision,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not decision.skip:
            raise RuntimeError("Cannot apply a non-skipping MagCache decision.")
        visual_chunks = visual.split(decision.chunk_size, dim=0)
        audio_chunks = audio.split(decision.chunk_size, dim=0)
        visual_out = []
        audio_out = []
        for index, lane_id in enumerate(decision.lanes):
            lane = self._lanes[lane_id]
            visual_out.append(visual_chunks[index] + lane.visual_residual)
            audio_out.append(audio_chunks[index] + lane.audio_residual)
            lane.step += 1
        self._skipped_forwards += len(decision.lanes)
        self._skipped_model_calls += 1
        self._finish_if_complete()
        return torch.cat(visual_out, dim=0), torch.cat(audio_out, dim=0)

    def record_computed(
        self,
        original_visual: torch.Tensor,
        original_audio: torch.Tensor,
        visual: torch.Tensor,
        audio: torch.Tensor,
        decision: MagCacheDecision,
    ) -> None:
        original_visual_chunks = original_visual.split(decision.chunk_size, dim=0)
        original_audio_chunks = original_audio.split(decision.chunk_size, dim=0)
        visual_chunks = visual.split(decision.chunk_size, dim=0)
        audio_chunks = audio.split(decision.chunk_size, dim=0)
        for index, lane_id in enumerate(decision.lanes):
            lane = self._lanes[lane_id]
            lane.visual_residual = (visual_chunks[index] - original_visual_chunks[index]).detach()
            lane.audio_residual = (audio_chunks[index] - original_audio_chunks[index]).detach()
            lane.step += 1
        self._computed_forwards += len(decision.lanes)
        self._computed_model_calls += 1
        self._finish_if_complete()

    def _finish_if_complete(self) -> None:
        expected = self.num_steps * len(self._seen_lanes)
        total = self._computed_forwards + self._skipped_forwards
        if total < expected:
            return
        self._finished = True
        self.last_generation_stats = {
            "generation": self._generation,
            "profile": self._profile,
            "computed_forwards": self._computed_forwards,
            "skipped_forwards": self._skipped_forwards,
            "total_forwards": total,
            "computed_model_calls": self._computed_model_calls,
            "skipped_model_calls": self._skipped_model_calls,
        }
        # Do not pin roughly a gigabyte of production-size embedding residuals
        # in VRAM while the cached Comfy MODEL remains in the execution cache.
        for lane in self._lanes.values():
            lane.visual_residual = None
            lane.audio_residual = None
        logging.info(
            "Kandinsky 6 MagCache (%s): computed %d, skipped %d of %d logical DiT forwards",
            self._profile,
            self._computed_forwards,
            self._skipped_forwards,
            total,
        )


__all__ = ["K6MagCacheState", "MagCacheDecision", "prepare_lane_ratios"]
