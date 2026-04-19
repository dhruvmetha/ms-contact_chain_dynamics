"""Interfaces for GT and image-based scene providers/mappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from taskbench.planners.stickpush_rh.types import PlannerAction, SceneState


@dataclass
class CameraContext:
    """Camera metadata required for image/world conversions."""

    intrinsics: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    meta: dict[str, Any] | None = None


class SceneStateProvider(ABC):
    """Provides planner-ready scene state from an environment."""

    @abstractmethod
    def get_scene_state(self, env) -> SceneState:
        raise NotImplementedError


class CoordinateMapper(ABC):
    """Maps points between image and world coordinates."""

    @abstractmethod
    def pixel_to_world(self, pixels: np.ndarray, camera_ctx: CameraContext) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def world_to_pixel(self, points: np.ndarray, camera_ctx: CameraContext) -> np.ndarray:
        raise NotImplementedError


class ActionImageProjector(ABC):
    """Projects a planner action into image space for overlays/models."""

    @abstractmethod
    def action_to_image_overlay(self, action: PlannerAction, camera_ctx: CameraContext) -> dict:
        raise NotImplementedError


class IdentityCoordinateMapper(CoordinateMapper):
    """Stub mapper for GT-only integration; returns inputs unchanged."""

    def pixel_to_world(self, pixels: np.ndarray, camera_ctx: CameraContext) -> np.ndarray:
        return np.asarray(pixels, dtype=np.float32)

    def world_to_pixel(self, points: np.ndarray, camera_ctx: CameraContext) -> np.ndarray:
        return np.asarray(points, dtype=np.float32)


class NullActionImageProjector(ActionImageProjector):
    """Stub projector used until image-based integration is added."""

    def action_to_image_overlay(self, action: PlannerAction, camera_ctx: CameraContext) -> dict:
        return {}

