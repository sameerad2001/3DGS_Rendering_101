from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianModel:
    positions: np.ndarray
    sh: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray


@dataclass
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y: float


@dataclass
class Scene:
    gaussian_model: GaussianModel
    camera: Camera