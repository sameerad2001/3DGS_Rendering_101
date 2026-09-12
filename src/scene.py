from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianModel:
    positions: np.ndarray   # (N, 3)
    sh:        np.ndarray   # (N, 48) for SH degree 3
    scales:    np.ndarray   # (N, 3)
    rotations: np.ndarray   # (N, 4)
    opacities: np.ndarray   # (N)


@dataclass
class Camera:
    position: np.ndarray    # (3)
    target:   np.ndarray    # (3)
    up:       np.ndarray    # (3)
    fov_y:    float
    near:     float
    far:      float
    width:    int
    height:   int


@dataclass
class Scene:
    gaussian_model: GaussianModel
    camera: Camera