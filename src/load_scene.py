import numpy as np
from plyfile import PlyData

from src.scene import GaussianModel, Scene


def load_scene(file):
    ply = PlyData.read(file)
    vertex = ply["vertex"].data

    positions = np.column_stack((
        vertex["x"],
        vertex["y"],
        vertex["z"],
    ))

    scales = np.column_stack((
        vertex["scale_0"],
        vertex["scale_1"],
        vertex["scale_2"],
    ))

    rotations = np.column_stack((
        vertex["rot_0"],
        vertex["rot_1"],
        vertex["rot_2"],
        vertex["rot_3"],
    ))

    opacities = np.asarray(vertex["opacity"])

    sh_names = [
        name for name in vertex.dtype.names
        if name.startswith("f_dc_") or name.startswith("f_rest_")
    ]
    sh = np.column_stack([vertex[name] for name in sh_names])

    gaussian_model = GaussianModel(
        positions=positions,
        sh=sh,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
    )

    return Scene(
        gaussian_model=gaussian_model,
        camera=None,
    )