import numpy as np


def sort(scene):
    camera = scene.camera
    positions = scene.gaussian_model.positions

    forward = camera.target - camera.position
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, camera.up)  # default up direction for a camera looking along -z
    right /= np.linalg.norm(right)

    camera_up = np.cross(right, forward)  # actual up direction for the current camera orientation

    view_matrix = np.array([
        [right[0],        right[1],        right[2],        -np.dot(right, camera.position)],
        [camera_up[0],    camera_up[1],    camera_up[2],    -np.dot(camera_up, camera.position)],
        [-forward[0],     -forward[1],     -forward[2],      np.dot(forward, camera.position)],
        [0.0,             0.0,             0.0,              1.0],
    ], dtype=np.float32)

    # (x, y, z) -> (x, y, z, 1)
    positions_homogenous = np.column_stack((
        positions,
        np.ones(len(positions), dtype=np.float32),
    ))

    view_positions = positions_homogenous @ view.T

    x = view_positions[:, 0]
    y = view_positions[:, 1]
    depth = -view_positions[:, 2]

    aspect_ratio = camera.width / camera.height
    tan_half_fov = np.tan(np.radians(camera.fov_y) * 0.5)

    half_height = depth * tan_half_fov
    half_width = half_height * aspect_ratio

    visible = (
        (depth >= camera.near)
        & (depth <= camera.far)
        # TODO:
        # Not proper frustum culling as we are not accounting for the size of the gaussian here 
        # but we can come back to this later
        & (np.abs(x) <= half_width)
        & (np.abs(y) <= half_height)
    )

    indices = np.nonzero(visible)[0]
    # indices = indices[np.argsort(depth[indices])[::-1]] # Back to front sorting
    indices = indices[np.argsort(depth[indices])] # TODO: Not sure if blending will work out too well with this approach

    return indices.astype(np.uint32)