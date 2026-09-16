import json
import numpy as np
from pathlib import Path
from PIL import Image

from src.load_scene import load_scene
from src.render import render
from src.scene import Camera


# Forward and up directions
VIEWS = {
    "front": (
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ),
    "back": (
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ),
    "left": (
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ),
    "right": (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ),
    "top": (
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    ),
    "bottom": (
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ),
}


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_camera(config):
    return Camera(
        position=np.array(config["position"], dtype=np.float32),
        target=np.array(config["target"], dtype=np.float32),
        up=np.array(config["up"], dtype=np.float32),
        fov_y=float(config["fov_y"]),
        near=float(config["near"]),
        far=float(config["far"]),
        width=int(config["width"]),
        height=int(config["height"]),
    )


def setup_view_camera(scene, forward, up):
    positions = scene.gaussian_model.positions
    camera = scene.camera

    bounds_min = np.min(positions, axis=0)
    bounds_max = np.max(positions, axis=0)

    center = (bounds_min + bounds_max) * 0.5
    size = bounds_max - bounds_min

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)

    half_size = size * 0.5

    half_width = np.dot(np.abs(right), half_size)
    half_height = np.dot(np.abs(camera_up), half_size)
    half_depth = np.dot(np.abs(forward), half_size)

    aspect_ratio = camera.width / camera.height
    half_fov_y = np.radians(camera.fov_y) * 0.5

    required_half_size = max(
        half_height,
        half_width / aspect_ratio,
    )

    distance = required_half_size / np.tan(half_fov_y)
    distance *= 1.25

    camera.position = (
        center + (-forward) * (half_depth + distance)
    ).astype(np.float32)

    camera.target = center.astype(np.float32)
    camera.up = up.copy()

    scene_radius = np.linalg.norm(size) * 0.5
    scene_depth = half_depth * 2.0

    camera.near = max(
        0.001,
        distance - scene_radius * 2.0,
    )

    camera.far = (
        distance
        + scene_depth
        + scene_radius * 2.0
    )


def save_image(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data, mode="RGBA").save(path)


def main():
    config = load_config()
    
    scene = load_scene(config["input_path"])
    scene.camera = create_camera(config["camera"])
    
    output_path = Path(config["output_path"])

    for name, (direction, up) in VIEWS.items():
        setup_view_camera(scene, direction, up)

        # TODO: render() initializes ogl and setups the pipeline from scratch every time...
        # not ideal for performance, will fix later ... maybe
        pixels = render(scene)

        view_output_path = output_path.with_name(
            f"{output_path.stem}_{name}{output_path.suffix}"
        )

        save_image(view_output_path, pixels)
        print(f"Saved {name}: {view_output_path}")


if __name__ == "__main__":
    main()
