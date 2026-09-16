import moderngl
import numpy as np
from pathlib import Path

from src.sort import sort


QUAD_EXTENT = 3.0


def render(scene):
    camera = scene.camera
    gaussians = scene.gaussian_model

    gaussian_indices = sort(scene) # CPU sort... on python, don't expect good performance

    if len(gaussian_indices) == 0:
        return np.zeros(
            (camera.height, camera.width, 4),
            dtype=np.uint8,
        )

    ctx = moderngl.create_standalone_context(require=410)

    vertex_shader = _load_shader("gaussian_quad.vert")
    fragment_shader = _load_shader("gaussian_quad.frag")

    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
    )

    quad_indices = np.array([
        0, 1, 2,
        0, 2, 3,
    ], dtype=np.uint32)

    index_buffer = ctx.buffer(quad_indices.tobytes())

    vao = ctx.vertex_array(
        program,
        [],
        index_buffer=index_buffer,
        index_element_size=4,
    )

    positions_tex = _create_texture(
        ctx,
        gaussians.positions,
        3,
        "f4",
    )

    scales_tex = _create_texture(
        ctx,
        gaussians.scales,
        3,
        "f4",
    )

    rotations_tex = _create_texture(
        ctx,
        gaussians.rotations,
        4,
        "f4",
    )

    opacities_tex = _create_texture(
        ctx,
        gaussians.opacities.reshape(-1, 1),
        1,
        "f4",
    )

    gaussian_indices_tex = _create_texture(
        ctx,
        gaussian_indices.reshape(-1, 1),
        1,
        "u4",
    )

    sh_tex = _create_texture(
        ctx,
        _pack_sh(gaussians.sh),
        4,
        "f4",
    )

    output_tex = ctx.texture(
        (camera.width, camera.height),
        4,
        dtype="f4",
    )

    framebuffer = ctx.framebuffer(
        color_attachments=[output_tex],
    )

    view = _create_view_matrix(camera)
    projection = _create_projection_matrix(camera)

    focal_y = camera.height / (
        2.0 * np.tan(np.radians(camera.fov_y) * 0.5)
    )
    focal_x = focal_y

    program["view"].write(
        view.T.astype(np.float32).tobytes()
    )
    program["projection"].write(
        projection.T.astype(np.float32).tobytes()
    )

    program["camera_position"].value = tuple(camera.position)
    program["focal"].value = (focal_x, focal_y)
    program["viewport_size"].value = (
        camera.width,
        camera.height,
    )
    program["quad_extent"].value = QUAD_EXTENT

    textures = [
        gaussian_indices_tex,
        positions_tex,
        scales_tex,
        rotations_tex,
        opacities_tex,
        sh_tex,
    ]

    names = [
        "gaussian_indices",
        "positions",
        "scales",
        "rotations",
        "opacities",
        "sh",
    ]

    for unit, (name, texture) in enumerate(zip(names, textures)):
        texture.use(unit)
        program[name].value = unit

    framebuffer.use()
    framebuffer.clear(0.0, 0.0, 0.0, 0.0)

    ctx.viewport = (
        0,
        0,
        camera.width,
        camera.height,
    )

    ctx.enable(moderngl.BLEND)
    ctx.disable(moderngl.DEPTH_TEST)

    ctx.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ONE,
        moderngl.ONE_MINUS_SRC_ALPHA,
    )

    vao.render(
        mode=moderngl.TRIANGLES,
        instances=len(gaussian_indices),
    )

    ctx.finish()

    result = np.frombuffer(
        framebuffer.read(components=4, dtype="f4"),
        dtype=np.float32,
    )

    result = result.reshape((
        camera.height,
        camera.width,
        4,
    )).copy()

    # Convert premultiplied RGB back to RGB i.e
    # RGB = premultiplied_RGB / alpha
    alpha = result[:, :, 3:4]
    result[:, :, :3] = np.divide(
        result[:, :, :3],
        alpha,
        out=np.zeros_like(result[:, :, :3]),
        where=alpha > 0.0,
    )

    result = np.clip(result, 0.0, 1.0)
    result = np.rint(result * 255.0).astype(np.uint8)

    vao.release()
    index_buffer.release()
    framebuffer.release()
    output_tex.release()

    for texture in textures:
        texture.release()

    program.release()
    ctx.release()

    return result


def _load_shader(name):
    project_root = Path(__file__).resolve().parent.parent
    shader_dir = project_root / "shaders"
    shader_path = shader_dir / name
    return shader_path.read_text(encoding="utf-8")


def _create_texture(ctx, data, components, dtype):
    data = np.ascontiguousarray(data)

    # Example of what the code below does:
    # [x0, y0, z0]
    # [x1, y1, z1]
    # [x2, y2, z2]
    # ...
    texels = data.reshape(-1, components)

    max_size = ctx.info["GL_MAX_TEXTURE_SIZE"]

    width = min(len(texels), max_size)
    height = (len(texels) + width - 1) // width

    if height > max_size:
        raise ValueError("Gaussian data exceeds maximum texture size")

    padded = np.zeros(
        (width * height, components),
        dtype=texels.dtype,
    )
    padded[:len(texels)] = texels

    texture = ctx.texture(
        (width, height),
        components,
        padded.tobytes(),
        dtype=dtype,
    )

    texture.filter = (
        moderngl.NEAREST,
        moderngl.NEAREST,
    )

    return texture


def _pack_sh(sh):
    count = len(sh)

    coefficients = np.zeros(
        (count, 3, 16),
        dtype=np.float32,
    )

    if sh.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 SH coefficients, got shape {sh.shape}"
        )

    coefficients[:, :, 0] = sh[:, :3]

    remaining = sh.shape[1] - 3

    if remaining > 0:
        if remaining % 3 != 0:
            raise ValueError(
                f"Invalid SH coefficient count: {sh.shape[1]}"
            )

        degree_coefficients = remaining // 3
        degree_coefficients = min(degree_coefficients, 15)

        rest = sh[:, 3:3 + degree_coefficients * 3]
        rest = rest.reshape(count, 3, degree_coefficients)

        coefficients[:, :, 1:1 + degree_coefficients] = rest

    return coefficients.reshape(-1, 4)


def _create_view_matrix(camera):
    forward = camera.target - camera.position
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, camera.up)
    right /= np.linalg.norm(right)

    camera_up = np.cross(right, forward)

    return np.array([
        [
            right[0],
            right[1],
            right[2],
            -np.dot(right, camera.position),
        ],
        [
            camera_up[0],
            camera_up[1],
            camera_up[2],
            -np.dot(camera_up, camera.position),
        ],
        [
            -forward[0],
            -forward[1],
            -forward[2],
            np.dot(forward, camera.position),
        ],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _create_projection_matrix(camera):
    aspect_ratio = camera.width / camera.height
    f = 1.0 / np.tan(np.radians(camera.fov_y) * 0.5)

    near = camera.near
    far = camera.far

    return np.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [
            0.0,
            0.0,
            (far + near) / (near - far),
            (2.0 * far * near) / (near - far),
        ],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)