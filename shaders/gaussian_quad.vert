#version 410 core

uniform mat4 view;
uniform mat4 projection;

uniform vec3 camera_position;
uniform vec2 focal;
uniform vec2 viewport_size;
uniform float quad_extent;

uniform usampler2D gaussian_indices;
uniform sampler2D positions;
uniform sampler2D scales;
uniform sampler2D rotations;
uniform sampler2D opacities;
uniform sampler2D sh;

flat out vec3 gaussian_color;
flat out float gaussian_opacity;

out vec2 local_position;


vec2 quad_vertices[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);


ivec2 linear_coord(sampler2D texture_sampler, int index)
{
    int width = textureSize(texture_sampler, 0).x;

    return ivec2(
        index % width,
        index / width
    );
}


ivec2 linear_coord(usampler2D texture_sampler, int index)
{
    int width = textureSize(texture_sampler, 0).x;

    return ivec2(
        index % width,
        index / width
    );
}


vec4 fetch_float(sampler2D texture_sampler, int index)
{
    return texelFetch(
        texture_sampler,
        linear_coord(texture_sampler, index),
        0
    );
}


uvec4 fetch_uint(usampler2D texture_sampler, int index)
{
    return texelFetch(
        texture_sampler,
        linear_coord(texture_sampler, index),
        0
    );
}


mat3 quat_to_mat3(vec4 q)
{
    q = normalize(q);

    float w = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return mat3(
        vec3(
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy)
        ),
        vec3(
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx)
        ),
        vec3(
            2.0 * (xz + wy),
            2.0 * (yz - wx),
            1.0 - 2.0 * (xx + yy)
        )
    );
}


vec3 evaluate_sh(uint gaussian_index, vec3 direction)
{
    int base = int(gaussian_index) * 12;

    vec4 r0 = fetch_float(sh, base + 0);
    vec4 r1 = fetch_float(sh, base + 1);
    vec4 r2 = fetch_float(sh, base + 2);
    vec4 r3 = fetch_float(sh, base + 3);

    vec4 g0 = fetch_float(sh, base + 4);
    vec4 g1 = fetch_float(sh, base + 5);
    vec4 g2 = fetch_float(sh, base + 6);
    vec4 g3 = fetch_float(sh, base + 7);

    vec4 b0 = fetch_float(sh, base + 8);
    vec4 b1 = fetch_float(sh, base + 9);
    vec4 b2 = fetch_float(sh, base + 10);
    vec4 b3 = fetch_float(sh, base + 11);

    vec3 c[16];

    c[0]  = vec3(r0.x, g0.x, b0.x);
    c[1]  = vec3(r0.y, g0.y, b0.y);
    c[2]  = vec3(r0.z, g0.z, b0.z);
    c[3]  = vec3(r0.w, g0.w, b0.w);

    c[4]  = vec3(r1.x, g1.x, b1.x);
    c[5]  = vec3(r1.y, g1.y, b1.y);
    c[6]  = vec3(r1.z, g1.z, b1.z);
    c[7]  = vec3(r1.w, g1.w, b1.w);

    c[8]  = vec3(r2.x, g2.x, b2.x);
    c[9]  = vec3(r2.y, g2.y, b2.y);
    c[10] = vec3(r2.z, g2.z, b2.z);
    c[11] = vec3(r2.w, g2.w, b2.w);

    c[12] = vec3(r3.x, g3.x, b3.x);
    c[13] = vec3(r3.y, g3.y, b3.y);
    c[14] = vec3(r3.z, g3.z, b3.z);
    c[15] = vec3(r3.w, g3.w, b3.w);

    float x = direction.x;
    float y = direction.y;
    float z = direction.z;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    vec3 color = 0.2820947918 * c[0];

    color += 0.4886025119 * (
        -y * c[1]
        + z * c[2]
        - x * c[3]
    );

    color +=
          1.0925484306 * xy * c[4]
        - 1.0925484306 * yz * c[5]
        + 0.3153915653 * (2.0 * zz - xx - yy) * c[6]
        - 1.0925484306 * xz * c[7]
        + 0.5462742153 * (xx - yy) * c[8];

    color +=
        -0.5900435899 * y * (3.0 * xx - yy) * c[9]
        + 2.8906114426 * xy * z * c[10]
        - 0.4570457995 * y * (4.0 * zz - xx - yy) * c[11]
        + 0.3731763326 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c[12]
        - 0.4570457995 * x * (4.0 * zz - xx - yy) * c[13]
        + 1.4453057213 * z * (xx - yy) * c[14]
        - 0.5900435899 * x * (xx - 3.0 * yy) * c[15];

    return max(color + 0.5, vec3(0.0));
}


void eigen_decomposition(
    mat2 covariance,
    out vec2 axis0,
    out vec2 axis1,
    out float lambda0,
    out float lambda1
)
{
    float a = covariance[0][0];
    float b = covariance[0][1];
    float c = covariance[1][1];

    float trace = a + c;
    float determinant = a * c - b * b;

    float discriminant = sqrt(max(
        0.0,
        0.25 * trace * trace - determinant
    ));

    lambda0 = max(0.5 * trace + discriminant, 1e-8);
    lambda1 = max(0.5 * trace - discriminant, 1e-8);

    if (abs(b) > 1e-6) {
        axis0 = normalize(vec2(lambda0 - c, b));
    } else if (a >= c) {
        axis0 = vec2(1.0, 0.0);
    } else {
        axis0 = vec2(0.0, 1.0);
    }

    axis1 = vec2(-axis0.y, axis0.x);
}


void main()
{
    uint gaussian_index = fetch_uint(gaussian_indices, gl_InstanceID).x;

    vec2 quad = quad_vertices[gl_VertexID];

    vec3 position_ws = fetch_float(positions, int(gaussian_index)).xyz;
    vec3 scale       = exp(fetch_float(scales, int(gaussian_index)).xyz);
    vec4 rotation    = fetch_float(rotations, int(gaussian_index));
    float opacity    = fetch_float(opacities, int(gaussian_index)).x;
    
    vec3 view_direction = normalize(position_ws - camera_position);
    gaussian_color      = evaluate_sh(gaussian_index, view_direction);
    gaussian_opacity    = 1.0 / (1.0 + exp(-opacity));

    vec4 center_pos_vs_homogenous = view * vec4(position_ws, 1.0);
    vec3 center_pos_vs            = center_pos_vs_homogenous.xyz;
    float depth                   = -center_pos_vs.z;

    mat3 R = quat_to_mat3(rotation);
    mat3 S = mat3(
        scale.x, 0.0,     0.0,
        0.0,     scale.y, 0.0,
        0.0,     0.0,     scale.z
    );

    mat3 M = R * S;
    mat3 covariance_ws = M * transpose(M);

    mat3 view_rotation = mat3(view);

    mat3 covariance_vs =
        view_rotation *
        covariance_ws *
        transpose(view_rotation);

    vec3 j0 = vec3(
        focal.x / depth,
        0.0,
        focal.x * center_pos_vs.x / (depth * depth)
    );

    vec3 j1 = vec3(
        0.0,
        focal.y / depth,
        focal.y * center_pos_vs.y / (depth * depth)
    );

    float c00 = dot(j0, covariance_vs * j0);
    float c01 = dot(j0, covariance_vs * j1);
    float c11 = dot(j1, covariance_vs * j1);

    // Expand gaussians, otherwise small gaussians will disappear.
    c00 += 0.3f;
    c11 += 0.3f;

    mat2 covariance_2d = mat2(
        c00, c01,
        c01, c11
    );

    vec2 axis0;
    vec2 axis1;

    float lambda0;
    float lambda1;

    eigen_decomposition(
        covariance_2d,
        axis0,
        axis1,
        lambda0,
        lambda1
    );

    vec2 axis0_px = quad_extent * sqrt(lambda0) * axis0;
    vec2 axis1_px = quad_extent * sqrt(lambda1) * axis1;

    vec2 offset_px =
        quad.x * axis0_px +
        quad.y * axis1_px;
    vec2 offset_ndc  = 2.0 * offset_px / viewport_size;
    vec4 center_clip = projection * center_pos_vs_homogenous;
    center_clip.xy  += offset_ndc * center_clip.w;
    gl_Position = center_clip;

    local_position = quad;
}