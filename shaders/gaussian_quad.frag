#version 410 core

uniform float quad_extent;

flat in vec3 gaussian_color;
flat in float gaussian_opacity;

in vec2 local_position;

out vec4 frag_color;


void main()
{
    float radius_squared = dot(local_position, local_position);

    if (radius_squared > 1.0)
        discard;

    float gaussian_radius_squared = radius_squared * quad_extent * quad_extent;
    float gaussian_weight         = exp(-0.5 * gaussian_radius_squared);
    float alpha                   = gaussian_opacity * gaussian_weight;

    if (alpha < 1.0 / 255.0)
        discard;

    frag_color = vec4(
        gaussian_color,
        alpha
    );
}