#version 450
// River ribbons: water tint with day-side sun attenuation so rivers fade into
// the night limb with the terrain instead of glowing.

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_sphere;

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 cam_pos_exag;
    vec4 params;
    vec4 sun;
    uvec4 flags;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    float ndl = clamp(dot(v_sphere, pc.sun.xyz), 0.0, 1.0);
    float light = mix(0.25, 1.0, ndl);
    out_color = vec4(v_color.rgb * light, v_color.a);
}
