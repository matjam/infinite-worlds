#version 450
// Cloud shell vertex stage: a static icosphere at CLOUD_ALTITUDE above the
// surface. Per-vertex cloud *coverage* comes from a storage buffer the CPU
// refills whenever a snapshot arrives (a precipitation-biased, smoothed field
// sampled at each shell vertex); the high-frequency structure is added by the
// fragment stage, so the cloud pattern is independent of the cell resolution.

#include "beauty.glsl"

layout(location = 0) in vec3 in_pos;

layout(set = 0, binding = 2, std430) readonly buffer Coverage {
    float coverage[];
};

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 cam_radius;  // xyz = camera position (km), w = planet radius (km)
    vec4 sun_phase;   // xyz = sun direction (unit, world), w = rotation phase (rad)
    vec4 misc;        // x = opacity fade 0..1, y = noise seed, z/w unused
} pc;

layout(location = 0) out vec3 v_dir;
layout(location = 1) out vec3 v_world;
layout(location = 2) out float v_cov;

void main() {
    vec3 n = normalize(in_pos);
    v_dir = n;
    v_cov = coverage[gl_VertexIndex];
    vec3 world = n * (pc.cam_radius.w * (1.0 + CLOUD_ALTITUDE));
    v_world = world;
    gl_Position = pc.view_proj * vec4(world, 1.0);
}
