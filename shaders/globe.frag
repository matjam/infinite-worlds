#version 450
// Flat-shaded cell colour with a single directional light. WP11 replaces this
// with the beauty view; this is deliberately the cheapest thing that reads.

layout(location = 0) in vec3 v_normal;
layout(location = 1) flat in vec4 v_color;
layout(location = 2) in vec3 v_world;
layout(location = 3) flat in uint v_mode;

layout(location = 0) out vec4 o_color;

const vec3 SUN_DIR = normalize(vec3(0.45, 0.30, 0.84));

void main() {
    float light = 1.0;
    if (v_mode == 0u) {
        float lambert = max(dot(normalize(v_normal), SUN_DIR), 0.0);
        light = 0.12 + 0.88 * lambert;
    }
    o_color = vec4(v_color.rgb * light, 1.0);
}
