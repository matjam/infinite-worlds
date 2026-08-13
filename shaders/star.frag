#version 450
// Procedural, seeded, hash-based starfield. No assets, fully deterministic for
// a given seed. Two grid layers give a mix of bright and faint stars.

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 params; // x = seed, y = brightness, z/w unused
} pc;

layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 o_color;

const vec3 SPACE = vec3(0.004, 0.006, 0.013);

vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

vec3 star_layer(vec3 dir, float scale, float density, float radius, float gain) {
    vec3 p = dir * scale;
    vec3 cell = floor(p);
    vec3 f = p - cell;
    vec3 h = hash33(cell + pc.params.x);
    if (h.x > density) {
        return vec3(0.0);
    }
    vec3 g = hash33(cell * 1.7 + 11.3 + pc.params.x);
    float d = length(f - g);
    float core = smoothstep(radius, 0.0, d);
    float glow = smoothstep(radius * 4.0, 0.0, d) * 0.16;
    // Colour temperature: cool blue-white to warm orange, biased to white.
    float t = h.y;
    vec3 tint = mix(vec3(0.72, 0.80, 1.0), vec3(1.0, 0.85, 0.66), t * t);
    float mag = gain * (0.25 + 0.75 * h.z * h.z);
    return tint * (core + glow) * mag;
}

void main() {
    vec3 dir = normalize(v_dir);
    vec3 col = SPACE;
    col += star_layer(dir, 110.0, 0.055, 0.055, 1.0);
    col += star_layer(dir, 260.0, 0.030, 0.090, 0.45);
    o_color = vec4(col * pc.params.y, 1.0);
}
