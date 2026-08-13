#version 450
// Cloud shell fragment stage: fBm value noise on the sphere, thresholded by the
// interpolated coverage field, lit by the same sun as the globe. The shell
// rotates slowly relative to the surface (the phase is a push constant) so the
// planet has some life; nothing here casts a shadow.

#include "beauty.glsl"

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 cam_radius;
    vec4 sun_phase;
    vec4 misc;
} pc;

layout(location = 0) in vec3 v_dir;
layout(location = 1) in vec3 v_world;
layout(location = 2) in float v_cov;

layout(location = 0) out vec4 o_color;

float hash13(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float value_noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i + vec3(0, 0, 0));
    float n100 = hash13(i + vec3(1, 0, 0));
    float n010 = hash13(i + vec3(0, 1, 0));
    float n110 = hash13(i + vec3(1, 1, 0));
    float n001 = hash13(i + vec3(0, 0, 1));
    float n101 = hash13(i + vec3(1, 0, 1));
    float n011 = hash13(i + vec3(0, 1, 1));
    float n111 = hash13(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

/// Normalised fBm in 0..1.
float fbm(vec3 p) {
    float sum = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    for (int i = 0; i < CLOUD_OCTAVES; ++i) {
        sum += amp * value_noise(p);
        norm += amp;
        amp *= CLOUD_GAIN;
        p *= CLOUD_LACUNARITY;
    }
    return sum / norm;
}

void main() {
    vec3 n = normalize(v_dir);
    // Rotate the sampling frame about the pole: the field is static in its own
    // frame, so this is a rigid drift of the whole cloud deck.
    float ph = pc.sun_phase.w;
    float c = cos(ph);
    float s = sin(ph);
    vec3 q = vec3(c * n.x - s * n.y, s * n.x + c * n.y, n.z);

    float cov = clamp(v_cov, 0.0, 1.0);
    float noise = fbm(q * CLOUD_FREQ + pc.misc.y);
    // Coverage sets the threshold: at coverage 1 everything passes, at 0
    // nothing does.
    float edge = 1.0 - cov;
    float density = smoothstep(edge, edge + CLOUD_SOFT, noise);
    float alpha = density * CLOUD_MAX_ALPHA * clamp(pc.misc.x, 0.0, 1.0);
    if (alpha <= 0.002) {
        discard;
    }

    vec3 L = normalize(pc.sun_phase.xyz);
    float day = smoothstep(TERMINATOR_LO, TERMINATOR_HI, dot(n, L));
    float diffuse = clamp((dot(n, L) + DIFFUSE_WRAP) / (1.0 + DIFFUSE_WRAP), 0.0, 1.0) * day;
    vec3 lit = vec3(CLOUD_AMBIENT) + SUN_COLOR * diffuse;
    o_color = vec4(to_srgb(lit), alpha);
}
