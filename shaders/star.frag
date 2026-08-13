#version 450
// The sky pass: a procedural, seeded starfield plus the planet's outer
// atmospheric halo. No assets, no time input — the field is static (real stars
// do not twinkle in vacuum, and a twinkling background reads as noise), and
// fully deterministic for a given seed.
//
// Structure of the star field:
//   * two grid layers, a sparse bright one and a dense faint one;
//   * a galactic band (a great circle) where density is STAR_BAND_GAIN higher
//     and a faint milky glow is added;
//   * a power-law magnitude distribution — many faint stars, few bright ones;
//   * colour temperature from blue-white to warm orange.
//
// The same shader runs twice per frame (params.w selects which):
//   * mode 0, before the globe, opaque: space and stars;
//   * mode 1, after the globe, alpha blended: the atmospheric halo.
// The halo is an analytic shell integral — the length of the sight line inside
// the atmosphere shell and in front of the planet, normalised by the grazing
// chord. Running it last means it fogs the globe near the limb and wraps the
// exaggerated peaks that stick out past the silhouette, instead of being
// painted over by them.

#include "beauty.glsl"

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 params; // x = seed, y = brightness, z = planet radius km, w = mode
    vec4 cam;    // xyz = camera position (km), w = halo strength 0..1
    vec4 sun;    // xyz = sun direction (unit, world), w unused
} pc;

layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 o_color;

vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

/// Smooth (trilinear) value noise in 0..1. Used for the band's dust lanes:
/// sampling the hash directly would tile the sky with visible boxes.
float value_noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = x - i;
    f = f * f * (3.0 - 2.0 * f);
    float n00 = mix(hash33(i + vec3(0, 0, 0)).x, hash33(i + vec3(1, 0, 0)).x, f.x);
    float n10 = mix(hash33(i + vec3(0, 1, 0)).x, hash33(i + vec3(1, 1, 0)).x, f.x);
    float n01 = mix(hash33(i + vec3(0, 0, 1)).x, hash33(i + vec3(1, 0, 1)).x, f.x);
    float n11 = mix(hash33(i + vec3(0, 1, 1)).x, hash33(i + vec3(1, 1, 1)).x, f.x);
    return mix(mix(n00, n10, f.y), mix(n01, n11, f.y), f.z);
}

/// One grid layer of stars. `density` is the fraction of grid cells holding a
/// star before the galactic-band boost; `gain` scales the whole layer.
vec3 star_layer(vec3 dir, float scale, float density, float radius, float gain) {
    vec3 p = dir * scale;
    vec3 cell = floor(p);
    vec3 f = p - cell;
    vec3 h = hash33(cell + pc.params.x);
    if (h.x > density) {
        return vec3(0.0);
    }
    // Keep the star away from the cell border so its glow is not clipped by
    // the (single-cell) lookup.
    vec3 g = hash33(cell * 1.7 + 11.3 + pc.params.x) * 0.6 + 0.2;
    float d = length(f - g);
    float core = smoothstep(radius, 0.0, d);
    float glow = smoothstep(radius * 3.0, 0.0, d) * 0.14;
    // Colour temperature: cool blue-white to warm orange, biased to white.
    float t = h.y;
    vec3 tint = mix(vec3(0.70, 0.79, 1.0), vec3(1.0, 0.82, 0.62), t * t);
    // Power law: pow(u, k) with k > 1 pushes most stars faint.
    float mag = gain * (0.03 + 0.97 * pow(h.z, STAR_MAG_POWER));
    return tint * (core + glow) * mag;
}

/// Nearest-intersection parameters of a ray with a sphere of radius `r`
/// centred on the origin. Returns false when the ray misses.
bool sphere_hit(vec3 o, vec3 d, float r, out float t0, out float t1) {
    float b = dot(o, d);
    float c = dot(o, o) - r * r;
    float disc = b * b - c;
    if (disc <= 0.0) {
        return false;
    }
    float s = sqrt(disc);
    t0 = -b - s;
    t1 = -b + s;
    return true;
}

/// Scattering along this sight line: `rgb` is the (linear) haze colour, `a` its
/// coverage. Alpha blending it approximates in-scatter plus extinction, and
/// unlike an additive term it cannot blow out the disc it is drawn over.
vec4 halo(vec3 dir) {
    float strength = pc.cam.w;
    float radius = pc.params.z;
    if (strength <= 0.0 || radius <= 0.0) {
        return vec4(0.0);
    }
    float outer = radius * (1.0 + ATMOS_THICKNESS);
    float a0, a1;
    if (!sphere_hit(pc.cam.xyz, dir, outer, a0, a1) || a1 <= 0.0) {
        return vec4(0.0);
    }
    a0 = max(a0, 0.0);
    // Only the part of the shell in front of the planet scatters towards us.
    float p0, p1;
    if (sphere_hit(pc.cam.xyz, dir, radius, p0, p1) && p0 > 0.0) {
        a1 = min(a1, p0);
    }
    float chord = max(a1 - a0, 0.0);
    float grazing = 2.0 * sqrt(max(outer * outer - radius * radius, 1e-6));
    float density = pow(clamp(chord / grazing, 0.0, 1.0), HALO_FALLOFF);
    // Brighter where the shell is in sunlight.
    vec3 mid = pc.cam.xyz + dir * (a0 + chord * 0.5);
    float day = smoothstep(TERMINATOR_LO, TERMINATOR_HI, dot(normalize(mid), pc.sun.xyz));
    return vec4(AIR_COLOR * mix(HALO_NIGHT, 1.0, day),
                clamp(density * HALO_STRENGTH * strength, 0.0, 1.0));
}

void main() {
    vec3 dir = normalize(v_dir);
    if (pc.params.w > 0.5) {
        vec4 h = halo(dir);
        o_color = vec4(to_srgb(h.rgb), h.a);
        return;
    }
    vec3 col = SPACE_COLOR;

    // Galactic band: 1 on the band's great circle, 0 outside its half-width.
    float band = smoothstep(STAR_BAND_WIDTH, 0.0, abs(dot(dir, GALACTIC_POLE)));
    float density = 1.0 + (STAR_BAND_GAIN - 1.0) * band;
    // Unresolved starlight in the band, mottled at two scales so it is neither
    // a clean stripe nor a grid of boxes.
    float dust = 0.45 * value_noise(dir * 7.0 + pc.params.x)
               + 0.30 * value_noise(dir * 19.0 + pc.params.x) + 0.35;
    col += vec3(0.85, 0.88, 1.0) * (STAR_BAND_GLOW * band * band * dust);

    col += star_layer(dir, 110.0, 0.055 * density, 0.055, 1.0);
    col += star_layer(dir, 260.0, 0.030 * density, 0.085, 0.42);
    col *= pc.params.y;

    o_color = vec4(to_srgb(col), 1.0);
}
