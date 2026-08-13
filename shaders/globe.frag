#version 450
// Globe / Mercator fragment stage.
//
// Two looks share this shader:
//   * data layers (flags.y == 0) keep the flat, palette-faithful shading the
//     switchable layers were designed against: one lambert term on the sphere
//     normal, no tone mapping, no atmosphere;
//   * the beauty view (flags.y == 1) lights the relief normal in linear light,
//     adds a sun glint on water, and closes with the atmospheric limb.
// Every constant lives in beauty.glsl.

#include "beauty.glsl"

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_world;
layout(location = 3) in vec3 v_sphere;
// x = surface kind (0 land, 1 ocean, 2 lake), y = ocean depth 0..1,
// z = ice fraction 0..1, w = reserved.
layout(location = 4) flat in vec4 v_mat;

layout(location = 0) out vec4 o_color;

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 cam_pos_exag;
    vec4 params;
    vec4 sun;
    uvec4 flags;
} pc;

void main() {
    bool beauty = pc.flags.y != 0u;
    bool mercator = pc.flags.x == 1u;
    vec3 L = normalize(pc.sun.xyz);

    if (!beauty) {
        // WP10 data layers: unchanged palette, one cheap lambert on the globe.
        float light = 1.0;
        if (!mercator) {
            light = 0.12 + 0.88 * max(dot(normalize(v_sphere), L), 0.0);
        }
        o_color = vec4(v_color.rgb * light, 1.0);
        return;
    }

    vec3 albedo = to_linear(v_color.rgb);
    vec3 N = normalize(v_normal);

    if (mercator) {
        // Flat map: hillshade from a fixed top-left light. No limb, no glint,
        // no clouds — none of them mean anything in a projection.
        float l = clamp(dot(N, MERCATOR_LIGHT), 0.0, 1.0);
        vec3 col = albedo * (MERCATOR_AMBIENT + (1.0 - MERCATOR_AMBIENT) * l);
        o_color = vec4(to_srgb(col), 1.0);
        return;
    }

    vec3 S = normalize(v_sphere);
    vec3 V = normalize(pc.cam_pos_exag.xyz - v_world);
    float kind = v_mat.x;
    float depth_t = v_mat.y;
    float ice_t = v_mat.z;
    bool water = kind > 0.5;

    // Sphere-level day mask: relief may brighten a slope, but never on the
    // night side.
    float day = smoothstep(TERMINATOR_LO, TERMINATOR_HI, dot(S, L));
    float wrapped = (dot(N, L) + DIFFUSE_WRAP) / (1.0 + DIFFUSE_WRAP);
    float diffuse = clamp(wrapped, 0.0, 1.0) * day;

    // Deep water reflects less sky than the shelf, which keeps the abyss dark
    // without crushing the shallows.
    float sky = water ? (1.0 - DEPTH_AMBIENT_FALLOFF * depth_t) : 1.0;
    vec3 light = AMBIENT + SKY_AMBIENT * sky * day + SUN_COLOR * diffuse;
    vec3 col = albedo * light;

    if (water) {
        // Blinn-Phong lobe on the mean water surface (the sphere normal), not
        // on the relief normal: the sea is flat and the bathymetry below it
        // must not carve up the highlight. Ice kills the glint.
        vec3 H = normalize(L + V);
        bool lake = kind > 1.5;
        float shininess = lake ? LAKE_GLINT_SHININESS : GLINT_SHININESS;
        float strength = lake ? GLINT_STRENGTH * LAKE_GLINT_SCALE : GLINT_STRENGTH;
        float spec = pow(max(dot(S, H), 0.0), shininess);
        col += GLINT_TINT * (spec * strength * day * (1.0 - ice_t));
    }

    // Atmospheric limb: a fresnel rim on the silhouette, brightest exactly at
    // the edge. The outer halo is drawn by the sky pass; this is its inside
    // half. pc.sun.w fades both out on descent.
    float rim = pow(1.0 - clamp(dot(S, V), 0.0, 1.0), LIMB_POWER);
    col += AIR_COLOR * (rim * LIMB_STRENGTH * pc.sun.w * mix(HALO_NIGHT, 1.0, day));

    o_color = vec4(to_srgb(col), 1.0);
}
