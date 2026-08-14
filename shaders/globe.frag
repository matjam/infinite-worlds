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
// Interpolated fractions of the cells meeting at each corner:
// x = land fraction (its 0.5 contour is the sub-cell shoreline),
// y = lake fraction (colours the water a drowned fragment turns into).
layout(location = 5) in vec2 v_landlake;
layout(location = 6) in float v_depth;
layout(location = 7) in vec3 v_land_color;

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
    vec3 S = normalize(v_sphere);
    float kind = v_mat.x;
    // Blended across water corners, not the flat per-cell value: shallow
    // shelf cells otherwise step hard against their deep neighbours.
    float depth_t = v_depth;
    float ice_t = v_mat.z;
    bool water = kind > 0.5;

    // --- Shoreline crinkle (sea AND lake shores, globe and Mercator).
    // `v_landlake.x` interpolates the corner cells' land fraction, so its 0.5
    // contour runs mid-way between land and water cell centres — re-drawing
    // the land/water decision at a noise-perturbed offset of that contour
    // dissolves the polygon edges into a wandering shoreline. The noise
    // amplitude is below 0.5 by construction, so open water (landness 0) and
    // deep inland (landness 1) can never flip. A lake fragment only joins in
    // where its landness dips below ~0.55: a lake smaller than a couple of
    // cells never gets there (its corners blend the surrounding land), which
    // is exactly the case where the contour would beach the whole lake.
    bool lake_frag = kind > 1.5;
    // Ocean fragments take the continuous per-fragment bathymetry ramp,
    // detail-modulated by the cell's baked colour so the water keeps its
    // grain. This replaces the flat per-cell ocean colour that stuck out as
    // pale angular plates wherever shallow cells clustered.
    if (water && !lake_frag) {
        vec3 ramp = ocean_ramp(v_depth);
        float lum = dot(albedo, vec3(0.4, 0.7, 0.4));
        float ref_lum = dot(ramp, vec3(0.4, 0.7, 0.4));
        float grain = clamp(lum / max(ref_lum, 1e-4), 0.8, 1.25);
        albedo = ramp * grain;
    }
    if (ice_t < 0.5) {
        float pitch = max(uintBitsToFloat(pc.flags.z), 1e-4);
        float freq = 5.5 / pitch;
        // Landness units per kilometre: the 0..1 landness blend spans about
        // one cell width. Beach and surf are ABSOLUTE widths converted into
        // this space, so they stay a few km at any budget.
        float cell_km = max(pitch * 6371.0, 1.0);
        float sand_band = SAND_WIDTH_KM / cell_km;
        float foam_band = FOAM_WIDTH_KM / cell_km;
        // Two shoreline fields, both land-positive. The SEA field contours
        // interpolated landness at 0.5. LAKE shores need their own field: a
        // small lake's landness never drops below ~2/3, so instead the LAKE
        // fraction (1/3 at a single lake cell's corners, 0 one ring out) is
        // contoured at ~0.2 — which wraps the lake sub-cell and lets its
        // shoreline wander exactly like the sea's, instead of the lake
        // rendering as a flat teal polygon.
        // Sea fragments never take the lake field: an ocean cell that
        // happens to touch a lake must keep its sea shoreline.
        bool lakeside = lake_frag || (v_landlake.y > 0.05 && kind < 0.5);
        float crinkled = lakeside
            ? LAKE_CONTOUR - v_landlake.y + shore_noise(S, freq * 1.7) * LAKE_NOISE_AMP
            : v_landlake.x - 0.5 + shore_noise(S, freq) * SHORE_NOISE_AMP;
        if (crinkled < 0.0 && (!water || lake_frag)) {
            // Water side of the line. Lakes ramp from shore tone to their
            // deep tone. The SEA side uses the same ocean ramp as the true
            // ocean fragments, with the crinkle only pulling the depth to
            // zero at the waterline: `v_depth` is continuous across the cell
            // edge, so the drowned band meets the neighbouring ocean cells
            // seamlessly instead of snapping from its own gradient to
            // theirs at the polygon boundary.
            // Reaches 1.0 by -crinkled ~ 0.15: boundary corners sit near
            // 0.17, so the band converges to the plain ocean ramp BEFORE
            // the polygon edge — that is what makes the seam vanish.
            float deep = smoothstep(0.02, 0.15, -crinkled);
            albedo = lakeside ? mix(LAKE_NEAR_ALBEDO, LAKE_DEEP_ALBEDO, deep)
                              : ocean_ramp(v_depth * deep);
            kind = lakeside ? 2.0 : 1.0;
            water = true;
            depth_t = v_depth * deep;
        } else if (water && crinkled > 0.0) {
            // Emergent fringe on the water side: it is LAND, so it shows the
            // neighbouring land cells' own vegetation/soil albedo — painting
            // it wall-to-wall sand built km-wide beach fields at every coast.
            albedo = to_linear(v_land_color);
            water = false;
        }
        // A narrow beach on the land side of the waterline, then a thin surf
        // line right at it — both a few km, whatever the cell size.
        if (!water && crinkled < sand_band) {
            albedo = mix(SAND_ALBEDO, albedo, smoothstep(0.3, 1.0, crinkled / sand_band));
        }
        float foam = 1.0 - smoothstep(0.0, foam_band, abs(crinkled));
        float surf = (lake_frag || v_landlake.y > 0.2) ? 0.18 : 0.4;
        albedo = mix(albedo, FOAM_ALBEDO, foam * surf);
    }

    // --- Sub-cell relief detail: hills as normal perturbation (land only).
    // Globe only: the Mercator branch's normal lives in map space, where
    // sphere-tangent perturbations would be nonsense.
    if (!mercator && !water && ice_t < 0.5) {
        float fp = length(fwidth(S)) * HILL_FREQ_B;
        float fade = 1.0 - smoothstep(HILL_FADE_FOOTPRINT, HILL_FADE_FOOTPRINT * 2.0, fp);
        if (fade > 0.0) {
            vec3 e1 = normalize(cross(S, abs(S.z) < 0.9 ? vec3(0, 0, 1) : vec3(1, 0, 0)));
            vec3 e2 = cross(S, e1);
            float slope = 1.0 - clamp(dot(N, S), 0.0, 1.0);
            float amp = (HILL_BASE + HILL_SLOPE_GAIN * slope) * fade;
            float e = 1.0 / HILL_FREQ_B;
            float h0 = shore_vnoise(S * HILL_FREQ_A) * 0.7 + shore_vnoise(S * HILL_FREQ_B) * 0.3;
            vec3 px = S + e1 * e;
            vec3 py = S + e2 * e;
            float hx = shore_vnoise(px * HILL_FREQ_A) * 0.7 + shore_vnoise(px * HILL_FREQ_B) * 0.3 - h0;
            float hy = shore_vnoise(py * HILL_FREQ_A) * 0.7 + shore_vnoise(py * HILL_FREQ_B) * 0.3 - h0;
            N = normalize(N + (e1 * hx + e2 * hy) * (amp * 4.0));
        }
    }

    if (mercator) {
        // Flat map: hillshade from a fixed top-left light. No limb, no glint,
        // no clouds — none of them mean anything in a projection.
        float l = clamp(dot(N, MERCATOR_LIGHT), 0.0, 1.0);
        vec3 col = albedo * (MERCATOR_AMBIENT + (1.0 - MERCATOR_AMBIENT) * l);
        o_color = vec4(to_srgb(col), 1.0);
        return;
    }

    vec3 V = normalize(pc.cam_pos_exag.xyz - v_world);

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
