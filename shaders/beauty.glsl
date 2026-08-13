// Beauty-view artist knobs (WP11, DESIGN.md §9).
//
// This file is the single home for every GPU-side visual constant of the
// beauty view: lighting, ocean glint, atmospheric limb and halo, starfield and
// clouds. It is `#include`d by globe.frag, star.frag and cloud.frag (build.rs
// passes `-I<shaders>` to glslc). The CPU-side knobs — the albedo palette,
// relief-gradient fit and cloud coverage field — live in
// `crates/iw-app/src/beauty.rs`; the sun geometry and altitude fades live in
// `crates/iw-render-vulkan/src/beauty.rs`.
//
// Convention: shading happens in linear light. Cell albedo arrives as sRGB
// bytes, is linearised, lit, and encoded back on the way out (the swapchain is
// UNORM, so the shader owns the transfer function).

#ifndef IW_BEAUTY_GLSL
#define IW_BEAUTY_GLSL

const float PI_ = 3.14159265358979;

// --- transfer functions ----------------------------------------------------

/// sRGB-ish decode. The 2.2 power is close enough to the piecewise curve for
/// albedo work and is a third of the cost.
vec3 to_linear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
/// Linear -> display encode, the inverse of [`to_linear`].
vec3 to_srgb(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); }

// --- global lighting -------------------------------------------------------

/// Sunlight colour (slightly warm white), linear.
const vec3 SUN_COLOR = vec3(1.06, 1.01, 0.95);
/// Ambient floor so the night side is never pure black. Linear.
const vec3 AMBIENT = vec3(0.045, 0.055, 0.075);
/// Extra ambient on the day side, tinted by the sky (Rayleigh bounce).
const vec3 SKY_AMBIENT = vec3(0.10, 0.13, 0.20);
/// Wrap-around term on the diffuse lambert, softening the terminator.
const float DIFFUSE_WRAP = 0.10;
/// Sphere-level day mask: relief may not light the night side. Terminator
/// spans these two values of dot(radial, sun).
const float TERMINATOR_LO = -0.12;
const float TERMINATOR_HI = 0.10;
/// How much of the (exaggerated) elevation gradient enters the shading normal.
/// 1.0 = geometrically exact for the displaced surface.
const float RELIEF_GAIN = 1.0;
/// Slope shading is clamped here so 200x exaggeration does not turn every
/// mountain face into a black or blown-out wall.
const float MAX_SLOPE = 3.0;

// --- water -----------------------------------------------------------------

/// Peak specular addition from the sun glint on open ocean, linear.
const float GLINT_STRENGTH = 0.20;
/// Blinn-Phong exponent of the glint lobe. Large = a small, tight sun disc.
const float GLINT_SHININESS = 2400.0;
/// Warm-white tint of the glint highlight.
const vec3 GLINT_TINT = vec3(1.0, 0.97, 0.90);
/// Lakes get a smaller, softer highlight than the open ocean.
const float LAKE_GLINT_SCALE = 0.55;
const float LAKE_GLINT_SHININESS = 900.0;
/// Deep water reflects a little less of the sky than the shelf; this scales
/// the sky ambient by (1 - DEPTH_AMBIENT_FALLOFF * depth_t).
const float DEPTH_AMBIENT_FALLOFF = 0.45;

// --- atmosphere ------------------------------------------------------------

/// Rayleigh-ish scattering tint of the limb and halo, linear.
const vec3 AIR_COLOR = vec3(0.30, 0.52, 1.00);
/// Fresnel exponent of the in-silhouette rim. Higher = thinner line.
/// 16 keeps the bright part of the line inside the outermost ~1% of the disc.
const float LIMB_POWER = 16.0;
/// Peak brightness of that rim, added in linear light. Deliberately small: the
/// halo pass already fogs the inner limb, and this only has to add the last
/// bright hairline right at the edge.
const float LIMB_STRENGTH = 0.18;
/// Outer halo thickness as a fraction of the planet radius (~190 km on Earth).
const float ATMOS_THICKNESS = 0.022;
/// Peak coverage of the halo where the sight line grazes the limb (the halo
/// pass is alpha blended over the finished globe).
const float HALO_STRENGTH = 0.80;
/// Falloff exponent applied to the normalised chord through the shell.
const float HALO_FALLOFF = 2.6;
/// Day/night wrap of the halo: the night limb keeps this fraction.
const float HALO_NIGHT = 0.06;

// --- starfield -------------------------------------------------------------

/// Deep-space background, linear (the sky pass encodes on the way out, so this
/// is roughly rgb(7, 8, 11) on screen).
const vec3 SPACE_COLOR = vec3(0.0006, 0.0008, 0.0018);
/// Galactic band: pole of the great circle the band is centred on, and its
/// angular half-width in cos units. Density inside is STAR_BAND_GAIN x.
const vec3 GALACTIC_POLE = normalize(vec3(0.42, -0.78, 0.46));
const float STAR_BAND_WIDTH = 0.24;
const float STAR_BAND_GAIN = 3.0;
/// Faint diffuse glow of the band itself.
const float STAR_BAND_GLOW = 0.006;
/// Brightness power law: magnitude = pow(hash, STAR_MAG_POWER). Larger =
/// fewer bright stars and a longer faint tail.
const float STAR_MAG_POWER = 3.5;

// --- clouds ----------------------------------------------------------------

/// Shell altitude as a fraction of the planet radius (~1.5%).
const float CLOUD_ALTITUDE = 0.015;
/// Opacity of a fully saturated cloud.
const float CLOUD_MAX_ALPHA = 0.92;
/// Noise octaves and the lacunarity/gain of the fBm.
const int CLOUD_OCTAVES = 4;
const float CLOUD_LACUNARITY = 2.17;
const float CLOUD_GAIN = 0.55;
/// Base frequency of the first octave, in cycles over the planet radius.
const float CLOUD_FREQ = 5.5;
/// Coverage remaps the noise: density = smoothstep(edge, edge + CLOUD_SOFT, n)
/// with edge = 1 - coverage. Small = crisp cloud edges.
const float CLOUD_SOFT = 0.34;
/// Ambient floor of the cloud lighting (clouds are bright even in shadow).
const float CLOUD_AMBIENT = 0.22;

// --- mercator --------------------------------------------------------------

/// Fixed hillshade light for the flat map: from the top left, fairly high.
const vec3 MERCATOR_LIGHT = normalize(vec3(-0.45, 0.45, 0.77));
/// Ambient floor of the Mercator hillshade (map legibility beats realism).
const float MERCATOR_AMBIENT = 0.62;

#endif
