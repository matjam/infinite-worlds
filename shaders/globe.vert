#version 450
// Globe / Mercator vertex stage.
//
// Per-vertex data is a unit-sphere position plus the id of the cell the corner
// belongs to. All per-cell data lives in storage buffers indexed by that id, so
// a cell-data update never touches the (static) geometry buffers.
//
// Mercator seam handling: every vertex wraps its longitude relative to its own
// cell centre longitude, and the cell centre wraps relative to the view centre
// longitude. A cell is therefore never split across the seam; the seam falls
// *between* cells at the antimeridian of the current view centre.
//
// Shading normals (WP11): each cell carries the gradient of its elevation in
// its own tangent basis (metres per metre, east and north). A corner averages
// the three cells that meet there, exactly as the displacement does, so the
// reconstructed normal is continuous across cell boundaries — smooth relief
// shading under flat per-cell albedo, which is what the Blue Marble looks like.

#include "beauty.glsl"

// x is the cell that owns this corner (and supplies the flat colour); y and z
// are the other cells sharing the corner. Displacement uses the mean of the
// three, so adjacent cells meet instead of leaving open cliffs between them.
layout(location = 0) in vec3 in_pos;
layout(location = 1) in uvec3 in_cells;

struct CellData {
    float elevation_m;
    uint color_rgba8;
    // packHalf2x16(d elevation / d east, d elevation / d north), m per m.
    uint gradient;
    // Byte 0: surface kind (0 land, 1 ocean, 2 lake). Byte 1: ocean depth ramp
    // position. Byte 2: ice fraction. Byte 3: reserved.
    uint material;
};
layout(set = 0, binding = 0, std430) readonly buffer Cells {
    CellData cells[];
};

struct CellStatic {
    float lat_rad;
    float lon_rad;
};
layout(set = 0, binding = 1, std430) readonly buffer Statics {
    CellStatic statics[];
};

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 cam_pos_exag;   // xyz = camera position (km), w = vertical exaggeration
    vec4 params;         // x = radius_km, y = base_offset_m, z = elev_norm_m, w = centre_lon_rad
    vec4 sun;            // xyz = sun direction (unit, world), w = atmosphere fade 0..1
    uvec4 flags;         // x = 0 globe / 1 mercator, y = 1 when beauty shading is on
} pc;

// Fractions of the three corner cells that are LAND (x) and LAKE (y), in
// thirds. Interpolated, the land fraction's 0.5 contour runs mid-way between
// land and water cell centres — a scale-free, sub-cell shoreline for the
// fragment stage to perturb (see the crinkle block in globe.frag); the lake
// fraction colours the water that drowned land fragments turn into.
layout(location = 5) out vec2 v_landlake;

layout(location = 0) out vec3 v_normal;
// Not flat: in the beauty view a corner averages the albedo of the cells that
// meet there *and share its surface kind*, which takes the honeycomb out of
// large uniform areas while leaving coastlines and ice margins crisp. Data
// layers emit the owning cell's colour on all three corners of every triangle,
// so their palettes stay exactly as flat as they were.
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_world;
layout(location = 3) out vec3 v_sphere;
layout(location = 4) flat out vec4 v_mat;

/// Surface kind byte of a cell's material word.
float kind_of(uint material) {
    return floor(unpackUnorm4x8(material).x * 255.0 + 0.5);
}

const float MERCATOR_LAT_LIMIT = 1.48352986; // 85 degrees

float wrap_pi(float a) {
    return a - 2.0 * PI_ * floor((a + PI_) / (2.0 * PI_));
}

void main() {
    CellData cd = cells[in_cells.x];
    CellData cy = cells[in_cells.y];
    CellData cz = cells[in_cells.z];
    float elevation_m = (cd.elevation_m + cy.elevation_m + cz.elevation_m) * (1.0 / 3.0);
    // Mean gradient of the cells meeting at this corner, in each cell's own
    // tangent basis. Neighbouring bases differ by less than a cell's angular
    // size, so averaging the components directly is accurate enough for
    // shading and costs no extra storage.
    vec2 grad = (unpackHalf2x16(cd.gradient)
               + unpackHalf2x16(cy.gradient)
               + unpackHalf2x16(cz.gradient)) * (1.0 / 3.0);

    vec3 n = normalize(in_pos);
    v_sphere = n;
    {
        float ka = kind_of(cd.material);
        float kb = kind_of(cy.material);
        float kc = kind_of(cz.material);
        v_landlake = vec2((ka < 0.5 ? 1.0 : 0.0) + (kb < 0.5 ? 1.0 : 0.0)
                              + (kc < 0.5 ? 1.0 : 0.0),
                          (ka > 1.5 ? 1.0 : 0.0) + (kb > 1.5 ? 1.0 : 0.0)
                              + (kc > 1.5 ? 1.0 : 0.0))
            * (1.0 / 3.0);
    }
    vec4 mat = unpackUnorm4x8(cd.material);
    float kind = floor(mat.x * 255.0 + 0.5);
    v_mat = vec4(kind, mat.y, mat.z, mat.w);

    vec4 albedo = unpackUnorm4x8(cd.color_rgba8);
    if (pc.flags.y != 0u) {
        float w = 1.0;
        if (kind_of(cy.material) == kind) {
            albedo += unpackUnorm4x8(cy.color_rgba8);
            w += 1.0;
        }
        if (kind_of(cz.material) == kind) {
            albedo += unpackUnorm4x8(cz.color_rgba8);
            w += 1.0;
        }
        albedo /= w;
    }
    v_color = albedo;

    float radius_km = pc.params.x;
    float exag = pc.cam_pos_exag.w;
    float disp_km = (elevation_m * exag + pc.params.y) * 0.001;

    // Slope of the displaced surface, clamped so extreme exaggeration stays
    // shadeable.
    vec2 slope = clamp(grad * exag * RELIEF_GAIN, -MAX_SLOPE, MAX_SLOPE);

    if (pc.flags.x == 0u) {
        vec3 east = cross(vec3(0.0, 0.0, 1.0), n);
        east = (dot(east, east) < 1e-12) ? vec3(1.0, 0.0, 0.0) : normalize(east);
        vec3 north = normalize(cross(n, east));
        v_normal = normalize(n - slope.x * east - slope.y * north);
        vec3 world = n * (radius_km + disp_km);
        v_world = world;
        gl_Position = pc.view_proj * vec4(world, 1.0);
    } else {
        float lon = atan(n.y, n.x);
        float lat = asin(clamp(n.z, -1.0, 1.0));
        float cell_lon = statics[in_cells.x].lon_rad;
        float local = wrap_pi(lon - cell_lon);
        float x = wrap_pi(cell_lon - pc.params.w) + local;
        float latc = clamp(lat, -MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT);
        float y = log(tan(0.25 * PI_ + 0.5 * latc));
        vec4 p = pc.view_proj * vec4(x, y, 0.0, 1.0);
        // Orthographic: w == 1, so NDC depth can be written directly. Reverse-Z,
        // so higher terrain gets the larger depth value.
        p.z = 0.5 + 0.25 * clamp(cd.elevation_m / pc.params.z, -1.0, 1.0);
        // Map space: +x east, +y north, +z out of the page.
        v_normal = normalize(vec3(-slope.x, -slope.y, 1.0));
        v_world = vec3(x, y, 0.0);
        gl_Position = p;
    }
}
