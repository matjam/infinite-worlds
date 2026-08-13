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

// x is the cell that owns this corner (and supplies the flat colour); y and z
// are the other cells sharing the corner. Displacement uses the mean of the
// three, so adjacent cells meet instead of leaving open cliffs between them.
layout(location = 0) in vec3 in_pos;
layout(location = 1) in uvec3 in_cells;

struct CellData {
    float elevation_m;
    uint color_rgba8;
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
    uvec4 flags;         // x = 0 globe / 1 mercator
} pc;

layout(location = 0) out vec3 v_normal;
layout(location = 1) flat out vec4 v_color;
layout(location = 2) out vec3 v_world;
layout(location = 3) flat out uint v_mode;

const float PI = 3.14159265358979;
const float MERCATOR_LAT_LIMIT = 1.48352986; // 85 degrees

float wrap_pi(float a) {
    return a - 2.0 * PI * floor((a + PI) / (2.0 * PI));
}

void main() {
    CellData cd = cells[in_cells.x];
    float elevation_m = (cd.elevation_m
                       + cells[in_cells.y].elevation_m
                       + cells[in_cells.z].elevation_m) * (1.0 / 3.0);
    vec3 n = normalize(in_pos);
    v_normal = n;
    v_color = unpackUnorm4x8(cd.color_rgba8);
    v_mode = pc.flags.x;

    float radius_km = pc.params.x;
    float disp_km = (elevation_m * pc.cam_pos_exag.w + pc.params.y) * 0.001;

    if (pc.flags.x == 0u) {
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
        float y = log(tan(0.25 * PI + 0.5 * latc));
        vec4 p = pc.view_proj * vec4(x, y, 0.0, 1.0);
        // Orthographic: w == 1, so NDC depth can be written directly. Reverse-Z,
        // so higher terrain gets the larger depth value.
        p.z = 0.5 + 0.25 * clamp(cd.elevation_m / pc.params.z, -1.0, 1.0);
        v_world = vec3(x, y, 0.0);
        gl_Position = p;
    }
}
