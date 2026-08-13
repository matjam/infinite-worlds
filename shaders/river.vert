#version 450
// River ribbons: CPU-built quads hugging the displaced terrain, one strip per
// drainage edge. Vertices carry their own elevation (already blended with the
// banks) and colour (flux-scaled alpha), so the stage is just the globe
// branch's displacement with no cell lookups.

layout(location = 0) in vec3 in_pos;         // unit sphere direction
layout(location = 1) in float in_elevation_m;
layout(location = 2) in vec4 in_color;

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 cam_pos_exag;   // xyz = camera position (km), w = vertical exaggeration
    vec4 params;         // x = radius_km, y = base_offset_m (already lifted)
    vec4 sun;            // xyz = sun direction
    uvec4 flags;
} pc;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec3 v_sphere;

void main() {
    vec3 n = normalize(in_pos);
    v_sphere = n;
    v_color = in_color;
    float disp_km = (in_elevation_m * pc.cam_pos_exag.w + pc.params.y) * 0.001;
    vec3 world = n * (pc.params.x + disp_km);
    gl_Position = pc.view_proj * vec4(world, 1.0);
}
