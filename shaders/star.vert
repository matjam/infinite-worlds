#version 450
// Fullscreen-triangle sky background (starfield + atmospheric halo). Emits the
// per-pixel view ray by unprojecting two points along it; drawn first with
// depth writes disabled, so the globe covers whatever it occludes.

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 params; // x = seed, y = brightness, z = planet radius km, w unused
    vec4 cam;    // xyz = camera position (km), w = halo strength 0..1
    vec4 sun;    // xyz = sun direction (unit, world), w unused
} pc;

layout(location = 0) out vec3 v_dir;

void main() {
    vec2 uv = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    vec2 ndc = uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0); // reverse-Z: 0.0 is the far plane

    // Reverse-Z: depth 1.0 is the near plane. Take a second, further sample to
    // get the ray direction without relying on the (infinite) far plane.
    vec4 a = pc.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec4 b = pc.inv_view_proj * vec4(ndc, 0.25, 1.0);
    v_dir = b.xyz / b.w - a.xyz / a.w;
}
