// field_sample.wgsl — The observer's retina.
//
// A fullscreen triangle. For each pixel, compute a direction from the
// observer into the diff field. March through the field. First deposit
// above threshold = what you see.
//
// This is not raytracing. There are no rays being cast into a scene.
// The observer is moving through a field that already contains all the
// information. Each pixel answers: "what deposit would I hit if I
// extended a line in this direction?"

// --- Uniforms ---

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    observer_pos: vec3<f32>,
    observer_speed: f32,
    field_size: vec3<f32>,
    tick: f32,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

// --- Field texture ---

@group(1) @binding(0)
var field_texture: texture_3d<f32>;
@group(1) @binding(1)
var field_sampler: sampler;

// --- Vertex shader: fullscreen triangle ---
// Generates a triangle that covers the entire screen.
// No vertex buffer needed — positions computed from vertex_index.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Fullscreen triangle trick — 3 vertices cover entire screen
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: 0,0 at top-left to 1,1 at bottom-right
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));

    return out;
}

// --- Fragment shader: field sampling ---

// Sample the field at a world position. Returns density and color.
fn sample_field(world_pos: vec3<f32>) -> vec4<f32> {
    // Convert world position to integer cell coordinates
    let cell = vec3<i32>(world_pos);

    // Out of bounds check
    let size = vec3<i32>(u.field_size);
    if any(cell < vec3<i32>(0)) || any(cell >= size) {
        return vec4<f32>(0.0);
    }

    return textureLoad(field_texture, cell, 0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray direction from screen UV
    // Map UV from [0,1] to clip space [-1,1]
    let clip = vec4<f32>(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,
        1.0,
        1.0
    );

    // Transform clip coords to world space using inverse view-projection
    let world_far = u.inv_view_proj * clip;
    let world_pos = world_far.xyz / world_far.w;

    // Ray direction from observer to this pixel's world point
    let ray_dir = normalize(world_pos - u.observer_pos);

    // --- March through the field ---
    // Logarithmic step sizes: close samples are dense, far samples are sparse.
    // This gives high detail nearby and automatic LOD at distance.
    //
    // The observer hits nearby deposits first (short lag, recent diffs).
    // Far deposits arrive later (long lag, old diffs). Lag IS distance.

    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    // March parameters
    let max_steps = 128u;
    var step_size = 0.5; // start with half-cell steps for precision
    let step_growth = 1.02; // each step slightly larger (logarithmic march)
    let density_threshold = 0.01; // minimum density to register as a hit
    let max_distance = 200.0; // don't march beyond this

    var distance = 1.0; // start 1 cell ahead (not at observer position)

    for (var i = 0u; i < max_steps; i = i + 1u) {
        let sample_pos = u.observer_pos + ray_dir * f32(distance);

        let field_value = sample_field(sample_pos);
        let density = field_value.r; // density is in red channel (first float)

        if density > density_threshold {
            // We hit a deposit. Extract color.
            // In our FieldCell: [density, color_r, color_g, color_b]
            let color = field_value.gba; // RGB in green, blue, alpha channels

            // Normalize color by density to get actual color
            var norm_color = vec3<f32>(0.7, 0.7, 0.7);
            if density > 0.5 {
                norm_color = color / density;
            }

            // Distance-based falloff — closer deposits are brighter
            let falloff = 1.0 / (1.0 + distance * 0.02);

            // Opacity from density — denser deposits are more opaque
            let opacity = min(saturate(density * 0.1) * falloff, 1.0);

            // Front-to-back compositing (like swimming through fog)
            let contribution = norm_color * opacity * (1.0 - accumulated_alpha);
            accumulated_color += contribution;
            accumulated_alpha += opacity * (1.0 - accumulated_alpha);

            // Early exit if fully opaque
            if accumulated_alpha > 0.95 {
                break;
            }
        }

        // Advance ray (logarithmic stepping)
        distance += step_size;
        step_size *= step_growth;

        if distance > max_distance {
            break;
        }
    }

    // Background: very dark blue, simulating empty vacuum
    let background = vec3<f32>(0.02, 0.02, 0.05);
    let final_color = accumulated_color + background * (1.0 - accumulated_alpha);

    // Velocity-dependent vignette — faster observer = darker edges
    // At v=0, full brightness everywhere. At v→c, only center is bright.
    let screen_center = vec2<f32>(0.5, 0.5);
    let dist_from_center = length(in.uv - screen_center) * 2.0; // 0 at center, ~1.4 at corners
    let vignette = 1.0 - u.observer_speed * dist_from_center * 0.8;

    // Tone mapping — simple Reinhard
    let mapped = final_color / (final_color + vec3<f32>(1.0));

    // Gamma correction
    let gamma_corrected = pow(mapped * max(vignette, 0.1), vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
