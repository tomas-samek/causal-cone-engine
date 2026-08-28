// retina.wgsl — display the observer's receptors.
//
// No marching. The CPU retina already holds, per receptor, everything that
// arrived along entity pipes. This shader thresholds density, shades the
// surface with the arrived normal, and composites over the procedural sky.

// --- Uniforms ---

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    observer_pos: vec3<f32>,
    observer_speed: f32,
    field_size: vec3<f32>,
    tick: f32,
    aabb_min: vec3<f32>,
    _pad1: f32,
    aabb_max: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

// --- Retina textures ---
// dc: (density, r, g, b)  — color is Σ density·color / Σ density
// nd: (nx, ny, nz, depth) — unit normal, density-weighted eye distance
// skin: r — density-weighted fraction of this receptor that is creature

@group(1) @binding(0)
var retina_dc: texture_2d<f32>;
@group(1) @binding(1)
var retina_nd: texture_2d<f32>;
@group(1) @binding(2)
var retina_sampler: sampler;
@group(1) @binding(3)
var retina_skin: texture_2d<f32>;

// keep in sync with src/retina.rs RETINA_ISO
const RETINA_ISO: f32 = 0.3;

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

// --- Procedural noise for reptile skin texture ---

// Hash function: pseudo-random from 3D position
fn hash3(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453);
}

// Voronoi / cellular noise — returns (distance_to_nearest, distance_to_second)
// Creates polygonal cell patterns that look like reptile scales
fn voronoi(p: vec3<f32>) -> vec2<f32> {
    let pi = floor(p);
    let pf = fract(p);

    var d1 = 8.0; // nearest
    var d2 = 8.0; // second nearest

    for (var z = -1i; z <= 1i; z++) {
        for (var y = -1i; y <= 1i; y++) {
            for (var x = -1i; x <= 1i; x++) {
                let offset = vec3<f32>(f32(x), f32(y), f32(z));
                let h = hash3(pi + offset);
                let r = offset + h * 0.85 - pf; // jittered cell center
                let d = dot(r, r);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
    }
    return vec2<f32>(sqrt(d1), sqrt(d2));
}

// Simple 3D value noise
fn value_noise(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    // Smooth interpolation
    let u = pf * pf * (3.0 - 2.0 * pf);

    let n000 = hash3(pi).x;
    let n100 = hash3(pi + vec3<f32>(1.0, 0.0, 0.0)).x;
    let n010 = hash3(pi + vec3<f32>(0.0, 1.0, 0.0)).x;
    let n110 = hash3(pi + vec3<f32>(1.0, 1.0, 0.0)).x;
    let n001 = hash3(pi + vec3<f32>(0.0, 0.0, 1.0)).x;
    let n101 = hash3(pi + vec3<f32>(1.0, 0.0, 1.0)).x;
    let n011 = hash3(pi + vec3<f32>(0.0, 1.0, 1.0)).x;
    let n111 = hash3(pi + vec3<f32>(1.0, 1.0, 1.0)).x;

    let x0 = mix(n000, n100, u.x);
    let x1 = mix(n010, n110, u.x);
    let x2 = mix(n001, n101, u.x);
    let x3 = mix(n011, n111, u.x);
    let y0 = mix(x0, x1, u.y);
    let y1 = mix(x2, x3, u.y);
    return mix(y0, y1, u.z);
}

// Fractal Brownian Motion — multi-octave noise for organic variation
fn fbm(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;
    for (var i = 0u; i < 3u; i++) {
        value += amplitude * value_noise(pos);
        pos *= 2.1;
        amplitude *= 0.5;
    }
    return value;
}

// ACES filmic tone mapping (approximation by Krzysztof Narkowicz)
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
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

    // Sun direction — matches sun placement in spawn_demo_scene
    // Sun is above and slightly forward (+Z), so light comes from upper-forward
    let sun_dir = normalize(vec3<f32>(0.0, 0.8, 0.3));

    // --- Read this pixel's receptor ---
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    let dc = textureSample(retina_dc, retina_sampler, in.uv);
    let density = dc.r;
    if density >= RETINA_ISO {
        let nd = textureSample(retina_nd, retina_sampler, in.uv);
        var norm_color = dc.gba;
        let n_len = length(nd.xyz);
        var normal = select(vec3<f32>(0.0, 1.0, 0.0), nd.xyz / max(n_len, 1e-5), n_len > 1e-4);
        // Creature or scenery, from a real per-source flag rather than a guess
        // at the colour — the floor's lit colour had grown green enough to pass
        // the old greenness test. `skin` is the density-weighted fraction of
        // this receptor that arrived from the dino, so most-of-it wins.
        let skin = textureSample(retina_skin, retina_sampler, in.uv).r;
        let is_creature = select(0.0, 1.0, skin > 0.5);
        let depth = nd.w;
        // Surface point for texture coordinates: along this pixel's ray at the arrived depth
        let sample_pos = u.observer_pos + ray_dir * depth;

        // --- Reptile skin texture ---
        if is_creature > 0.5 {
            // Scale pattern — Voronoi at two frequencies
            // Large scales on body, fine detail overlay
            let scale_pos_large = sample_pos * 0.35;
            let scale_pos_fine = sample_pos * 0.9;
            let vor_large = voronoi(scale_pos_large);
            let vor_fine = voronoi(scale_pos_fine);

            // Scale edge darkening: where cell borders are, darken
            let edge_large = smoothstep(0.0, 0.15, vor_large.y - vor_large.x);
            let edge_fine = smoothstep(0.0, 0.12, vor_fine.y - vor_fine.x);
            let scale_edge = edge_large * 0.7 + edge_fine * 0.3;

            // Scale center bump: cells are slightly raised
            let scale_bump = (1.0 - vor_large.x * 1.2) * 0.6 + (1.0 - vor_fine.x * 1.5) * 0.3;

            // Perturb normal for scale bumps (tangent-space perturbation)
            let eps = 0.8;
            let vor_dx = voronoi(scale_pos_large + vec3<f32>(eps, 0.0, 0.0));
            let vor_dy = voronoi(scale_pos_large + vec3<f32>(0.0, eps, 0.0));
            let vor_dz = voronoi(scale_pos_large + vec3<f32>(0.0, 0.0, eps));
            let bump_grad = vec3<f32>(
                vor_dx.x - vor_large.x,
                vor_dy.x - vor_large.x,
                vor_dz.x - vor_large.x
            ) / eps;
            normal = normalize(normal + bump_grad * 0.8);

            // Color variation — organic mottling
            let mottle = fbm(sample_pos * 0.15);
            let stripe = sin(sample_pos.y * 0.4 + sample_pos.z * 0.15 + mottle * 3.0) * 0.5 + 0.5;

            // Darken in scale grooves, vary hue across body
            norm_color *= mix(0.55, 1.0, scale_edge); // groove darkening
            norm_color *= mix(0.85, 1.15, scale_bump); // raised centers brighter

            // Subtle dorsal stripe pattern (darker along spine/back)
            let dorsal = smoothstep(0.3, 0.7, stripe);
            norm_color = mix(
                norm_color,
                norm_color * vec3<f32>(0.7, 0.85, 0.55), // darker, more olive in stripes
                dorsal * 0.35
            );

            // Warm belly tint (lower Y = more yellow/tan)
            let belly_blend = smoothstep(252.0, 246.0, sample_pos.y);
            norm_color = mix(norm_color, norm_color * vec3<f32>(1.3, 1.15, 0.7), belly_blend * 0.3);
        }

        // Diffuse shading: Lambert (N dot L), clamped with ambient floor
        let n_dot_l = max(dot(normal, sun_dir), 0.0);
        let ambient = 0.10;
        let diffuse = ambient + (1.0 - ambient) * n_dot_l;

        // Rim light: subtle brightening at grazing angles (Fresnel-like)
        let n_dot_v = abs(dot(normal, -ray_dir));
        let rim = pow(1.0 - n_dot_v, 3.0) * 0.3;

        // Specular highlight — reptile skin has a waxy sheen
        let half_vec = normalize(sun_dir - ray_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let specular = pow(n_dot_h, 32.0) * 0.25 * is_creature;

        accumulated_color = norm_color * (diffuse + rim) + vec3<f32>(1.0, 0.95, 0.8) * specular;
        accumulated_alpha = 1.0;
    }

    // Sky gradient background — blue zenith fading to warm horizon
    let sky_up = ray_dir.y; // -1 = down, 0 = horizon, +1 = up
    let horizon_color = vec3<f32>(0.7, 0.6, 0.5);  // warm haze
    let zenith_color = vec3<f32>(0.3, 0.5, 0.8);   // blue sky
    let ground_color = vec3<f32>(0.15, 0.12, 0.1);  // dark ground
    var background: vec3<f32>;
    if sky_up > 0.0 {
        // Sky: blend horizon to zenith
        let t = saturate(sky_up * 2.0); // 0 at horizon, 1 at zenith
        background = mix(horizon_color, zenith_color, t);
    } else {
        // Below horizon: blend horizon to dark ground
        let t = saturate(-sky_up * 3.0);
        background = mix(horizon_color, ground_color, t);
    }
    // Subtle sun glow near sun direction
    let sun_alignment = max(dot(ray_dir, sun_dir), 0.0);
    background += vec3<f32>(1.0, 0.8, 0.4) * pow(sun_alignment, 32.0) * 0.5;

    let final_color = accumulated_color + background * (1.0 - accumulated_alpha);

    // Velocity-dependent vignette — faster observer = darker edges
    // At v=0, full brightness everywhere. At v→c, only center is bright.
    let screen_center = vec2<f32>(0.5, 0.5);
    let dist_from_center = length(in.uv - screen_center) * 2.0; // 0 at center, ~1.4 at corners
    let vignette = 1.0 - u.observer_speed * dist_from_center * 0.8;

    // ACES filmic tone mapping — better contrast and color than Reinhard
    let mapped = aces_tonemap(final_color * max(vignette, 0.1));

    // Gamma correction
    let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, 1.0);
}
