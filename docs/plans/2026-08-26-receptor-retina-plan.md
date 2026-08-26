# Receptor Retina Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 512³ `FieldCell` grid with a persistent receptor array on the image plane that entities feed through delta-carrying pipes, with occlusion given only by pipe transmittance.

**Architecture:** A new `src/retina.rs` owns receptors, entity→receptor pipes (SoA, per-entity slices), per-entity transmittance τ toward the eye, and a delta-only arrival pass. `field.rs` keeps entity transport (Phases 1–2) and gains a `Source` builder that replaces the grid deposit; the same `segment_transmittance` also attenuates long radiation edges (the dino's floor shadow), which lets `ray_blocked` go. The renderer uploads two small 2D textures and the new shader only thresholds, shades, and composites. The grid path stays compilable (dead) until the human visual check, then is deleted.

**Tech Stack:** Rust 2021, glam 0.29, rayon 1.10, wgpu 23, half 2, WGSL. No new dependencies.

**Spec:** `docs/plans/2026-08-26-receptor-retina-design.md`

## Global Constraints

- `RETINA_W = 320`, `RETINA_H = 180`, `RETINA_ISO = 0.3`, `DELTA_EPS = 1e-4`, `RELINK_SHIFT = 0.5` receptors, `ATTEN_END_MARGIN = 1.5` cells, `ATTEN_SAMPLES_PER_CELL = 1.0`, `ATTEN_THRESHOLD = 0.6`, `ATTEN_K_DEFAULT = 0.5`.
- Tuning values (`tune_density`, `tune_color`, `atten_k`) clamp to [1/1024, 1024] via `scale_tune`; every tuning keypress logs all current values.
- Receptors never decay and are never cleared except inside `relink` (drop → zero → rebuild). Every arrival is `new − pipe_last`.
- Occlusion is transmittance only. No depth test anywhere. `Receptor.depth` feeds shader cosmetics only.
- Vacuum entities never get retina pipes. Sky and sun glow stay procedural.
- Keys: `H` stats, `1`/`2` density, `3`/`4` color, `5`/`6` atten_k, `7`/`8` receptor resolution. `T`, `I`, `[`, `]`, `-`, `=` untouched.
- Deviation from the spec's sketch (recorded here so nobody "fixes" it back): `segment_transmittance` takes `&[Source]` + `&SpatialHash` instead of `&DiffField`, so `retina.rs` is testable without the 2 GB `DiffField::new()`; `skip` is `&[usize]` so both endpoints of an edge can be excluded. `Source` also carries `occluder` (frustum-independent) separately from `drawable`.
- Task 7 is a HUMAN CHECKPOINT. Task 8 (grid deletion) must not start until the user approves the picture.

## File Structure

- `src/retina.rs` (new) — `Source`, `SpatialHash`, `segment_transmittance`, `Receptor`, `PipeState`, `RetinaStats`, `Retina` (relink / arrive / tick / resize / log_stats), unit tests.
- `src/field.rs` — `pub(crate) cutoff_window`; `transport()` extracted from `tick()`; `advance_entities() -> Vec<Source>`; `tick()` rewired; `edge_atten`; `atten_k`; `ray_blocked` removed (Task 5), grid code removed (Task 8).
- `src/renderer.rs` — two 2D retina textures, whole-texture upload when dirty, resolution change, tuning/stats plumbing.
- `shaders/retina.wgsl` (new) — threshold + shade + composite. `shaders/field_sample.wgsl` deleted in Task 8.
- `src/main.rs` — `mod retina;`, keys `5`–`8`.
- `README.md`, `ARCHITECTURE.md`, `ROADMAP.md` — Task 8.

---

### Task 1: Branch, `Source`, `SpatialHash`, `segment_transmittance`

**Files:**
- Create: `src/retina.rs`
- Modify: `src/main.rs:9-13` (add `mod retina;`), `src/field.rs:310` (`pub(crate) fn cutoff_window`)

**Interfaces:**
- Produces:
  - `pub struct Source { position: Vec3, radii: Vec3, normal: Vec3, opacity: f32, density: f32, color: [f32; 3], drawable: bool, occluder: bool }` with `kernel_radii()` and `kernel(p) -> f32`
  - `pub struct SpatialHash` with `build(&[Source]) -> Self` and `density_at(&self, &[Source], p: Vec3, skip: &[usize]) -> f32`
  - `pub fn segment_transmittance(sources: &[Source], hash: &SpatialHash, from: Vec3, to: Vec3, skip: &[usize], k: f32) -> f32`
  - constants listed in Global Constraints

- [ ] **Step 1: Create the branch and bring the tuning commits over**

```bash
git checkout main
git checkout -b retina
git cherry-pick 19b8e60 e67127e            # H stats dump, tuning keys 1-4
git cherry-pick d55b70b desaturation-attenuation   # the spec, then this plan (branch tip)
cargo build --release 2>&1 | tail -3
```
Expected: all cherry-picks apply cleanly (they only touch `field.rs`, `renderer.rs`, `main.rs`, docs); build OK.

- [ ] **Step 2: Write the failing tests**

Create `src/retina.rs` with only the test module for now:

```rust
// Retina — the observer as a receptor array.
//
// The image is not a sample of a cached volume. It is the persistent state of
// W×H receptors on the image plane, each the sum of what has arrived along
// entity→receptor pipes. Pipes carry deltas only. Occlusion is transmittance
// along the pipe — nothing else.

use glam::{Mat4, Vec3, Vec4};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::field::cutoff_window;

pub const RETINA_W: u32 = 320;
pub const RETINA_H: u32 = 180;
pub const RETINA_ISO: f32 = 0.3;
pub const DELTA_EPS: f32 = 1e-4;
pub const RELINK_SHIFT: f32 = 0.5;
pub const ATTEN_END_MARGIN: f32 = 1.5;
pub const ATTEN_SAMPLES_PER_CELL: f32 = 1.0;
pub const ATTEN_THRESHOLD: f32 = 0.6;
pub const ATTEN_K_DEFAULT: f32 = 0.5;
pub const MIN_RETINA_DIM: u32 = 20;
pub const MAX_RETINA_DIM: u32 = 2560;

#[cfg(test)]
mod tests {
    use super::*;

    fn src(pos: Vec3, radii: Vec3, opacity: f32) -> Source {
        Source {
            position: pos,
            radii,
            normal: Vec3::Y,
            opacity,
            density: opacity,
            color: [1.0, 1.0, 1.0],
            drawable: true,
            occluder: true,
        }
    }

    #[test]
    fn kernel_is_one_at_center_and_zero_past_cutoff() {
        let s = src(Vec3::ZERO, Vec3::new(2.0, 1.0, 1.0), 1.0);
        assert!((s.kernel(Vec3::ZERO) - 1.0).abs() < 1e-6);
        assert_eq!(s.kernel(Vec3::new(4.0, 0.0, 0.0)), 0.0); // e = 4 → cut
        assert!(s.kernel(Vec3::new(2.0, 0.0, 0.0)) > 0.3);     // e = 1 → exp(-1)
        // point source falls back to unit radii
        let p = src(Vec3::ZERO, Vec3::ZERO, 1.0);
        assert_eq!(p.kernel_radii(), Vec3::ONE);
    }

    #[test]
    fn transmittance_is_one_with_no_occluder_on_the_segment() {
        let sources = vec![src(Vec3::new(10.0, 0.0, -5.0), Vec3::ONE, 10.0)];
        let hash = SpatialHash::build(&sources);
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, &[], 0.5);
        assert!((t - 1.0).abs() < 1e-6, "off-segment occluder attenuated: {}", t);
    }

    #[test]
    fn transmittance_matches_fine_quadrature_for_centered_occluder() {
        let occ = src(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 10.0);
        let sources = vec![occ];
        let hash = SpatialHash::build(&sources);
        let k = 0.5;
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, &[], k);
        // Independent reference: fine quadrature of the same integrand
        let mut integral = 0.0f32;
        let dz = 0.01;
        let mut z = -10.0 + ATTEN_END_MARGIN;
        while z < -ATTEN_END_MARGIN {
            let rho = occ.opacity * occ.kernel(Vec3::new(0.0, 0.0, z));
            integral += (rho - ATTEN_THRESHOLD).max(0.0) * dz;
            z += dz;
        }
        let expected = (-k * integral).exp();
        assert!(t < 0.5, "centered opaque occluder barely attenuates: {}", t);
        assert!((t - expected).abs() < 0.15 * expected.max(0.05), "τ={} expected≈{}", t, expected);
    }

    #[test]
    fn transmittance_skips_listed_sources_and_end_margins() {
        let a = src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 50.0); // at `from`
        let b = src(Vec3::ZERO, Vec3::ONE, 50.0);                  // at `to`
        let mid = src(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 50.0);
        let sources = vec![a, b, mid];
        let hash = SpatialHash::build(&sources);
        // Endpoint sources sit inside the 1.5-cell margins → never sampled
        let t_ends = segment_transmittance(&sources[..2], &SpatialHash::build(&sources[..2]),
            a.position, b.position, &[], 0.5);
        assert!((t_ends - 1.0).abs() < 1e-6, "endpoint sources leaked into the integral: {}", t_ends);
        // Skipping the middle occluder restores full transmittance
        let t_skip = segment_transmittance(&sources, &hash, a.position, b.position, &[2], 0.5);
        assert!((t_skip - 1.0).abs() < 1e-6);
        let t_block = segment_transmittance(&sources, &hash, a.position, b.position, &[], 0.5);
        assert!(t_block < 0.1);
    }

    #[test]
    fn spatial_hash_ignores_non_occluders() {
        let mut ghost = src(Vec3::ZERO, Vec3::ONE, 100.0);
        ghost.occluder = false;
        let sources = vec![ghost];
        let hash = SpatialHash::build(&sources);
        assert_eq!(hash.density_at(&sources, Vec3::ZERO, &[]), 0.0);
    }
}
```

Add to `src/main.rs` after `mod walker;`:
```rust
mod retina;
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test retina:: 2>&1 | tail -20`
Expected: compile errors — `Source`, `SpatialHash`, `segment_transmittance` not found.

- [ ] **Step 4: Implement**

In `src/field.rs:310` change `fn cutoff_window` to `pub(crate) fn cutoff_window`.

Add to `src/retina.rs` above the test module:

```rust
/// One entity's contribution as the retina sees it. Index-aligned with
/// `DiffField::entities` so pipes and edges can name entities by index.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub position: Vec3,
    /// Deposit radii (axis-aligned gaussian). ZERO → point source, unit radii.
    pub radii: Vec3,
    pub normal: Vec3,
    /// Static density used for transmittance (magnitude × boost, untuned).
    pub opacity: f32,
    /// Boosted density delivered this tick (τ applied by the retina).
    pub density: f32,
    /// Boosted color delivered this tick.
    pub color: [f32; 3],
    /// Gets pipes: solid, in frustum, not culled by trie depth.
    pub drawable: bool,
    /// Attenuates segments: solid, regardless of frustum.
    pub occluder: bool,
}

impl Source {
    pub fn kernel_radii(&self) -> Vec3 {
        if self.radii == Vec3::ZERO { Vec3::ONE } else { self.radii }
    }

    /// Feathered gaussian, 1.0 at the center, exactly 0.0 at exponent 4.
    /// Same kernel the grid deposit used, evaluated analytically.
    pub fn kernel(&self, p: Vec3) -> f32 {
        let d = (p - self.position) / self.kernel_radii();
        let e = d.dot(d);
        if e >= 4.0 { 0.0 } else { (-e).exp() * cutoff_window(e) }
    }
}

/// Uniform grid over occluder sources; cell = the widest kernel extent so a
/// 27-cell neighbourhood always covers every kernel that can reach a point.
pub struct SpatialHash {
    cell: f32,
    map: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHash {
    pub fn build(sources: &[Source]) -> Self {
        let max_r = sources.iter()
            .filter(|s| s.occluder)
            .map(|s| s.kernel_radii().max_element())
            .fold(1.0f32, f32::max);
        let cell = max_r * 2.0;
        let mut map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, s) in sources.iter().enumerate() {
            if !s.occluder { continue; }
            map.entry(Self::key(s.position, cell)).or_default().push(i);
        }
        Self { cell, map }
    }

    fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
        ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32)
    }

    /// Σ opacity·kernel over occluders near `p`, excluding indices in `skip`.
    pub fn density_at(&self, sources: &[Source], p: Vec3, skip: &[usize]) -> f32 {
        let (cx, cy, cz) = Self::key(p, self.cell);
        let mut rho = 0.0f32;
        for dz in -1..=1_i32 {
            for dy in -1..=1_i32 {
                for dx in -1..=1_i32 {
                    if let Some(bucket) = self.map.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in bucket {
                            if skip.contains(&j) { continue; }
                            rho += sources[j].opacity * sources[j].kernel(p);
                        }
                    }
                }
            }
        }
        rho
    }
}

/// Fraction of light surviving the segment from → to:
/// exp(−k · ∫ max(0, ρ − ATTEN_THRESHOLD) ds), skipping ATTEN_END_MARGIN at
/// both ends so an entity's own kernel (and its partner's) never self-shadows.
pub fn segment_transmittance(
    sources: &[Source],
    hash: &SpatialHash,
    from: Vec3,
    to: Vec3,
    skip: &[usize],
    k: f32,
) -> f32 {
    let ab = to - from;
    let len = ab.length();
    if len - 2.0 * ATTEN_END_MARGIN <= 0.0 { return 1.0; }
    let dir = ab / len;
    let step = 1.0 / ATTEN_SAMPLES_PER_CELL;
    let mut integral = 0.0f32;
    let mut t = ATTEN_END_MARGIN + 0.5 * step;
    while t < len - ATTEN_END_MARGIN {
        let rho = hash.density_at(sources, from + dir * t, skip);
        integral += (rho - ATTEN_THRESHOLD).max(0.0) * step;
        t += step;
    }
    (-k * integral).exp()
}
```

- [ ] **Step 5: Run tests and build**

Run: `cargo test retina:: 2>&1 | tail -15 && cargo build --release 2>&1 | tail -2`
Expected: 5 passed; build OK (unused-import warnings for `Mat4`, `Vec4`, rayon are fine for now).

- [ ] **Step 6: Commit**

```bash
git add src/retina.rs src/main.rs src/field.rs
git commit -m "feat(retina): Source, SpatialHash, segment_transmittance"
```

---

### Task 2: Receptors, projection, and linking

**Files:**
- Modify: `src/retina.rs`

**Interfaces:**
- Consumes: Task 1 types.
- Produces:
  - `pub struct Receptor { density, color: [f32;3], normal: Vec3, depth }` (`Copy`, `Default`, `PartialEq`)
  - `pub struct RetinaStats { pipes_total, pipes_sent, relinks, mean_trans, relink_ms }`
  - `pub struct Retina` with `new(w, h)`, `resize(w, h)`, `needs_relink(view_proj, aabb_min, aabb_max) -> bool`, `relink(sources, hash, view_proj, eye, atten_k)`, `pipes_of(i) -> impl Iterator<Item=(u32, f32)>`, `transmittance(i) -> f32`, `depth_of(i) -> f32`, `pub receptors`, `pub stats`, `pub dirty`, `pub width`, `pub height`
  - `pub fn eye_from_view_proj(view_proj: Mat4) -> Vec3`
  - test helper `fn test_view_proj(w, h) -> Mat4` (eye at origin looking −Z, 90° fov)

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests`:

```rust
    /// Eye at origin looking down −Z, 90° fov, aspect w/h. With odd w/h a
    /// point on the axis projects exactly onto the center receptor's center.
    pub(super) fn test_view_proj(w: u32, h: u32) -> Mat4 {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, w as f32 / h as f32, 0.1, 500.0);
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        proj * view
    }

    #[test]
    fn eye_is_recovered_from_view_proj() {
        let eye = eye_from_view_proj(test_view_proj(63, 35));
        assert!(eye.length() < 1e-3, "eye={:?}", eye);
    }

    #[test]
    fn point_source_on_axis_links_center_receptor_with_unit_weight() {
        let vp = test_view_proj(63, 35);
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, 1.0)];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        let center: u32 = 17 * 63 + 31;
        let w = r.pipes_of(0).find(|&(rc, _)| rc == center).map(|(_, w)| w);
        assert!(w.is_some(), "no pipe to the center receptor");
        assert!((w.unwrap() - 1.0).abs() < 1e-5, "center weight {}", w.unwrap());
        assert!((r.depth_of(0) - 10.0).abs() < 1e-3);
        assert!(r.pipes_of(0).count() >= 1);
    }

    #[test]
    fn gaussian_footprint_is_symmetric_and_sized_by_projection() {
        // 90° fov at distance 10: 1 cell = 35/2/10 = 1.75 receptors.
        // radii 2 → σ = 3.5 receptors, extent 2σ = 7 → ~π·7² ≈ 150 receptors.
        let vp = test_view_proj(63, 35);
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(2.0), 1.0)];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        let pipes: HashMap<u32, f32> = r.pipes_of(0).collect();
        assert!(pipes.len() > 100 && pipes.len() < 200, "footprint {} pipes", pipes.len());
        for k in 1..=5u32 {
            let l = pipes[&(17 * 63 + 31 - k)];
            let rt = pipes[&(17 * 63 + 31 + k)];
            let up = pipes[&((17 - k) * 63 + 31)];
            assert!((l - rt).abs() < 1e-5 && (l - up).abs() < 1e-5, "asymmetric at k={}", k);
        }
    }

    #[test]
    fn off_screen_and_non_drawable_sources_get_no_pipes() {
        let vp = test_view_proj(63, 35);
        let mut hidden = src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 1.0);
        hidden.drawable = false;
        let sources = vec![
            src(Vec3::new(0.0, 0.0, 10.0), Vec3::ONE, 1.0),   // behind the eye
            src(Vec3::new(100.0, 0.0, -10.0), Vec3::ONE, 1.0), // far off to the right
            hidden,
        ];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        for i in 0..3 { assert_eq!(r.pipes_of(i).count(), 0, "source {} got pipes", i); }
        assert_eq!(r.stats.pipes_total, 0);
    }

    #[test]
    fn occluded_source_has_low_transmittance() {
        let vp = test_view_proj(63, 35);
        let sources = vec![
            src(Vec3::new(0.0, 0.0, -20.0), Vec3::ONE, 40.0), // behind
            src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 40.0), // in front
        ];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        assert!(r.transmittance(0) < 0.2, "back τ={}", r.transmittance(0));
        assert!((r.transmittance(1) - 1.0).abs() < 1e-3, "front τ={}", r.transmittance(1));
    }

    #[test]
    fn relink_trigger_follows_projected_shift() {
        let vp = test_view_proj(63, 35);
        let mut r = Retina::new(63, 35);
        let (lo, hi) = (Vec3::new(-5.0, -5.0, -15.0), Vec3::new(5.0, 5.0, -5.0));
        assert!(r.needs_relink(vp, lo, hi), "fresh retina must relink");
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 1.0)];
        r.relink(&sources, &SpatialHash::build(&sources), vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        assert!(!r.needs_relink(vp, lo, hi), "same view must not relink");
        // Tiny nudge: 0.001 cells at distance 10 ≈ 0.002 receptors → no relink
        let nudge = Mat4::from_translation(Vec3::new(0.001, 0.0, 0.0));
        assert!(!r.needs_relink(vp * nudge, lo, hi));
        // 1 cell sideways ≈ 1.75 receptors → relink
        let shove = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        assert!(r.needs_relink(vp * shove, lo, hi));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test retina:: 2>&1 | tail -20`
Expected: compile errors — `Retina`, `Receptor`, `eye_from_view_proj` not found.

- [ ] **Step 3: Implement**

Add above the test module:

```rust
/// Persistent state of one image-plane cell. Sums, never averages: the
/// shader divides by density. Never decays; only deltas ever touch it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Receptor {
    pub density: f32,
    pub color: [f32; 3],
    pub normal: Vec3,
    pub depth: f32,
}

impl Receptor {
    fn add(&mut self, p: &PipeState) {
        self.density += p.density;
        self.color[0] += p.color[0];
        self.color[1] += p.color[1];
        self.color[2] += p.color[2];
        self.normal += p.normal;
        self.depth += p.depth;
    }
    fn sub(&mut self, p: &PipeState) {
        self.density -= p.density;
        self.color[0] -= p.color[0];
        self.color[1] -= p.color[1];
        self.color[2] -= p.color[2];
        self.normal -= p.normal;
        self.depth -= p.depth;
    }
    fn add_receptor(&mut self, o: &Receptor) {
        self.density += o.density;
        self.color[0] += o.color[0];
        self.color[1] += o.color[1];
        self.color[2] += o.color[2];
        self.normal += o.normal;
        self.depth += o.depth;
    }
}

/// What a pipe last delivered — the delta baseline. Same shape as a receptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct PipeState {
    pub density: f32,
    pub color: [f32; 3],
    pub normal: Vec3,
    pub depth: f32,
}

impl PipeState {
    fn scaled(&self, w: f32) -> PipeState {
        PipeState {
            density: self.density * w,
            color: [self.color[0] * w, self.color[1] * w, self.color[2] * w],
            normal: self.normal * w,
            depth: self.depth * w,
        }
    }
    fn minus(&self, o: &PipeState) -> PipeState {
        PipeState {
            density: self.density - o.density,
            color: [self.color[0] - o.color[0], self.color[1] - o.color[1], self.color[2] - o.color[2]],
            normal: self.normal - o.normal,
            depth: self.depth - o.depth,
        }
    }
    fn max_abs(&self) -> f32 {
        self.density.abs()
            .max(self.color[0].abs()).max(self.color[1].abs()).max(self.color[2].abs())
            .max(self.normal.abs().max_element())
            .max(self.depth.abs())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RetinaStats {
    pub pipes_total: usize,
    pub pipes_sent: usize,
    pub relinks: u64,
    pub mean_trans: f32,
    pub relink_ms: f32,
}

/// Image-space footprint of a projected axis-aligned gaussian: center,
/// eye distance, inverse 2×2 covariance (e = a·du² + 2b·du·dv + c·dv²),
/// and half-extents of the e < 4 bounding box.
struct Footprint {
    u: f32,
    v: f32,
    depth: f32,
    a: f32,
    b: f32,
    c: f32,
    hu: f32,
    hv: f32,
}

/// World point → (u right, v down, in receptor units; eye distance). None if
/// behind the eye. Matches the shader's UV convention (v=0 at the top).
fn project(vp: &Mat4, w: u32, h: u32, p: Vec3) -> Option<(f32, f32, f32)> {
    let c = *vp * Vec4::new(p.x, p.y, p.z, 1.0);
    if c.w <= 1e-4 { return None; }
    let ndc = c.truncate() / c.w;
    Some((
        (ndc.x * 0.5 + 0.5) * w as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * h as f32,
        c.w,
    ))
}

fn footprint(vp: &Mat4, w: u32, h: u32, s: &Source) -> Option<Footprint> {
    let (u, v, depth) = project(vp, w, h, s.position)?;
    let r = s.kernel_radii();
    let (mut suu, mut suv, mut svv) = (0.0f32, 0.0f32, 0.0f32);
    for axis in [Vec3::X * r.x, Vec3::Y * r.y, Vec3::Z * r.z] {
        if let Some((pu, pv, _)) = project(vp, w, h, s.position + axis) {
            let du = pu - u;
            let dv = pv - v;
            suu += du * du;
            suv += du * dv;
            svv += dv * dv;
        }
    }
    // Floor the variance at half a receptor so distant entities keep a pipe.
    let min_var = 0.25;
    suu = suu.max(min_var);
    svv = svv.max(min_var);
    let det = (suu * svv - suv * suv).max(1e-6);
    Some(Footprint {
        u, v, depth,
        a: svv / det,
        b: -suv / det,
        c: suu / det,
        hu: 2.0 * suu.sqrt(),
        hv: 2.0 * svv.sqrt(),
    })
}

/// Observer position from the view-projection (same trick Phase 2 uses).
pub fn eye_from_view_proj(view_proj: Mat4) -> Vec3 {
    let inv = view_proj.inverse();
    Vec3::new(inv.col(3).x, inv.col(3).y, inv.col(3).z)
}

pub struct Retina {
    pub width: u32,
    pub height: u32,
    pub receptors: Vec<Receptor>,
    // Pipes entity → receptor, SoA, per-entity contiguous slices.
    pipe_start: Vec<u32>,
    pipe_count: Vec<u32>,
    pipe_receptor: Vec<u32>,
    pipe_weight: Vec<f32>,
    pipe_last: Vec<PipeState>,
    /// τ toward the eye, per entity (1.0 for entities without pipes).
    entity_trans: Vec<f32>,
    /// Eye distance per entity at the last relink.
    entity_depth: Vec<f32>,
    last_view_proj: Option<Mat4>,
    pub stats: RetinaStats,
    /// Set when any delta arrived since the renderer last uploaded.
    pub dirty: bool,
}

impl Retina {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            receptors: vec![Receptor::default(); (width * height) as usize],
            pipe_start: Vec::new(),
            pipe_count: Vec::new(),
            pipe_receptor: Vec::new(),
            pipe_weight: Vec::new(),
            pipe_last: Vec::new(),
            entity_trans: Vec::new(),
            entity_depth: Vec::new(),
            last_view_proj: None,
            stats: RetinaStats::default(),
            dirty: true,
        }
    }

    /// Change resolution. Everything is rebuilt at the next tick's relink.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.clamp(MIN_RETINA_DIM, MAX_RETINA_DIM);
        let height = height.clamp(MIN_RETINA_DIM, MAX_RETINA_DIM);
        *self = Retina::new(width, height);
    }

    pub fn pipes_of(&self, i: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let (start, count) = if i < self.pipe_start.len() {
            (self.pipe_start[i] as usize, self.pipe_count[i] as usize)
        } else { (0, 0) };
        (start..start + count).map(move |k| (self.pipe_receptor[k], self.pipe_weight[k]))
    }

    pub fn transmittance(&self, i: usize) -> f32 {
        self.entity_trans.get(i).copied().unwrap_or(1.0)
    }

    pub fn depth_of(&self, i: usize) -> f32 {
        self.entity_depth.get(i).copied().unwrap_or(0.0)
    }

    /// True when the scene AABB's corners have shifted ≥ RELINK_SHIFT
    /// receptors between the last linking view and this one.
    pub fn needs_relink(&self, view_proj: Mat4, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        let Some(last) = self.last_view_proj else { return true; };
        if last == view_proj { return false; }
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { aabb_min.x } else { aabb_max.x },
                if i & 2 == 0 { aabb_min.y } else { aabb_max.y },
                if i & 4 == 0 { aabb_min.z } else { aabb_max.z },
            );
            match (project(&last, self.width, self.height, corner),
                   project(&view_proj, self.width, self.height, corner)) {
                (Some(a), Some(b)) => {
                    if (a.0 - b.0).abs().max((a.1 - b.1).abs()) >= RELINK_SHIFT { return true; }
                }
                _ => return true,
            }
        }
        false
    }

    /// Full rebuild: drop every pipe (subtracting what it last sent — the
    /// receptors are then exactly zero), re-project every drawable source,
    /// recompute τ toward the eye.
    pub fn relink(&mut self, sources: &[Source], hash: &SpatialHash, view_proj: Mat4, eye: Vec3, atten_k: f32) {
        let t0 = std::time::Instant::now();
        let (w, h) = (self.width, self.height);

        // 1. Drop. Receptors are the exact sum of pipe_last, so this lands on
        //    zero; the explicit reset below only removes float drift.
        for (k, &rc) in self.pipe_receptor.iter().enumerate() {
            self.receptors[rc as usize].sub(&self.pipe_last[k]);
        }
        debug_assert!(self.receptors.iter().all(|r| r.density.abs() < 1e-2),
            "receptors not zero after dropping all pipes");
        for r in &mut self.receptors { *r = Receptor::default(); }

        // 2–3. Project footprints; 5. τ toward the eye (parallel — this is the cost).
        let fps: Vec<Option<Footprint>> = sources.par_iter()
            .map(|s| if s.drawable { footprint(&view_proj, w, h, s) } else { None })
            .collect();
        let n = sources.len();
        self.entity_trans = (0..n).into_par_iter().map(|i| {
            if fps[i].is_some() {
                segment_transmittance(sources, hash, sources[i].position, eye, &[i], atten_k)
            } else { 1.0 }
        }).collect();
        self.entity_depth = fps.iter().map(|f| f.as_ref().map(|f| f.depth).unwrap_or(0.0)).collect();

        // 4. Pipes with feathered gaussian weights.
        self.pipe_start = vec![0; n];
        self.pipe_count = vec![0; n];
        self.pipe_receptor.clear();
        self.pipe_weight.clear();
        for (i, fp) in fps.iter().enumerate() {
            self.pipe_start[i] = self.pipe_receptor.len() as u32;
            let Some(fp) = fp else { continue; };
            let u0 = (fp.u - fp.hu).floor().max(0.0) as i64;
            let u1 = (fp.u + fp.hu).ceil().min(w as f32 - 1.0) as i64;
            let v0 = (fp.v - fp.hv).floor().max(0.0) as i64;
            let v1 = (fp.v + fp.hv).ceil().min(h as f32 - 1.0) as i64;
            if u0 > u1 || v0 > v1 { continue; }
            for rv in v0..=v1 {
                for ru in u0..=u1 {
                    let du = ru as f32 + 0.5 - fp.u;
                    let dv = rv as f32 + 0.5 - fp.v;
                    let e = fp.a * du * du + 2.0 * fp.b * du * dv + fp.c * dv * dv;
                    if e >= 4.0 { continue; }
                    self.pipe_receptor.push(rv as u32 * w + ru as u32);
                    self.pipe_weight.push((-e).exp() * cutoff_window(e));
                }
            }
            self.pipe_count[i] = self.pipe_receptor.len() as u32 - self.pipe_start[i];
        }
        self.pipe_last = vec![PipeState::default(); self.pipe_receptor.len()];

        // 6. Bookkeeping.
        self.last_view_proj = Some(view_proj);
        let linked: Vec<f32> = (0..n).filter(|&i| self.pipe_count[i] > 0).map(|i| self.entity_trans[i]).collect();
        self.stats.relinks += 1;
        self.stats.pipes_total = self.pipe_receptor.len();
        self.stats.mean_trans = if linked.is_empty() { 1.0 } else { linked.iter().sum::<f32>() / linked.len() as f32 };
        self.stats.relink_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.dirty = true;
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test retina:: 2>&1 | tail -20`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/retina.rs
git commit -m "feat(retina): receptors, projection footprints, relink with per-entity transmittance"
```

---

### Task 3: Delta arrival, tick, stats

**Files:**
- Modify: `src/retina.rs`

**Interfaces:**
- Consumes: Task 2 `Retina`.
- Produces:
  - `Retina::arrive(&mut self, sources: &[Source]) -> usize` (pipes sent)
  - `Retina::tick(&mut self, sources: &[Source], view_proj: Mat4, aabb_min: Vec3, aabb_max: Vec3, force_relink: bool, atten_k: f32)`
  - `Retina::log_stats(&self)`
  - `Retina::direct_sum(&self, sources: &[Source]) -> Vec<Receptor>` (reference, used by tests and `log_stats` self-check in debug)

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests`:

```rust
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX >> 1) as f32
    }

    fn scene() -> Vec<Source> {
        vec![
            src(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(2.0), 5.0),
            src(Vec3::new(3.0, 1.0, -12.0), Vec3::new(1.0, 2.0, 1.0), 3.0),
            src(Vec3::new(-4.0, -2.0, -9.0), Vec3::ONE, 8.0),
            src(Vec3::new(1.0, -1.0, -20.0), Vec3::splat(3.0), 2.0),
            src(Vec3::new(0.0, 4.0, -15.0), Vec3::ZERO, 1.0),
        ]
    }

    fn assert_receptors_match(r: &Retina, sources: &[Source], tol: f32) {
        let want = r.direct_sum(sources);
        for (i, (got, want)) in r.receptors.iter().zip(&want).enumerate() {
            assert!((got.density - want.density).abs() < tol, "receptor {} density {} vs {}", i, got.density, want.density);
            for c in 0..3 {
                assert!((got.color[c] - want.color[c]).abs() < tol, "receptor {} color[{}]", i, c);
            }
            assert!((got.normal - want.normal).length() < tol, "receptor {} normal", i);
            assert!((got.depth - want.depth).abs() < tol * 30.0, "receptor {} depth", i);
        }
    }

    #[test]
    fn receptors_equal_direct_sum_after_random_updates() {
        let vp = test_view_proj(63, 35);
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let mut sources = scene();
        let mut r = Retina::new(63, 35);
        let mut seed = 7u64;
        for _ in 0..20 {
            for s in &mut sources {
                s.density = 1.0 + 40.0 * lcg(&mut seed);
                s.color = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
            }
            r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
            assert_receptors_match(&r, &sources, 1e-2);
        }
        assert_eq!(r.stats.relinks, 1, "static view must link exactly once");
    }

    #[test]
    fn settled_scene_sends_nothing() {
        let vp = test_view_proj(63, 35);
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let sources = scene();
        let mut r = Retina::new(63, 35);
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert!(r.stats.pipes_sent > 0);
        r.dirty = false;
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert_eq!(r.stats.pipes_sent, 0, "settled scene still sends deltas");
        assert!(!r.dirty);
    }

    #[test]
    fn relink_keeps_receptors_exact() {
        let vp_a = test_view_proj(63, 35);
        let vp_b = vp_a * Mat4::from_translation(Vec3::new(2.0, 0.5, 0.0));
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let sources = scene();
        let mut r = Retina::new(63, 35);
        r.tick(&sources, vp_a, lo, hi, false, ATTEN_K_DEFAULT);
        let before: Vec<Receptor> = r.receptors.clone();
        r.tick(&sources, vp_b, lo, hi, false, ATTEN_K_DEFAULT);
        assert_eq!(r.stats.relinks, 2);
        assert_ne!(before, r.receptors, "view moved but the image did not");
        assert_receptors_match(&r, &sources, 1e-2);
    }

    #[test]
    fn source_that_stops_drawing_is_withdrawn() {
        let vp = test_view_proj(63, 35);
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let mut sources = scene();
        let mut r = Retina::new(63, 35);
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        sources[0].drawable = false; // culled between relinks (e.g. trie depth)
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert_receptors_match(&r, &sources, 1e-2);
    }

    #[test]
    fn resolution_change_preserves_mean_density() {
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(3.0), 10.0)];
        let mean = |w: u32, h: u32| {
            let mut r = Retina::new(w, h);
            r.tick(&sources, test_view_proj(w, h), lo, hi, false, ATTEN_K_DEFAULT);
            r.receptors.iter().map(|x| x.density).sum::<f32>() / (w * h) as f32
        };
        let (coarse, fine) = (mean(63, 35), mean(126, 70));
        assert!((coarse - fine).abs() < 0.2 * fine, "coarse {} vs fine {}", coarse, fine);
        let mut r = Retina::new(63, 35);
        r.resize(126, 70);
        assert_eq!(r.receptors.len(), 126 * 70);
        r.resize(1, 1);
        assert_eq!((r.width, r.height), (MIN_RETINA_DIM, MIN_RETINA_DIM));
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test retina:: 2>&1 | tail -20`
Expected: compile errors — `tick`, `direct_sum` not found.

- [ ] **Step 3: Implement**

Add inside `impl Retina`:

```rust
    /// What entity `i` offers its pipes this tick (before the footprint
    /// weight). A source that stopped drawing offers zero, so its pipes
    /// withdraw what they last sent.
    fn contribution(&self, i: usize, s: &Source) -> PipeState {
        if !s.drawable { return PipeState::default(); }
        let d = s.density * self.entity_trans[i];
        PipeState {
            density: d,
            color: [s.color[0] * self.entity_trans[i], s.color[1] * self.entity_trans[i], s.color[2] * self.entity_trans[i]],
            normal: s.normal * d,
            depth: self.entity_depth[i] * d,
        }
    }

    /// Phase 3′: every pipe sends `new − last` if it exceeds DELTA_EPS.
    /// Parallel over entities; receptors are shared by many entities, so each
    /// rayon split accumulates into a scratch image that is reduced at the end.
    pub fn arrive(&mut self, sources: &[Source]) -> usize {
        let n_rec = self.receptors.len();
        let n = sources.len().min(self.pipe_count.len());

        // Contributions first (needs &self), then the per-entity mutable views
        // of pipe_last (contiguous, ascending) — direct field borrows, no unsafe.
        let contribs: Vec<PipeState> = (0..n).map(|i| self.contribution(i, &sources[i])).collect();
        let mut slices: Vec<&mut [PipeState]> = Vec::with_capacity(n);
        let mut rest: &mut [PipeState] = &mut self.pipe_last;
        for i in 0..n {
            let (head, tail) = rest.split_at_mut(self.pipe_count[i] as usize);
            slices.push(head);
            rest = tail;
        }

        let pipe_start = &self.pipe_start;
        let pipe_receptor = &self.pipe_receptor;
        let pipe_weight = &self.pipe_weight;

        let (scratch, sent) = slices.into_par_iter().enumerate()
            .fold(
                || (vec![Receptor::default(); n_rec], 0usize),
                |(mut acc, mut sent), (i, last)| {
                    let start = pipe_start[i] as usize;
                    for (off, l) in last.iter_mut().enumerate() {
                        let k = start + off;
                        let new = contribs[i].scaled(pipe_weight[k]);
                        let delta = new.minus(l);
                        if delta.max_abs() > DELTA_EPS {
                            acc[pipe_receptor[k] as usize].add(&delta);
                            *l = new;
                            sent += 1;
                        }
                    }
                    (acc, sent)
                },
            )
            .reduce(
                || (vec![Receptor::default(); n_rec], 0usize),
                |(mut a, sa), (b, sb)| {
                    for (x, y) in a.iter_mut().zip(&b) { x.add_receptor(y); }
                    (a, sa + sb)
                },
            );

        if sent > 0 {
            for (r, s) in self.receptors.iter_mut().zip(&scratch) { r.add_receptor(s); }
            self.dirty = true;
        }
        self.stats.pipes_sent = sent;
        sent
    }

    /// One retina step: relink if the view or the links moved, then arrive.
    pub fn tick(&mut self, sources: &[Source], view_proj: Mat4, aabb_min: Vec3, aabb_max: Vec3, force_relink: bool, atten_k: f32) {
        if force_relink || self.needs_relink(view_proj, aabb_min, aabb_max) {
            let hash = SpatialHash::build(sources);
            let eye = eye_from_view_proj(view_proj);
            self.relink(sources, &hash, view_proj, eye, atten_k);
        }
        self.arrive(sources);
    }

    /// Reference image: Σ over pipes of contribution·weight, from scratch.
    /// The incremental receptors must equal this (up to DELTA_EPS per pipe).
    pub fn direct_sum(&self, sources: &[Source]) -> Vec<Receptor> {
        let mut out = vec![Receptor::default(); self.receptors.len()];
        for i in 0..sources.len().min(self.pipe_count.len()) {
            let c = self.contribution(i, &sources[i]);
            for (rc, w) in self.pipes_of(i) {
                out[rc as usize].add(&c.scaled(w));
            }
        }
        out
    }

    pub fn log_stats(&self) {
        let above = self.receptors.iter().filter(|r| r.density >= RETINA_ISO).count();
        let max_d = self.receptors.iter().map(|r| r.density).fold(0.0f32, f32::max);
        log::info!(
            "Retina {}x{}: {} receptors ≥ iso ({:.1}%), max density {:.2}; pipes {} total / {} sent last tick; mean τ {:.3}; {} relinks (last {:.2} ms)",
            self.width, self.height, above,
            100.0 * above as f64 / self.receptors.len().max(1) as f64, max_d,
            self.stats.pipes_total, self.stats.pipes_sent, self.stats.mean_trans,
            self.stats.relinks, self.stats.relink_ms,
        );
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test retina:: 2>&1 | tail -20`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/retina.rs
git commit -m "feat(retina): delta-only arrival, tick, stats"
```

---

### Task 4: Wire the retina into `DiffField::tick`

**Files:**
- Modify: `src/field.rs` — struct (`~195-245`), `new()` (`~328-360`), `tick()` (`~1615-2242`), tests module
- Modify: `src/renderer.rs:477-479` (`dump_field_stats`)

**Interfaces:**
- Consumes: `crate::retina::{Retina, Source, RETINA_W, RETINA_H, ATTEN_K_DEFAULT}`.
- Produces:
  - `DiffField { pub retina: Retina, pub atten_k: f32, pub retina_force_relink: bool, sources: Vec<Source> }`
  - `fn transport(&mut self, view_proj: Mat4) -> bool` — refresh + active set + atmosphere + Phase 1 + consumption + Phase 2; returns whether a cross-link refresh ran this tick
  - `fn advance_entities(&mut self)` — walker step, move/bounce, AABB, animation, emission → `self.sources` (index-aligned with `entities`)
  - `pub fn tick(&mut self, view_proj)` — transport → advance → retina.tick
  - `#[allow(dead_code)] pub fn tick_grid(&mut self, view_proj)` — the old body (transport + Phase 0 + Phase 3), kept as reference until Task 8
  - `pub fn dump_field_stats(&self)` — retina stats + floor probe (incoming only)

- [ ] **Step 1: Write the failing test**

Append to `mod tests` in `src/field.rs`:

```rust
    #[test]
    #[ignore] // builds the full 512³ field (~2 GB, slow) — run: cargo test --release -- --ignored
    fn retina_sees_the_dino_and_settles() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        for _ in 0..60 {
            field.tick(vp);
        }
        let r = &field.retina;
        let above = r.receptors.iter().filter(|x| x.density >= crate::retina::RETINA_ISO).count();
        assert!(above > 500, "only {} receptors above iso — dino not on the retina", above);
        assert!(r.receptors.iter().all(|x| x.density.is_finite() && x.color.iter().all(|c| c.is_finite())),
            "non-finite receptor");
        // Walker-group sources are drawable and linked
        let linked_walkers = field.entities.iter().enumerate()
            .filter(|(i, e)| e.is_walker && r.pipes_of(*i).count() > 0).count();
        assert!(linked_walkers > 100, "only {} walker entities have pipes", linked_walkers);
        // Freeze the world: no motion, no animation. Lighting keeps
        // converging (consumption learning nudges mass boosts), so demand
        // "almost silent", not zero — the unit tests prove exact zero.
        field.walker.speed_c = 0.0;
        for e in &mut field.entities { e.velocity = glam::Vec3::ZERO; }
        field.freeze_animation = true;
        for _ in 0..10 { field.tick(vp); }
        let s = field.retina.stats;
        assert!(s.pipes_sent * 100 < s.pipes_total,
            "frozen scene still sends {} of {} pipes", s.pipes_sent, s.pipes_total);
    }
```

Add `pub freeze_animation: bool` to `DiffField` (default `false`); when true, `advance_entities` skips the tail-wag and jaw offsets and the oscillation offset. It is test-only plumbing but harmless.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release -- --ignored retina_sees 2>&1 | tail -5`
Expected: compile error — no `retina`, `freeze_animation` fields.

- [ ] **Step 3: Split `tick()` into `transport()` + `tick_grid()`**

In `src/field.rs`:

1. Rename the existing `pub fn tick(&mut self, view_proj: glam::Mat4)` to `#[allow(dead_code)] pub fn tick_grid(&mut self, view_proj: glam::Mat4)`.
2. Cut everything from the top of `tick_grid` up to and including the end of the Phase 2 `par_iter().for_each(...)` block — **except** the Phase 0 decay block (`// Phase 0: AABB-restricted decay` … the `for z in 0..fs { … }` loop) — into a new method:

```rust
    /// Light transport for one sim step: cross-link refresh, active set,
    /// atmosphere modulation, Phase 1 deliver, consumption, Phase 2 push.
    /// Returns true if the cross-link refresh ran (the retina relinks then).
    fn transport(&mut self, view_proj: glam::Mat4) -> bool {
        let mut refreshed = false;
        if self.travel_since_refresh >= LINK_REFRESH_DIST
            && self.tick.saturating_sub(self.last_refresh_tick) >= MIN_REFRESH_SPACING
        {
            self.travel_since_refresh = 0.0;
            self.last_refresh_tick = self.tick;
            self.refresh_cross_links();
            refreshed = true;
        }
        // ... compute_active_set, atmosphere block, Phase 1, consumption, Phase 2 — verbatim ...
        refreshed
    }
```

3. `tick_grid` becomes: `let _ = self.transport(view_proj);` then the Phase 0 decay block, then the walker motion + Phase 3a/3b + trie diagnostics + `self.tick += 1;` verbatim.

Run `cargo test 2>&1 | tail -3` — all existing tests must still pass (nothing changed behaviourally; Phase 0 only touches `cells`, which Phases 1–2 never read).

- [ ] **Step 4: Add the retina fields and `advance_entities`**

Struct fields (after `tune_color`):
```rust
    /// The observer's retina — receptor array fed by entity pipes.
    pub retina: crate::retina::Retina,
    /// Transmittance strength for occlusion (keys 5/6).
    pub atten_k: f32,
    /// Set by tuning keys so the next tick relinks even if the view is static.
    pub retina_force_relink: bool,
    /// Test plumbing: park tail/jaw/skin animation so a scene can be frozen.
    pub freeze_animation: bool,
    /// Per-entity retina sources, rebuilt each tick (reused allocation).
    sources: Vec<crate::retina::Source>,
```
In `new()`:
```rust
            retina: crate::retina::Retina::new(crate::retina::RETINA_W, crate::retina::RETINA_H),
            atten_k: crate::retina::ATTEN_K_DEFAULT,
            retina_force_relink: false,
            freeze_animation: false,
            sources: Vec::new(),
```

New method — this is the old Phase 3a movement/animation and Phase 3b emission formula, minus every grid write:

```rust
    /// Move and animate entities, track the AABB, and build this tick's
    /// retina sources. Replaces the grid deposit: the emission formula is
    /// the old Phase 3b one, but it goes to the retina as a Source.
    fn advance_entities(&mut self) {
        use crate::retina::Source;
        let walk_delta = self.walker.step();
        self.travel_since_refresh += walk_delta.length();
        let walk_offset_z = self.walker.offset.z;
        let tune_density = self.tune_density;
        let tune_color = self.tune_color;
        let freeze = self.freeze_animation;
        let tick = self.tick;

        let n = self.entities.len();
        self.sources.clear();
        self.sources.resize(n, Source {
            position: glam::Vec3::ZERO, radii: glam::Vec3::ZERO, normal: glam::Vec3::Y,
            opacity: 0.0, density: 0.0, color: [0.0; 3], drawable: false, occluder: false,
        });

        let mut aabb_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut aabb_max = glam::Vec3::splat(0.0);
        for ent_idx in 0..n {
            // Read everything that lives outside `entities` first, so the
            // mutable entity borrow below does not conflict.
            let visible = self.visible_set[ent_idx];
            let show_depth = self.show_trie_depth;
            let (depth_cull, mass_boost, depth_col) =
                match self.consumption_states.get(ent_idx).and_then(|s| s.as_ref()) {
                    Some(state) => (
                        state.depth > self.render_depth_cutoff,
                        if !state.learning && state.consumed > 0 {
                            1.0 + (state.consumed as f32).ln().max(0.0) * 0.05
                        } else { 1.0 },
                        Some(crate::consumption::depth_color(state.depth)),
                    ),
                    None => (false, 1.0f32, None),
                };

            let entity = &mut self.entities[ent_idx];
            if entity.is_walker { entity.position += walk_delta; }
            entity.position += entity.velocity;
            for i in 0..3 {
                if entity.position[i] < 1.0 || entity.position[i] >= (FIELD_SIZE - 1) as f32 {
                    entity.velocity[i] *= -1.0;
                    entity.position[i] = entity.position[i].clamp(1.0, (FIELD_SIZE - 2) as f32);
                }
            }
            if entity.is_heat || entity.is_vacuum { continue; }

            let extent = if entity.deposit_radii != glam::Vec3::ZERO { entity.deposit_radii * 2.0 } else { glam::Vec3::splat(1.0) };
            aabb_min = aabb_min.min(entity.position - extent);
            aabb_max = aabb_max.max(entity.position + extent);

            // Animated deposit position (skin oscillation, tail wag, jaw).
            let mut deposit_pos = entity.position;
            if !freeze && entity.oscillation_amplitude > 0.0 {
                deposit_pos += entity.surface_normal * entity.oscillation_phase.sin() * entity.oscillation_amplitude;
            }
            if !freeze && (entity.group == GROUP_TAIL || entity.group == GROUP_TAIL_TIP) {
                let time = tick as f32 / 30.0;
                let frequency = std::f32::consts::PI;
                let center_z = FIELD_SIZE as f32 / 2.0 + walk_offset_z;
                let z_frac = ((center_z - entity.position.z) / 24.0).clamp(0.0, 1.0);
                let amplitude = 3.0 * z_frac;
                let phase = time * frequency + z_frac * 2.0;
                deposit_pos.x += amplitude * phase.sin();
            }
            if !freeze && (entity.group == GROUP_JAW || entity.group == GROUP_MOUTH) {
                let time = tick as f32 / 30.0;
                let frequency = std::f32::consts::PI * 0.5;
                let center = FIELD_SIZE as f32 / 2.0;
                let pivot_z = center + 8.0 + walk_offset_z;
                let z_frac = ((entity.position.z - pivot_z) / 8.0).clamp(0.0, 1.0);
                let open_amount = (time * frequency).sin().abs();
                deposit_pos.y -= z_frac * 2.5 * open_amount;
            }
            if !freeze { entity.oscillation_phase += entity.oscillation_freq; }

            let use_gaussian = entity.deposit_radii != glam::Vec3::ZERO;
            let (density_boost, color_boost) = if use_gaussian {
                (40.0 * tune_density, 10.0 * tune_color)
            } else {
                (10.0 * tune_density, 10.0 * tune_color)
            };
            let static_boost = if use_gaussian { 40.0 } else { 10.0 };

            // Visible? (frustum + trie-depth cutoff)
            let drawable = visible && !depth_cull;

            // Consumption mass boost: entities that consume more deposit denser
            let mag = entity.deposit_magnitude * mass_boost;
            let absorbed = 1.0 - entity.pass_through;
            let entity_color = if show_depth {
                depth_col.unwrap_or([0.3, 0.3, 0.3])
            } else { entity.color };
            let total_r = (entity_color[0] * mag + entity.incoming.r * absorbed * entity_color[0] + entity.reemit_r) * color_boost;
            let total_g = (entity_color[1] * mag + entity.incoming.g * absorbed * entity_color[1] + entity.reemit_g) * color_boost;
            let total_b = (entity_color[2] * mag + entity.incoming.b * absorbed * entity_color[2] + entity.reemit_b) * color_boost;
            let total_d = (mag + entity.incoming.density * absorbed) * density_boost;

            self.sources[ent_idx] = Source {
                position: deposit_pos,
                radii: entity.deposit_radii,
                normal: if entity.surface_normal.length_squared() > 0.0 { entity.surface_normal } else { glam::Vec3::Y },
                opacity: entity.deposit_magnitude * static_boost,
                density: total_d,
                color: [total_r, total_g, total_b],
                drawable,
                occluder: true,
            };
        }
        self.aabb_min = aabb_min.max(glam::Vec3::ZERO);
        self.aabb_max = aabb_max.min(glam::Vec3::splat(FIELD_SIZE as f32));
    }
```

Borrowing: `entity` is `&mut self.entities[ent_idx]`; every other read in the loop body comes from locals taken above it, and the final `self.sources[ent_idx] = …` writes a different field, which the borrow checker allows because both are direct field accesses in the same method body.

- [ ] **Step 5: The new `tick()`**

```rust
    /// Run one simulation tick — push-driven pipe propagation, then arrival
    /// at the observer's retina.
    pub fn tick(&mut self, view_proj: glam::Mat4) {
        let refreshed = self.transport(view_proj);
        self.advance_entities();
        let force = refreshed || self.retina_force_relink;
        self.retina_force_relink = false;
        // Borrow split: sources and retina are separate fields.
        let sources = std::mem::take(&mut self.sources);
        self.retina.tick(&sources, view_proj, self.aabb_min, self.aabb_max, force, self.atten_k);
        self.sources = sources;

        if self.tick % 300 == 0 && self.tick > 0 {
            // (trie diagnostics block — move verbatim from tick_grid)
        }
        self.tick += 1;
    }
```

- [ ] **Step 6: Retarget `dump_field_stats`**

Replace the body of `pub fn dump_field_stats(&self)`: call `self.retina.log_stats();` then keep the floor probe but drop the `cell_col` part (log only `avg incoming` under / far). `percentile` stays (it has a test) — mark `#[allow(dead_code)]` if the compiler complains.

- [ ] **Step 7: Run everything**

Run: `cargo test 2>&1 | tail -5 && cargo test --release -- --ignored 2>&1 | tail -8`
Expected: all unit tests pass; the three ignored tests (`walker_group_moves_rigidly`, `cross_links_follow_walker`, `retina_sees_the_dino_and_settles`) pass.

If `retina_sees_the_dino_and_settles` fails on `above > 500`: print `field.retina.stats` and `field.retina.receptors.iter().map(|r| r.density).fold(0., f32::max)` — a density max of 0 means `visible_set` was never true for walkers (check `transport` runs `compute_active_set` before `advance_entities`).

- [ ] **Step 8: Commit**

```bash
git add src/field.rs src/renderer.rs
git commit -m "feat(field): retina-fed tick — transport, advance_entities → Sources; grid tick kept as reference"
```

---

### Task 5: Density-attenuated radiation edges, `ray_blocked` removed, keys 5/6

**Files:**
- Modify: `src/field.rs` — struct, `flatten_edges` (`~785`), `build_radiation_links` (`~679-780`), `refresh_cross_links` (`~815-1000`), Phase 2 in `transport`
- Modify: `src/renderer.rs` (tuning method), `src/main.rs` (keys)

**Interfaces:**
- Consumes: `segment_transmittance`, `SpatialHash`, `Source`.
- Produces:
  - `DiffField.edge_atten: Vec<f32>` parallel to `edge_gammas`
  - `fn occluder_sources(&self) -> Vec<Source>` (positions, opacity, occluder=!heat&&!vacuum, drawable=false)
  - `fn compute_edge_atten(&mut self, only_cross: bool)`
  - `Renderer::tune_atten_scale(factor: f32)`

- [ ] **Step 1: Write the failing tests**

Append to `mod tests` in `src/field.rs`:

```rust
    #[test]
    #[ignore] // builds the full 512³ field (~2 GB, slow) — run: cargo test --release -- --ignored
    fn long_edges_are_attenuated_short_edges_are_not() {
        let field = DiffField::new();
        let cd_sq = field.link_connect_dist * field.link_connect_dist;
        let mut long_below_one = 0u32;
        let mut long_total = 0u32;
        for (i, e) in field.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                let t = field.edge_targets[k];
                let d_sq = e.position.distance_squared(field.entities[t].position);
                let a = field.edge_atten[k];
                assert!(a > 0.0 && a <= 1.0, "edge {}→{} atten {}", i, t, a);
                if d_sq < cd_sq {
                    assert_eq!(a, 1.0, "connection edge {}→{} attenuated", i, t);
                } else {
                    long_total += 1;
                    if a < 0.999 { long_below_one += 1; }
                }
            }
        }
        assert!(long_total > 1000, "expected many radiation edges, got {}", long_total);
        assert!(long_below_one > 0, "no radiation edge is attenuated — occluders ignored");
        assert!(long_below_one < long_total, "every radiation edge is attenuated — self-shadowing");
    }

    #[test]
    #[ignore] // builds the full 512³ field (~2 GB, slow) — run: cargo test --release -- --ignored
    fn floor_under_the_dino_receives_less_light() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        for _ in 0..60 { field.tick(vp); }
        let mut wmin = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut wmax = glam::Vec3::ZERO;
        for e in field.entities.iter().filter(|e| e.is_walker) {
            wmin = wmin.min(e.position);
            wmax = wmax.max(e.position);
        }
        let center = (wmin + wmax) * 0.5;
        let (mut under, mut un, mut far, mut fn_) = (0.0f64, 0u32, 0.0f64, 0u32);
        for e in field.entities.iter().filter(|e| e.group == GROUP_FLOOR) {
            let inside = e.position.x >= wmin.x && e.position.x <= wmax.x && e.position.z >= wmin.z && e.position.z <= wmax.z;
            let dx = e.position.x - center.x;
            let dz = e.position.z - center.z;
            if inside { under += e.incoming.density as f64; un += 1; }
            else if dx * dx + dz * dz > 400.0 { far += e.incoming.density as f64; fn_ += 1; }
        }
        let (under, far) = (under / un.max(1) as f64, far / fn_.max(1) as f64);
        assert!(un > 0 && fn_ > 0);
        assert!(under < 0.8 * far, "no shadow: under={:.3} far={:.3}", under, far);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --release -- --ignored long_edges 2>&1 | tail -5`
Expected: compile error — no `edge_atten`.

- [ ] **Step 3: Implement**

Struct: add `edge_atten: Vec<f32>,` after `edge_gammas`; in `new()`: `edge_atten: Vec::new(),`. In `flatten_edges`, after `self.edge_gammas = vec![1.0; total];` add `self.edge_atten = vec![1.0; total];`.

Helper + compute:

```rust
    /// Static occluder set for edge attenuation: every solid entity at its
    /// current position. Nothing is drawable here — this is only for τ.
    fn occluder_sources(&self) -> Vec<crate::retina::Source> {
        self.entities.iter().map(|e| crate::retina::Source {
            position: e.position,
            radii: e.deposit_radii,
            normal: glam::Vec3::Y,
            opacity: e.deposit_magnitude * if e.deposit_radii != glam::Vec3::ZERO { 40.0 } else { 10.0 },
            density: 0.0,
            color: [0.0; 3],
            drawable: false,
            occluder: !e.is_heat && !e.is_vacuum,
        }).collect()
    }

    /// τ for every edge longer than link_connect_dist (radiation links);
    /// connection edges stay 1.0. `only_cross` restricts to walker↔world
    /// edges (what a cross-link refresh rebuilt) — internal edges keep the
    /// values carried over by the caller.
    fn compute_edge_atten(&mut self, only_cross: bool) {
        use crate::retina::{segment_transmittance, SpatialHash};
        let t0 = std::time::Instant::now();
        let sources = self.occluder_sources();
        let hash = SpatialHash::build(&sources);
        let cd_sq = self.link_connect_dist * self.link_connect_dist;
        let k = self.atten_k;
        // (source, edge index, target) for every edge
        let mut edges: Vec<(usize, usize, usize)> = Vec::with_capacity(self.edge_targets.len());
        for (i, e) in self.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for kk in start..start + e.edge_count as usize {
                edges.push((i, kk, self.edge_targets[kk]));
            }
        }
        let walker: Vec<bool> = self.entities.iter().map(|e| e.is_walker).collect();
        let atten: Vec<(usize, f32)> = edges.par_iter().filter_map(|&(i, kk, t)| {
            if only_cross && walker[i] == walker[t] { return None; }
            let a = sources[i].position;
            let b = sources[t].position;
            if a.distance_squared(b) < cd_sq { return Some((kk, 1.0)); }
            Some((kk, segment_transmittance(&sources, &hash, a, b, &[i, t], k)))
        }).collect();
        let mut attenuated = 0usize;
        for (kk, a) in atten {
            self.edge_atten[kk] = a;
            if a < 0.999 { attenuated += 1; }
        }
        log::info!("Edge attenuation ({}): {} edges < 1, {:.2} ms",
            if only_cross { "cross" } else { "all" }, attenuated, t0.elapsed().as_secs_f64() * 1000.0);
    }
```

`build_radiation_links`: delete the `block_grid` construction, the `ray_blocked` call (push candidates unconditionally), `blocked_count`, and the "blocked by LOS" wording in the log. After the gamma loop at the end add `self.compute_edge_atten(false);`. Then delete `fn ray_blocked` entirely.

`refresh_cross_links`: delete `block_grid`, `block_cell`, `block_radius_sq`, and the `ray_blocked` gate. Carry retained edges' attenuation positionally exactly like deposits: add `let mut temp_atten: Vec<Vec<f32>> = vec![Vec::new(); n];`, push `self.edge_atten[k]` next to each `temp_deposits[i].push(...)`, and in the restore loop set `self.edge_atten[k] = temp_atten[i][off];` for `off < retained`. After `self.build_reverse_edges();` add `self.compute_edge_atten(true);`.

`build_connections` calls `flatten_edges` before `link_connect_dist` is set — that's fine: `compute_edge_atten` is only called from the radiation builder, and `new()` must set `field.link_connect_dist = connect_dist;` **before** `field.build_radiation_links(...)`. Move those two assignment lines above the `build_radiation_links` call in `new()`.

Phase 2 in `transport`: add `let edge_atten = &self.edge_atten;` next to `let edge_gammas = &self.edge_gammas;` and change the weight line to:
```rust
                let mut w = edge_gammas[k] * distance_factors[idx] * edge_atten[k];
```

Renderer:
```rust
    pub fn tune_atten_scale(&mut self, factor: f32) {
        self.diff_field.atten_k = crate::field::scale_tune(self.diff_field.atten_k, factor);
        self.diff_field.retina_force_relink = true;
        self.diff_field.compute_edge_atten_public();
        self.log_tuning();
    }
```
with `log_tuning` printing `"Tuning: density ×{:.4}, color ×{:.4}, atten_k ×{:.4}"`. Add to `DiffField`: `pub fn compute_edge_atten_public(&mut self) { self.compute_edge_atten(false); }` (kept separate so the private fn signature stays free to change).

`main.rs`: after the `Digit4` arm:
```rust
                            KeyCode::Digit5 => { state.renderer.tune_atten_scale(0.5); }
                            KeyCode::Digit6 => { state.renderer.tune_atten_scale(2.0); }
```

- [ ] **Step 4: Run everything**

Run: `cargo test 2>&1 | tail -3 && cargo test --release -- --ignored 2>&1 | tail -8`
Expected: all pass. Look at the `Edge attenuation (all)` log line from `DiffField::new()` in the ignored run (`RUST_LOG=info`): it should be low tens of ms; `(cross)` low single-digit ms.

If `floor_under_the_dino_receives_less_light` fails with `under ≈ far`: the radiation cap (10 closest per entity) may be filled by dino-internal edges now that LOS no longer prunes them — raise `max_radiation` from 10 to 14 in both builders (the remedy the spec names) and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/field.rs src/renderer.rs src/main.rs
git commit -m "feat: density-attenuated radiation edges via segment_transmittance; remove ray_blocked; keys 5/6"
```

---

### Task 6: Shader, renderer upload, keys H/7/8

**Files:**
- Create: `shaders/retina.wgsl`
- Modify: `src/renderer.rs`, `src/main.rs`

**Interfaces:**
- Consumes: `diff_field.retina.{width, height, receptors, dirty}`, `RETINA_ISO`.
- Produces: `Renderer::scale_retina(factor: f32)`; retina textures `retina_dc` (density, r, g, b — color already divided by density) and `retina_nd` (unit normal xyz, mean depth), both `Rgba16Float`, bilinear-sampled.

- [ ] **Step 1: Write `shaders/retina.wgsl`**

Copy `shaders/field_sample.wgsl` to `shaders/retina.wgsl`, then:

Replace the header comment with:
```wgsl
// retina.wgsl — display the observer's receptors.
//
// No marching. The CPU retina already holds, per receptor, everything that
// arrived along entity pipes. This shader thresholds density, shades the
// surface with the arrived normal, and composites over the procedural sky.
```

Replace the `@group(1)` bindings and delete `sample_field`, `sample_density`, `compute_normal`:
```wgsl
// --- Retina textures ---
// dc: (density, r, g, b)  — color is Σ density·color / Σ density
// nd: (nx, ny, nz, depth) — unit normal, density-weighted eye distance

@group(1) @binding(0)
var retina_dc: texture_2d<f32>;
@group(1) @binding(1)
var retina_nd: texture_2d<f32>;
@group(1) @binding(2)
var retina_sampler: sampler;

const RETINA_ISO: f32 = 0.3;
```

Replace everything in `fs_main` from `// --- March through the field ---` through the closing of the `if t_enter <= t_exit …` block with:
```wgsl
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    let dc = textureSample(retina_dc, retina_sampler, in.uv);
    let density = dc.r;
    if density >= RETINA_ISO {
        let nd = textureSample(retina_nd, retina_sampler, in.uv);
        var norm_color = dc.gba;
        let n_len = length(nd.xyz);
        var normal = select(vec3<f32>(0.0, 1.0, 0.0), nd.xyz / max(n_len, 1e-5), n_len > 1e-4);
        // Surface point for texture coordinates: along this pixel's ray at the arrived depth
        let sample_pos = u.observer_pos + ray_dir * nd.w;

        // --- Reptile skin texture --- (verbatim from field_sample.wgsl)
        let greenness = norm_color.g / max(norm_color.r + norm_color.g + norm_color.b, 0.01);
        let is_creature = step(0.35, greenness) * step(0.1, norm_color.g);
        if is_creature > 0.5 {
            // ... the whole creature block, unchanged ...
        }

        // Lambert + rim + specular — verbatim
        let n_dot_l = max(dot(normal, sun_dir), 0.0);
        let ambient = 0.10;
        let diffuse = ambient + (1.0 - ambient) * n_dot_l;
        let n_dot_v = abs(dot(normal, -ray_dir));
        let rim = pow(1.0 - n_dot_v, 3.0) * 0.3;
        let half_vec = normalize(sun_dir - ray_dir);
        let n_dot_h = max(dot(normal, half_vec), 0.0);
        let specular = pow(n_dot_h, 32.0) * 0.25 * is_creature;
        accumulated_color = norm_color * (diffuse + rim) + vec3<f32>(1.0, 0.95, 0.8) * specular;
        accumulated_alpha = 1.0;
    }
```
Keep the sky/sun-glow/vignette/ACES/gamma tail unchanged. The `Uniforms` struct stays identical (the shader still needs `inv_view_proj`, `observer_pos`, `observer_speed`; `field_size`/`aabb_*` are unused but harmless).

- [ ] **Step 2: Renderer**

In `src/renderer.rs`:

1. Replace the `field_texture` / `field_bind_group` / `upload_buf` fields with:
```rust
    retina_dc: wgpu::Texture,
    retina_nd: wgpu::Texture,
    retina_bind_group: wgpu::BindGroup,
    retina_layout: wgpu::BindGroupLayout,
    retina_sampler: wgpu::Sampler,
    retina_size: (u32, u32),
    upload_buf: Vec<u16>,
```
2. Add a constructor helper (free function) used by `new()` and on resolution change:
```rust
fn create_retina_textures(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::Texture, wgpu::BindGroup) {
    let make = |label: &str| device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let dc = make("RetinaDC");
    let nd = make("RetinaND");
    let dc_view = dc.create_view(&Default::default());
    let nd_view = nd.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("RetinaBindGroup"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dc_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&nd_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    });
    (dc, nd, bind_group)
}
```
3. In `new()`: the bind group layout has three entries — two `Texture { view_dimension: D2, sample_type: Float { filterable: true }, multisampled: false }` at bindings 0 and 1 and the `Sampler(Filtering)` at 2. Sampler: `ClampToEdge`, `Linear` mag/min (same as today minus `address_mode_w`). Shader source: `include_str!("../shaders/retina.wgsl")`. Create textures with `create_retina_textures(&device, &layout, &sampler, RETINA_W, RETINA_H)` where the dims come from `diff_field.retina` (construct `diff_field` first). `upload_buf: Vec::new()`.
4. Replace the slab upload in `render()` with:
```rust
        // Resolution changed (keys 7/8)? Recreate the textures.
        let (rw, rh) = (self.diff_field.retina.width, self.diff_field.retina.height);
        if (rw, rh) != self.retina_size {
            let (dc, nd, bg) = create_retina_textures(&self.device, &self.retina_layout, &self.retina_sampler, rw, rh);
            self.retina_dc = dc;
            self.retina_nd = nd;
            self.retina_bind_group = bg;
            self.retina_size = (rw, rh);
            self.diff_field.retina.dirty = true;
        }
        if self.diff_field.retina.dirty {
            // write_texture needs bytes_per_row % 256 == 0 (true at 320 wide,
            // not at every resolution keys 7/8 can produce) → padded rows.
            let row_bytes = ((rw * 8 + 255) / 256) * 256;
            let stride = (row_bytes / 2) as usize; // u16 per padded row
            let total = stride * rh as usize;
            self.upload_buf.resize(total * 2, 0);
            let (dc_buf, nd_buf) = self.upload_buf.split_at_mut(total);
            let f = |x: f32| half::f16::from_f32(x).to_bits();
            for (i, r) in self.diff_field.retina.receptors.iter().enumerate() {
                let (x, y) = ((i % rw as usize), (i / rw as usize));
                let o = y * stride + x * 4;
                let d = r.density.clamp(0.0, 60000.0);
                let inv = if r.density > 1e-6 { 1.0 / r.density } else { 0.0 };
                let nl = r.normal.length();
                let nrm = if nl > 1e-6 { r.normal / nl } else { glam::Vec3::Y };
                dc_buf[o] = f(d);
                dc_buf[o + 1] = f((r.color[0] * inv).min(60000.0));
                dc_buf[o + 2] = f((r.color[1] * inv).min(60000.0));
                dc_buf[o + 3] = f((r.color[2] * inv).min(60000.0));
                nd_buf[o] = f(nrm.x);
                nd_buf[o + 1] = f(nrm.y);
                nd_buf[o + 2] = f(nrm.z);
                nd_buf[o + 3] = f((r.depth * inv).min(60000.0));
            }
            for (tex, buf) in [(&self.retina_dc, &*dc_buf), (&self.retina_nd, &*nd_buf)] {
                self.queue.write_texture(
                    wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    bytemuck::cast_slice(buf),
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(rh) },
                    wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
                );
            }
            self.diff_field.retina.dirty = false;
        }
```
5. `set_bind_group(1, &self.retina_bind_group, &[])`. Remove `last_uploaded_tick` and the `FIELD_SIZE` import if unused.
6. Add:
```rust
    pub fn scale_retina(&mut self, factor: f32) {
        let r = &mut self.diff_field.retina;
        let w = (r.width as f32 * factor).round() as u32;
        let h = (r.height as f32 * factor).round() as u32;
        r.resize(w, h);
        log::info!("Retina resolution: {}x{}", r.width, r.height);
    }
```

`main.rs`: after the `Digit6` arm:
```rust
                            KeyCode::Digit7 => { state.renderer.scale_retina(0.5); }
                            KeyCode::Digit8 => { state.renderer.scale_retina(2.0); }
```
`H` already calls `dump_field_stats`, which Task 4 retargeted.

- [ ] **Step 3: Build and smoke run**

Run: `cargo build --release 2>&1 | tail -3` then `RUST_LOG=info cargo run --release 2>&1 | head -40` (Ctrl-C after the window shows). The `H` key must log the `Retina 320x180: …` line; the title bar shows FPS.
Expected: builds; no wgpu validation errors in the log (a validation error names the binding/layout mismatch — fix the layout, not the shader).

- [ ] **Step 4: Commit**

```bash
git add shaders/retina.wgsl src/renderer.rs src/main.rs
git commit -m "feat(render): retina display shader + 2D receptor textures; keys 7/8 resolution"
```

---

### Task 7: CHECKPOINT — human visual verification (HUMAN, not a subagent task)

Cannot be done headlessly (no swapchain capture, no synthetic keys).

- [ ] User runs `RUST_LOG=info cargo run --release` and checks: dino silhouette present and crisp; jaw opening reads; floor and rock visible; shadow under the dino that **follows it** as it paces; `5`/`6` deepen/lighten the shadow; `1`–`4` behave; `7`/`8` change resolution without artifacts; `H` prints sane stats (pipes sent → 0 when nothing moves is not expected here — the dino always moves — but should be far below `pipes total`).
- [ ] User reports FPS and any of: bloated silhouette (raise density tune or `RETINA_ISO`), mushy self-overlap (raise `atten_k`), missing body parts (lower `atten_k`).
- [ ] If the picture is wrong in a way tuning does not fix: STOP, do not start Task 8, revisit the spec's Risks section with the user.

---

### Task 8: Delete the grid path; docs

**Files:**
- Modify: `src/field.rs`, `src/renderer.rs`, `src/main.rs`, `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`
- Delete: `shaders/field_sample.wgsl`

- [ ] **Step 1: Remove grid code**

In `src/field.rs` delete: `FIELD_CELLS`, `FieldCell` + its `Default`, the `cells` field and its init, `dirty_slabs`, `prev_deposit_idx` (Entity field + init), `deposit_queue`, `fn index`, `fn in_bounds` (if only grid code used them — `advance_entities` bounces on `FIELD_SIZE` directly), `tick_grid` entirely, `as_bytes`, and the grid-only branch of `dump_field_stats`. `FIELD_SIZE` stays (world extent for bounce, scene placement, AABB clamp). The header comment becomes:

```rust
// Diff Field — entities and the pipes between them.
//
// Light propagates through CONNECTIONS between entities, not through any
// grid. The observer is a receptor array (retina.rs) fed by the same kind
// of pipes. Nothing is cached in space; what you see is what arrived.
```

In `src/renderer.rs` delete any remaining `FIELD_SIZE` use in `Uniforms` init only if the shader no longer declares `field_size`/`aabb_*` — otherwise leave the uniforms as they are (they're cheap and keep the struct layout). Delete the header comment's "Just a 3D texture" wording; describe the receptor upload.

In `src/main.rs` update the header comment (no "field spreads at c" line) — the retina is the observer.

Delete `shaders/field_sample.wgsl`.

Update the ignored-test comments: `// builds the full demo scene (slow) — run: cargo test --release -- --ignored` (no 2 GB any more).

- [ ] **Step 2: Build and test**

Run: `cargo build --release 2>&1 | grep -E "warning|error" | head; cargo test 2>&1 | tail -3; cargo test --release -- --ignored 2>&1 | tail -8`
Expected: no warnings about dead grid code; all tests pass; the ignored suite is now fast (no 2 GB allocation) — note the time.

- [ ] **Step 3: Docs**

`README.md` — controls table gains:
```
| H | Dump retina + floor-probe stats to the log |
| `1` / `2` | Halve / double density tuning |
| `3` / `4` | Halve / double color tuning |
| `5` / `6` | Halve / double occlusion strength (`atten_k`) |
| `7` / `8` | Halve / double receptor resolution |
```
and "What You're Seeing" rewritten: the screen is the receptor array; each receptor is the sum of what entities delivered along pipes, attenuated by the density between them and the eye; the shader thresholds and shades it.

`ARCHITECTURE.md` — Core Principle: replace the grid paragraph with the retina (receptors are the observer's state; pipes carry deltas; occlusion = transmittance). Data Model: delete `FieldCell`; add `Receptor`, `Source`, pipes SoA, `edge_atten`. Per-Tick Pipeline table: rows become Cross-link refresh / Active set / Atmosphere / Phase 1 / Consumption / Phase 2 / Advance entities / Relink (conditional) / Phase 3′ arrive. GPU Upload: two `W×H` Rgba16Float textures, whole, when dirty. GPU Render: threshold + shade + composite, no march. Demo Scene: shadow paragraph → "radiation edges longer than `link_connect_dist` carry `edge_atten`, the transmittance of the density between their endpoints; the floor under the dino receives less because the body sits on those segments — the shadow is an arrival deficit that moves with the walker."

`ROADMAP.md` — add "Per-pipe transmittance (retina Approach 2)" and "Hierarchical receptors (Approach 3)" as follow-ups.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove the grid retina; docs for the receptor retina"
```

---

## Final Verification

- [ ] `cargo test` and `cargo test --release -- --ignored` green.
- [ ] `RUST_LOG=info cargo run --release`: FPS at or above the grid path's; `Edge attenuation (cross)` lines low single-digit ms; `Retina … relink` low single-digit ms while the observer is still, tolerable while flying.
- [ ] Human: silhouette crisp; jaw reads; shadow follows the pacing dino; `5`/`6` visibly change the shadow after the next refresh/relink; `H` shows `pipes sent` far below `pipes total` on a mostly still scene.
- [ ] Branch `retina` merged to `main`; `desaturation-attenuation` deleted (its two useful commits were cherry-picked in Task 1; the rest is superseded).
