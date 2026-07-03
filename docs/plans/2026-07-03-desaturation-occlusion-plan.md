# Desaturation + Density-Attenuated Pipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument and desaturate the field so the jaw and floor contrast read (B1), and replace the startup LOS oracle with per-edge transmittance derived from the deposited density itself so the shadow is an arrival deficit that follows the dino (B2′).

**Architecture:** B1 adds an on-demand stats dump (`H`) and live tuning multipliers (`1`–`4`) so the user calibrates by eye; chosen values get baked into the boost constants at a mid-plan checkpoint. B2′ adds `edge_atten: Vec<f32>` multiplied into the Phase 2 push weight for edges longer than `link_connect_dist`, computed by integrating grid density along each edge (with end margins), initialized at tick 15 and recomputed at every cross-link refresh; `ray_blocked` is deleted.

**Tech Stack:** Rust 2021, existing `field.rs`/`renderer.rs`/`main.rs`. No new dependencies.

**Spec:** `docs/plans/2026-07-03-desaturation-occlusion-design.md` (approved, B2′ revision).

## Global Constraints

- Tuning multipliers (`tune_density`, `tune_color`, `atten_k`) clamp to [1/1024, 1024]; every tuning keypress logs all current tuning values.
- `ATTEN_THRESHOLD = 0.6`; `ATTEN_K_DEFAULT = 0.5`; attenuation init at tick `ATTEN_INIT_TICK = 15`; end margins 1.5 cells; ~1 sample/cell.
- Attenuation applies ONLY to edges longer than `link_connect_dist`; connection-edge transport, the shader, and the decay constant are untouched.
- `ray_blocked` and all its call sites are removed (no LOS oracle anywhere).
- Keys: `H` stats dump; `1`/`2` tune_density ÷2/×2; `3`/`4` tune_color ÷2/×2; `5`/`6` atten_k ÷2/×2. No collisions with existing T, I, `[`, `]`, `-`, `=`.
- Task 3 is a HUMAN CHECKPOINT (user calibration session) — not a subagent task; Task 4 bakes the numbers the user reports.
- Build `cargo build --release`; fast tests `cargo test`; ignored field tests `cargo test --release -- --ignored`.

## File Structure

- `src/field.rs` — stats dump, tuning fields, `segment_attenuation` (pure), `edge_atten` + recompute, `deposit_extents` helper, `ray_blocked` removal, tests.
- `src/renderer.rs` — pass-through methods for `H`/`1`–`6`.
- `src/main.rs` — key bindings.
- `README.md` / `ARCHITECTURE.md` — controls + attenuation paragraph (folded into Task 5).

---

### Task 1: B1a — field stats dump (key `H`)

**Files:**
- Modify: `src/field.rs` (pure helper + `dump_field_stats`), `src/renderer.rs:~476` (pass-through), `src/main.rs:~152` (key)
- Test: `#[cfg(test)] mod tests` in `src/field.rs`

**Interfaces:**
- Produces: `fn percentile(sorted: &[f32], p: f32) -> f32` (module-level, pure); `DiffField::dump_field_stats(&self)` (public); `Renderer::dump_field_stats(&self)`.

- [ ] **Step 1: Write the failing percentile tests**

Add to the `tests` module in `src/field.rs`:

```rust
    #[test]
    fn percentile_picks_correct_ranks() {
        let sorted: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert_eq!(percentile(&sorted, 0.5), 50.0);
        assert_eq!(percentile(&sorted, 0.9), 90.0);
        assert_eq!(percentile(&sorted, 0.99), 99.0);
        assert_eq!(percentile(&sorted, 1.0), 100.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[7.0], 0.5), 7.0);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test percentile`
Expected: COMPILE ERROR — `percentile` not found.

- [ ] **Step 3: Implement the helper and the dump**

Module level in `src/field.rs` (before `impl DiffField`):

```rust
/// Rank percentile of a sorted slice (p in 0..=1). Empty slice → 0.0.
fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f32 * p).ceil() as usize).max(1) - 1;
    sorted[idx.min(sorted.len() - 1)]
}
```

Method on `DiffField` (place after `dump_trie_info`-adjacent public methods, e.g. right before `tick()`):

```rust
    /// On-demand diagnostics (key H): saturation histograms over the active
    /// AABB and a floor-contrast probe. Answers "where does the 50-cap
    /// bite?" and "which transport path carries the ground pattern?".
    pub fn dump_field_stats(&self) {
        let fs = FIELD_SIZE as usize;
        let x0 = self.aabb_min.x.max(0.0) as usize;
        let x1 = (self.aabb_max.x as usize + 1).min(fs);
        let y0 = self.aabb_min.y.max(0.0) as usize;
        let y1 = (self.aabb_max.y as usize + 1).min(fs);
        let z0 = self.aabb_min.z.max(0.0) as usize;
        let z1 = (self.aabb_max.z as usize + 1).min(fs);

        let mut density = Vec::new();
        let mut colors = Vec::new();
        let mut capped = 0u64;
        let mut nonzero = 0u64;
        for z in z0..z1 {
            for y in y0..y1 {
                let row = z * fs * fs + y * fs;
                for x in x0..x1 {
                    let c = &self.cells[row + x];
                    if c.density <= 0.0 { continue; }
                    nonzero += 1;
                    density.push(c.density);
                    colors.push(c.color_r.max(c.color_g).max(c.color_b));
                    if c.density >= 49.5 { capped += 1; }
                }
            }
        }
        density.sort_by(|a, b| a.partial_cmp(b).unwrap());
        colors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        log::info!(
            "Field stats (AABB {}x{}x{}, {} nonzero cells): density p50={:.3} p90={:.3} p99={:.3} max={:.3}, {:.1}% at cap; color(max-ch) p50={:.3} p90={:.3} p99={:.3} max={:.3}",
            x1 - x0, y1 - y0, z1 - z0, nonzero,
            percentile(&density, 0.5), percentile(&density, 0.9),
            percentile(&density, 0.99), percentile(&density, 1.0),
            100.0 * capped as f64 / nonzero.max(1) as f64,
            percentile(&colors, 0.5), percentile(&colors, 0.9),
            percentile(&colors, 0.99), percentile(&colors, 1.0),
        );

        // Floor contrast probe: tiles under the walker footprint vs far tiles
        let mut wmin = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut wmax = glam::Vec3::ZERO;
        for e in self.entities.iter().filter(|e| e.is_walker) {
            wmin = wmin.min(e.position);
            wmax = wmax.max(e.position);
        }
        let center = (wmin + wmax) * 0.5;
        let (mut under_inc, mut under_col, mut under_n) = (0.0f64, 0.0f64, 0u32);
        let (mut far_inc, mut far_col, mut far_n) = (0.0f64, 0.0f64, 0u32);
        for e in self.entities.iter().filter(|e| e.group == GROUP_FLOOR) {
            let cell_col = {
                let ix = e.position.x as i32;
                let iy = e.position.y as i32;
                let iz = e.position.z as i32;
                if Self::in_bounds(ix, iy, iz) {
                    let c = &self.cells[Self::index(ix as u32, iy as u32, iz as u32)];
                    c.color_r.max(c.color_g).max(c.color_b) as f64
                } else { 0.0 }
            };
            let under = e.position.x >= wmin.x && e.position.x <= wmax.x
                && e.position.z >= wmin.z && e.position.z <= wmax.z;
            let dx = e.position.x - center.x;
            let dz = e.position.z - center.z;
            if under {
                under_inc += e.incoming.density as f64;
                under_col += cell_col;
                under_n += 1;
            } else if dx * dx + dz * dz > 400.0 {
                far_inc += e.incoming.density as f64;
                far_col += cell_col;
                far_n += 1;
            }
        }
        log::info!(
            "Floor probe: under dino n={} avg incoming={:.3} avg cell color={:.3} | far (>20 cells) n={} avg incoming={:.3} avg cell color={:.3}",
            under_n, under_inc / under_n.max(1) as f64, under_col / under_n.max(1) as f64,
            far_n, far_inc / far_n.max(1) as f64, far_col / far_n.max(1) as f64,
        );
    }
```

`src/renderer.rs`, after `double_time_lapse`:

```rust
    pub fn dump_field_stats(&self) {
        self.diff_field.dump_field_stats();
    }
```

`src/main.rs`, after the `KeyCode::Equal` arm:

```rust
                            KeyCode::KeyH => {
                                state.renderer.dump_field_stats();
                            }
```

- [ ] **Step 4: Run tests and build**

Run: `cargo test` — expected `16 passed; 2 ignored`.
Run: `cargo build --release` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/field.rs src/renderer.rs src/main.rs
git commit -m "feat: field saturation stats + floor contrast probe on H key"
```

---

### Task 2: B1b — live tuning keys

**Files:**
- Modify: `src/field.rs` (fields + pass-3b application + clamp helper), `src/renderer.rs` (4 methods), `src/main.rs` (4 keys)
- Test: `#[cfg(test)] mod tests` in `src/field.rs`

**Interfaces:**
- Produces: `pub tune_density: f32`, `pub tune_color: f32` on `DiffField` (default 1.0); module fn `fn scale_clamped(v: f32, factor: f32) -> f32`; `Renderer::{tune_density_scale, tune_color_scale}(&mut self, factor: f32)`.
- Task 5 extends the same logging line with `atten_k`.

- [ ] **Step 1: Write the failing clamp test**

```rust
    #[test]
    fn scale_tune_stays_in_range() {
        assert_eq!(scale_tune(1.0, 2.0), 2.0);
        assert_eq!(scale_tune(1.0, 0.5), 0.5);
        assert_eq!(scale_tune(1024.0, 2.0), 1024.0);
        let lo = scale_tune(1.0 / 1024.0, 0.5);
        assert!((lo - 1.0 / 1024.0).abs() < 1e-9);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test scale_tune`
Expected: COMPILE ERROR — `scale_tune` not found.

- [ ] **Step 3: Implement**

Module level in `src/field.rs`:

```rust
/// Multiply a live-tuning value (keys 1-6), clamped to [1/1024, 1024].
pub fn scale_tune(v: f32, factor: f32) -> f32 {
    (v * factor).clamp(1.0 / 1024.0, 1024.0)
}
```

`struct DiffField` (after `deposit_queue`):

```rust
    /// Live calibration multipliers (keys 1-4): applied on top of the
    /// hardcoded deposit boosts. Baked into constants after calibration.
    pub tune_density: f32,
    pub tune_color: f32,
```

`DiffField::new()` literal: `tune_density: 1.0,` and `tune_color: 1.0,`.

In `tick()`, just before the Phase 3a comment block (next to the `walk_delta` lines), capture locals:

```rust
        let tune_density = self.tune_density;
        let tune_color = self.tune_color;
```

In pass 3b, change:

```rust
            let (density_boost, color_boost) = if is_body { (40.0, 10.0) } else { (10.0, 10.0) };
```

to:

```rust
            let (density_boost, color_boost) = if is_body {
                (40.0 * tune_density, 10.0 * tune_color)
            } else {
                (10.0 * tune_density, 10.0 * tune_color)
            };
```

(Pass 3b runs inside `tick()` where the locals are in scope; vacuum scatter is deliberately not scaled — its densities are orders of magnitude below iso.)

`src/renderer.rs`, after `dump_field_stats`:

```rust
    pub fn tune_density_scale(&mut self, factor: f32) {
        self.diff_field.tune_density = crate::field::scale_tune(self.diff_field.tune_density, factor);
        self.log_tuning();
    }

    pub fn tune_color_scale(&mut self, factor: f32) {
        self.diff_field.tune_color = crate::field::scale_tune(self.diff_field.tune_color, factor);
        self.log_tuning();
    }

    fn log_tuning(&self) {
        log::info!(
            "Tuning: density ×{:.4}, color ×{:.4}",
            self.diff_field.tune_density, self.diff_field.tune_color
        );
    }
```

`src/main.rs`, after the `KeyCode::KeyH` arm:

```rust
                            KeyCode::Digit1 => {
                                state.renderer.tune_density_scale(0.5);
                            }
                            KeyCode::Digit2 => {
                                state.renderer.tune_density_scale(2.0);
                            }
                            KeyCode::Digit3 => {
                                state.renderer.tune_color_scale(0.5);
                            }
                            KeyCode::Digit4 => {
                                state.renderer.tune_color_scale(2.0);
                            }
```

- [ ] **Step 4: Run tests and build**

Run: `cargo test` — expected `17 passed; 2 ignored`.
Run: `cargo build --release` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/field.rs src/renderer.rs src/main.rs
git commit -m "feat: live density/color tuning keys (1-4)"
```

---

### Task 3: CHECKPOINT — user calibration session (HUMAN, not a subagent task)

The controller hands the app to the user:

- [ ] User runs `cargo run --release`, presses `H` for baseline stats, then dials `1`/`2` (density) and `3`/`4` (color) until: jaw opening reads, silhouette tightens toward 1× radii, floor contrast visible, feet/arms and heat-darkened entities do NOT vanish.
- [ ] User reports the final logged values (`Tuning: density ×D, color ×C`) plus an `H` dump at those settings.
- [ ] Controller records D and C for Task 4. If no acceptable setting exists (e.g. jaw needs the retune floor where feet vanish), STOP and revisit the design (mouth-carver Option A returns to the table).

---

### Task 4: Bake the calibrated constants

**Files:**
- Modify: `src/field.rs` (pass-3b boost line)

**Interfaces:**
- Consumes: D (density multiplier) and C (color multiplier) reported by the user in Task 3.

- [ ] **Step 1: Fold D and C into the constants**

In pass 3b, replace the tuned boost line with the products written as literals — e.g. if the user reports D=0.0156 (=1/64) and C=0.25, then `40.0 × D = 0.625`, `10.0 × C = 2.5`, `10.0 × D = 0.156`:

```rust
            // Calibrated 2026-07-03 (was body 40/10, floor 10/10 before the
            // desaturation retune — see docs/plans/2026-07-03-desaturation-occlusion-design.md)
            let (density_boost, color_boost) = if is_body {
                (<40.0×D as literal> * tune_density, <10.0×C as literal> * tune_color)
            } else {
                (<10.0×D as literal> * tune_density, <10.0×C as literal> * tune_color)
            };
```

The `tune_*` multipliers stay in the expression with their defaults of 1.0 — the keys keep working relative to the new baseline.

- [ ] **Step 2: Verify with the user's numbers**

Run: `cargo build --release`, `cargo test` (17 passed), then a smoke run — the app must look identical to the user's calibrated state at default tuning (×1.0/×1.0). Ask the user to confirm.

- [ ] **Step 3: Commit**

```bash
git add src/field.rs
git commit -m "feat: bake calibrated deposit boosts (desaturation retune)"
```

---

### Task 5: B2′ — density-attenuated pipes (+ deposit_extents helper, − ray_blocked, docs)

**Files:**
- Modify: `src/field.rs` (constants, `segment_attenuation`, `edge_atten`, recompute, Phase 2 multiply, `ray_blocked` removal, `deposit_extents`), `src/renderer.rs` (atten_k methods + logging), `src/main.rs` (keys 5/6), `README.md` (controls), `ARCHITECTURE.md` (attenuation paragraph)
- Test: unit tests + one new ignored integration test in `src/field.rs`

**Interfaces:**
- Consumes: `scale_tune` (Task 2), `link_connect_dist` (existing), the refresh trigger block at the top of `tick()` (existing).
- Produces: `fn segment_attenuation(a: glam::Vec3, b: glam::Vec3, k: f32, threshold: f32, sample: impl Fn(i32, i32, i32) -> f32) -> f32`; `fn deposit_extents(radii: glam::Vec3) -> (i32, i32, i32)`; fields `edge_atten: Vec<f32>`, `pub atten_k: f32` on `DiffField`; `fn recompute_edge_attenuation(&mut self)`; constants `ATTEN_THRESHOLD: f32 = 0.6`, `ATTEN_K_DEFAULT: f32 = 0.5`, `ATTEN_INIT_TICK: u64 = 15`.

- [ ] **Step 1: Write the failing unit tests**

```rust
    #[test]
    fn segment_attenuation_empty_space_is_one() {
        let a = glam::Vec3::new(10.0, 10.0, 10.0);
        let b = glam::Vec3::new(30.0, 10.0, 10.0);
        assert_eq!(segment_attenuation(a, b, 0.5, 0.6, |_, _, _| 0.0), 1.0);
    }

    #[test]
    fn segment_attenuation_dense_wall_blocks() {
        let a = glam::Vec3::new(10.0, 10.0, 10.0);
        let b = glam::Vec3::new(30.0, 10.0, 10.0);
        // 4-cell dense wall in the middle of the segment
        let atten = segment_attenuation(a, b, 0.5, 0.6, |x, _, _| {
            if (18..22).contains(&x) { 10.0 } else { 0.0 }
        });
        assert!(atten < 0.01, "wall not blocking: {}", atten);
    }

    #[test]
    fn segment_attenuation_haze_below_threshold_is_free() {
        let a = glam::Vec3::new(10.0, 10.0, 10.0);
        let b = glam::Vec3::new(30.0, 10.0, 10.0);
        assert_eq!(segment_attenuation(a, b, 0.5, 0.6, |_, _, _| 0.5), 1.0);
    }

    #[test]
    fn segment_attenuation_ignores_endpoint_density() {
        let a = glam::Vec3::new(10.0, 10.0, 10.0);
        let b = glam::Vec3::new(20.0, 10.0, 10.0);
        // Dense only within 1 cell of either endpoint — inside the 1.5-cell margins
        let atten = segment_attenuation(a, b, 0.5, 0.6, |x, _, _| {
            if x <= 11 || x >= 19 { 50.0 } else { 0.0 }
        });
        assert_eq!(atten, 1.0);
    }

    #[test]
    fn segment_attenuation_monotone_in_k() {
        let a = glam::Vec3::new(10.0, 10.0, 10.0);
        let b = glam::Vec3::new(30.0, 10.0, 10.0);
        let wall = |x: i32, _: i32, _: i32| if x == 20 { 5.0 } else { 0.0 };
        let low = segment_attenuation(a, b, 0.25, 0.6, wall);
        let high = segment_attenuation(a, b, 1.0, 0.6, wall);
        assert!(high < low);
        assert!(low < 1.0);
    }

    #[test]
    fn segment_attenuation_short_edge_is_one() {
        let a = glam::Vec3::new(10.0, 10.0, 10.0);
        let b = glam::Vec3::new(12.0, 10.0, 10.0); // ≤ 2 × margin
        assert_eq!(segment_attenuation(a, b, 0.5, 0.6, |_, _, _| 50.0), 1.0);
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test segment_attenuation`
Expected: COMPILE ERROR — `segment_attenuation` not found.

- [ ] **Step 3: Implement the pure pieces**

Module level in `src/field.rs` (near `cutoff_window`):

```rust
/// How much of a pipe's light survives crossing the deposited field.
/// Integrates grid density above `threshold` along the segment at ~1
/// sample/cell, excluding 1.5-cell margins at both ends (an edge
/// legitimately starts inside its own endpoint's surface density — light
/// leaving a surface must not be strangled by its own source).
/// Occlusion is thus an arrival deficit caused by the deposits themselves.
fn segment_attenuation(
    a: glam::Vec3,
    b: glam::Vec3,
    k: f32,
    threshold: f32,
    sample: impl Fn(i32, i32, i32) -> f32,
) -> f32 {
    const END_MARGIN: f32 = 1.5;
    let ab = b - a;
    let len = ab.length();
    if len <= 2.0 * END_MARGIN {
        return 1.0;
    }
    let dir = ab / len;
    let mut integral = 0.0f32;
    let mut t = END_MARGIN;
    while t <= len - END_MARGIN {
        let p = a + dir * t;
        let d = sample(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
        integral += (d - threshold).max(0.0);
        t += 1.0;
    }
    if integral <= 0.0 { 1.0 } else { (-k * integral).exp() }
}

/// Shared box extents for footprint clears (pass 3a) and deposit kernels
/// (pass 3b) — these MUST stay identical or moved footprints leave residue.
fn deposit_extents(radii: glam::Vec3) -> (i32, i32, i32) {
    if radii != glam::Vec3::ZERO {
        ((radii.x * 2.0).ceil() as i32,
         (radii.y * 2.0).ceil() as i32,
         (radii.z * 2.0).ceil() as i32)
    } else {
        (1, 1, 1) // 3x3x3 tent for floor/rock
    }
}
```

Constants near `LINK_REFRESH_DIST`:

```rust
/// Density above this attenuates pipes (2× iso): solid bodies block,
/// floor-surface deposits and atmosphere haze below it pass freely.
pub const ATTEN_THRESHOLD: f32 = 0.6;
/// Attenuation strength (keys 5/6 tune it live).
pub const ATTEN_K_DEFAULT: f32 = 0.5;
/// First full attenuation pass — the grid is empty at graph-build time and
/// reaches decay equilibrium after ~15 ticks.
pub const ATTEN_INIT_TICK: u64 = 15;
```

Run: `cargo test segment_attenuation` — expected `6 passed`.

- [ ] **Step 4: Wire `edge_atten` through the SoA and Phase 2**

`struct DiffField` (after `tune_color`):

```rust
    /// Per-edge transmittance from density along the pipe (1.0 = clear).
    /// Only edges longer than link_connect_dist are ever attenuated.
    edge_atten: Vec<f32>,
    /// Attenuation strength (live-tunable, keys 5/6).
    pub atten_k: f32,
```

`DiffField::new()` literal: `edge_atten: Vec::new(),` and `atten_k: ATTEN_K_DEFAULT,`.

In `flatten_edges`, next to `self.edge_gammas = vec![1.0; total];` add:

```rust
        self.edge_atten = vec![1.0; total];
```

Recompute method (place after `refresh_cross_links`):

```rust
    /// Resample transmittance for every long edge from the current grid.
    /// Called once at ATTEN_INIT_TICK and after every cross-link refresh.
    fn recompute_edge_attenuation(&mut self) {
        let t0 = std::time::Instant::now();
        let long_dist_sq = self.link_connect_dist * self.link_connect_dist;
        let k = self.atten_k;
        let cells = &self.cells;
        let entities = &self.entities;
        let edge_targets = &self.edge_targets;
        let edge_atten = &mut self.edge_atten;
        let sample = |x: i32, y: i32, z: i32| -> f32 {
            if Self::in_bounds(x, y, z) {
                cells[Self::index(x as u32, y as u32, z as u32)].density
            } else {
                0.0
            }
        };
        let mut long_edges = 0u64;
        for e in entities.iter() {
            let start = e.edge_start as usize;
            for kk in start..start + e.edge_count as usize {
                let a = e.position;
                let b = entities[edge_targets[kk]].position;
                if a.distance_squared(b) <= long_dist_sq { continue; }
                edge_atten[kk] = segment_attenuation(a, b, k, ATTEN_THRESHOLD, &sample);
                long_edges += 1;
            }
        }
        log::debug!(
            "Edge attenuation: {} long edges resampled, {:.2} ms",
            long_edges,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
```

In `tick()`, extend the existing refresh block at the top:

```rust
        if self.travel_since_refresh >= LINK_REFRESH_DIST
            && self.tick.saturating_sub(self.last_refresh_tick) >= MIN_REFRESH_SPACING
        {
            self.travel_since_refresh = 0.0;
            self.last_refresh_tick = self.tick;
            self.refresh_cross_links();
            self.recompute_edge_attenuation();
        }
        if self.tick == ATTEN_INIT_TICK {
            self.recompute_edge_attenuation();
        }
```

Phase 2: next to `let edge_gammas = &self.edge_gammas;` add `let edge_atten = &self.edge_atten;`, and change the weight line:

```rust
                let mut w = edge_gammas[k] * distance_factors[idx];
```

to:

```rust
                let mut w = edge_gammas[k] * edge_atten[k] * distance_factors[idx];
```

- [ ] **Step 5: Remove `ray_blocked` and its call sites**

- Delete the entire `fn ray_blocked(...)` function.
- In `build_radiation_links`: delete the block-grid construction (`block_cell`, `block_radius_sq`, `block_grid` loop) and the LOS check inside the candidate loop (the `if Self::ray_blocked(...) { blocked_count += 1; continue; }` block plus the `blocked_count` variable); update the log line to drop the blocked count:
  `log::info!("Radiation links: {} directed edges (capped at {} per entity)", count, max_radiation);`
- In `refresh_cross_links`: delete the same block-grid construction and LOS check in the cross-radiation search.

- [ ] **Step 6: Use `deposit_extents` in both Phase 3 passes**

Pass 3a — replace:

```rust
            // Clear-box extents match the deposit kernel's extents.
            let use_gaussian = entity.deposit_radii != glam::Vec3::ZERO;
            let (half_x, half_y, half_z) = if use_gaussian {
                ((entity.deposit_radii.x * 2.0).ceil() as i32,
                 (entity.deposit_radii.y * 2.0).ceil() as i32,
                 (entity.deposit_radii.z * 2.0).ceil() as i32)
            } else {
                (1i32, 1i32, 1i32)
            };
```

with:

```rust
            // Clear-box extents match the deposit kernel's extents (shared helper).
            let (half_x, half_y, half_z) = deposit_extents(entity.deposit_radii);
```

Pass 3b — replace:

```rust
            let use_gaussian = entity.deposit_radii != glam::Vec3::ZERO;
            let (half_x, half_y, half_z) = if use_gaussian {
                // 2× radii so gaussians overlap heavily between adjacent skeleton
                // points and fade smoothly (exp(-4) ≈ 0.02 at boundary).
                ((entity.deposit_radii.x * 2.0).ceil() as i32,
                 (entity.deposit_radii.y * 2.0).ceil() as i32,
                 (entity.deposit_radii.z * 2.0).ceil() as i32)
            } else {
                (1i32, 1i32, 1i32) // 3x3x3 tent for floor/rock
            };
```

with:

```rust
            let use_gaussian = entity.deposit_radii != glam::Vec3::ZERO;
            let (half_x, half_y, half_z) = deposit_extents(entity.deposit_radii);
```

- [ ] **Step 7: Keys 5/6 and logging**

`src/renderer.rs` — extend `log_tuning` and add:

```rust
    pub fn atten_k_scale(&mut self, factor: f32) {
        self.diff_field.atten_k = crate::field::scale_tune(self.diff_field.atten_k, factor);
        self.log_tuning();
    }
```

and change `log_tuning` to:

```rust
    fn log_tuning(&self) {
        log::info!(
            "Tuning: density ×{:.4}, color ×{:.4}, atten_k {:.4}",
            self.diff_field.tune_density, self.diff_field.tune_color, self.diff_field.atten_k
        );
    }
```

`src/main.rs`, after `Digit4`:

```rust
                            KeyCode::Digit5 => {
                                state.renderer.atten_k_scale(0.5);
                            }
                            KeyCode::Digit6 => {
                                state.renderer.atten_k_scale(2.0);
                            }
```

(Note: a changed `atten_k` takes effect at the next attenuation recompute — while the dino walks that is ≤ a few seconds; document this in the README row.)

- [ ] **Step 8: Write the failing ignored integration test**

```rust
    #[test]
    #[ignore] // builds the full 512³ field (~2 GB, slow) — run: cargo test --release -- --ignored
    fn attenuation_shadows_follow_walker() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        // Past ATTEN_INIT_TICK and through several refreshes while walking
        for _ in 0..100 {
            field.tick(vp);
        }
        // Solid body AABB at the *walked* position
        let mut wmin = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut wmax = glam::Vec3::ZERO;
        for e in field.entities.iter().filter(|e| e.is_walker) {
            wmin = wmin.min(e.position);
            wmax = wmax.max(e.position);
        }
        let center = (wmin + wmax) * 0.5;
        let long_dist_sq = field.link_connect_dist * field.link_connect_dist;

        let (mut crossing_sum, mut crossing_n) = (0.0f64, 0u32);
        let (mut far_sum, mut far_n) = (0.0f64, 0u32);
        let mut min_crossing = 1.0f32;
        for e in field.entities.iter() {
            let start = e.edge_start as usize;
            for kk in start..start + e.edge_count as usize {
                let t = field.edge_targets[kk];
                let a = e.position;
                let b = field.entities[t].position;
                if a.distance_squared(b) <= long_dist_sq { continue; }
                let mid = (a + b) * 0.5;
                let inside = mid.cmpge(wmin).all() && mid.cmple(wmax).all();
                if inside {
                    crossing_sum += field.edge_atten[kk] as f64;
                    crossing_n += 1;
                    min_crossing = min_crossing.min(field.edge_atten[kk]);
                }
                let dx = mid.x - center.x;
                let dz = mid.z - center.z;
                if e.group == GROUP_FLOOR
                    && field.entities[t].group == GROUP_FLOOR
                    && dx * dx + dz * dz > 625.0
                {
                    far_sum += field.edge_atten[kk] as f64;
                    far_n += 1;
                }
            }
        }
        assert!(crossing_n > 0, "no long edges cross the body");
        assert!(far_n > 0, "no far floor edges found");
        assert!(min_crossing < 0.5, "no edge through the body is attenuated: min={}", min_crossing);
        let far_avg = far_sum / far_n as f64;
        let crossing_avg = crossing_sum / crossing_n as f64;
        assert!(
            far_avg > crossing_avg,
            "far edges ({:.3}) not clearer than body-crossing edges ({:.3})",
            far_avg, crossing_avg
        );
    }
```

- [ ] **Step 9: Run everything**

Run: `cargo test` — expected `23 passed; 3 ignored` (17 prior + 6 attenuation units; percentile/clamp counts from Tasks 1–2 included).
Run: `cargo test --release -- --ignored` — all three integration tests pass.
Run: `cargo build --release` — clean; **no dead-code warning for `ray_blocked`** (it must be deleted, not orphaned).

- [ ] **Step 10: Update docs**

`README.md` controls table — add after the `-`/`=` row:

```markdown
| H | Dump field saturation stats + floor contrast probe to the log |
| 1 / 2 | Halve / double the density tuning multiplier |
| 3 / 4 | Halve / double the color tuning multiplier |
| 5 / 6 | Halve / double the pipe attenuation strength (applies at next refresh) |
```

`ARCHITECTURE.md` — in the radiation-links description (Edges section) and/or after the Per-Tick Pipeline table, add:

```markdown
Long-range (radiation) edges are **attenuated by the field itself**: each
pipe's transmittance is `exp(−k · ∫ density above threshold)` sampled along
its segment, recomputed as the walker moves. A pipe crossing the dino's
body carries almost nothing, so shadows are an *arrival deficit* caused by
the deposits — there is no line-of-sight oracle.
```

- [ ] **Step 11: Commit**

```bash
git add src/field.rs src/renderer.rs src/main.rs README.md ARCHITECTURE.md
git commit -m "feat: density-attenuated pipes — deposits themselves occlude (replaces LOS oracle)"
```

---

### Task 6 (OPTIONAL — only if the user still sees the clear-dip flicker after calibration): B1c equilibrium seeding

**Files:**
- Modify: `src/field.rs` (deposit_queue tuple + pass 3a/3b)

**Interfaces:**
- Consumes: `deposit_queue` (now `Vec<(usize, glam::Vec3, bool)>` — third element = "cleared this tick").

- [ ] **Step 1: Extend the queue with a cleared flag**

Change the field type to `deposit_queue: Vec<(usize, glam::Vec3, bool)>` (update the doc comment). In pass 3a, compute `let cleared = entity.prev_deposit_idx >= 0 && entity.prev_deposit_idx != new_base_idx;` BEFORE the clear block (which tests the same condition), and change the push to `self.deposit_queue.push((ent_idx, deposit_pos, cleared));`.

- [ ] **Step 2: Apply the equilibrium boost in pass 3b**

Destructure `let (ent_idx, deposit_pos, cleared) = self.deposit_queue[qi];` and after the boost lines add:

```rust
            // Equilibrium seeding: a freshly cleared footprint restarts from
            // one deposit instead of the decay steady state (1/(1-0.85) ≈
            // 6.67×), dipping the iso edge for ~0.5 s. Land this entity's own
            // contribution at equilibrium instantly. (Contributions from
            // static neighbors into the cleared cells still dip — partial
            // fix, accepted by design.)
            let eq = if cleared { 1.0 / (1.0 - 0.85) } else { 1.0 };
            let total_r = total_r * eq;
            let total_g = total_g * eq;
            let total_b = total_b * eq;
            let total_d = total_d * eq;
```

- [ ] **Step 3: Verify**

Run: `cargo test` (all pass), `cargo test --release -- --ignored` (all pass), visual check with the user: residual shadow/edge flicker reduced while walking.

- [ ] **Step 4: Commit**

```bash
git add src/field.rs
git commit -m "fix: seed freshly cleared footprints at decay equilibrium (clear-dip flicker)"
```

---

## Final Verification

- [ ] `cargo test` and `cargo test --release -- --ignored` all green.
- [ ] Smoke run: FPS in the usual range; attenuation recompute logs low single-digit ms.
- [ ] Human: jaw opening visible; shadow present, soft-edged, and **following the dino** (watch the ground as it paces); `5`/`6` visibly deepen/lighten the shadow after the next refresh; body lighting not dimmer than before (if it is, raise `max_radiation` from 10 to 14 in `build_radiation_links` and `refresh_cross_links` — the expected remedy for cap crowding, per the spec).
