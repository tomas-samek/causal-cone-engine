# Subluminal Dino Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the demo dino walk at a true 1e-6 c (cells per world-tick) with visible motion, via a time-lapse world clock, rigid group translation, and periodic walker↔world light-link refresh.

**Architecture:** A new `WalkController` (own module, pure logic) converts the stored subluminal velocity into a per-sim-step displacement (`speed × time_lapse`). Every dino entity carries an `is_walker` flag and receives that displacement uniformly in Phase 3 (rigid body — no per-entity bounce). When cumulative travel exceeds 1 cell, only the edges crossing the walker/world boundary are rebuilt so the shadow and lighting follow; dino-internal and world-internal edges (and their in-flight deposits) are preserved.

**Tech Stack:** Rust 2021, glam, existing `field.rs` SoA edge machinery. No new dependencies.

**Spec:** `docs/plans/2026-07-03-subluminal-dino-design.md` (approved).

## Global Constraints

- `DINO_SPEED_C = 1e-6` — dino speed in cells per world-tick (fraction of c). Never exceeded.
- `TIME_LAPSE_DEFAULT = 100_000` — world-ticks per sim step.
- `TIME_LAPSE_MAX = 1_048_576` — upper clamp (2^20); lower clamp is 1.
- `WALK_SPAN = 6.0` — max cells of drift from spawn along the walk axis (±Z). Chosen so the feet never reach the rock (which needs Z-offset > +8) and the tail stays roughly over the floor.
- `LINK_REFRESH_DIST = 1.0` — cells of walker travel between cross-link refreshes.
- Walk axis is ±Z only. No rotation, no gait (out of scope per spec).
- Walker entities keep `velocity = Vec3::ZERO`; all their motion comes from the controller.
- The observer, its 0.5c cap, and its `speed()` readout are untouched.
- Build/verify with `cargo build --release` (debug is too slow for the field sim). Unit tests: `cargo test`. Expensive field tests are `#[ignore]`d: `cargo test --release -- --ignored`.

## File Structure

- **Create** `src/walker.rs` — `WalkController` + constants. Pure, no `DiffField` dependency, fully unit-testable.
- **Modify** `src/main.rs` — `mod walker;`, two key bindings, title-bar text.
- **Modify** `src/field.rs` — `is_walker` flag on `Entity`, walker integration in `tick()` Phase 3, absolute-coordinate animation fixes (tail wag, jaw), `refresh_cross_links()`, stored link distances, `#[ignore]`d integration tests.
- **Modify** `src/renderer.rs` — three small pass-through methods (`time_lapse`, `halve_time_lapse`, `double_time_lapse`).
- **Modify** `README.md`, `ARCHITECTURE.md` — docs (Task 5).

---

### Task 1: WalkController module

**Files:**
- Create: `src/walker.rs`
- Modify: `src/main.rs:9-12` (add `mod walker;`)
- Test: inline `#[cfg(test)]` in `src/walker.rs`

**Interfaces:**
- Consumes: nothing (pure module, only `glam::Vec3`).
- Produces (used by Tasks 2–4):
  - `pub struct WalkController { pub speed_c: f32, pub direction: Vec3, pub time_lapse: u64, pub offset: Vec3, pub span: f32 }`
  - `pub fn WalkController::new() -> Self`
  - `pub fn WalkController::step(&mut self) -> Vec3` — returns this step's displacement, updates `offset`, flips `direction` at span.
  - `pub fn WalkController::halve_time_lapse(&mut self)` / `pub fn double_time_lapse(&mut self)`
  - `pub const DINO_SPEED_C: f32`, `pub const TIME_LAPSE_DEFAULT: u64`, `pub const TIME_LAPSE_MAX: u64`, `pub const WALK_SPAN: f32`

- [ ] **Step 1: Write the failing tests**

Create `src/walker.rs` containing ONLY the test module for now (so the test fails to compile against missing items — that is the red state), and register the module. In `src/main.rs`, after `mod consumption;` add:

```rust
mod walker;
```

`src/walker.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn step_displacement_is_speed_times_lapse() {
        let mut w = WalkController::new();
        let d = w.step();
        // 1e-6 cells/world-tick × 100_000 world-ticks/step = 0.1 cells/step, along +Z
        assert!((d.z - 0.1).abs() < 1e-5, "dz = {}", d.z);
        assert_eq!(d.x, 0.0);
        assert_eq!(d.y, 0.0);
        assert!((w.offset.z - 0.1).abs() < 1e-5);
    }

    #[test]
    fn lapse_of_one_gives_literal_subluminal_step() {
        let mut w = WalkController::new();
        w.time_lapse = 1;
        let d = w.step();
        assert!((d.z - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn offset_is_exact_sum_of_returned_deltas() {
        let mut w = WalkController::new();
        let mut sum = Vec3::ZERO;
        for _ in 0..500 {
            sum += w.step();
        }
        assert_eq!(sum, w.offset);
    }

    #[test]
    fn turns_around_at_span_and_stays_bounded() {
        let mut w = WalkController::new();
        // 0.1 cells/step, span 6.0 → direction must flip around step 60
        let mut flipped = false;
        for _ in 0..500 {
            w.step();
            if w.direction.z < 0.0 {
                flipped = true;
            }
            // never drifts beyond span + one step of slack
            assert!(w.offset.z.abs() <= WALK_SPAN + 0.11, "offset = {}", w.offset.z);
        }
        assert!(flipped, "walker never turned around");
    }

    #[test]
    fn time_lapse_clamps() {
        let mut w = WalkController::new();
        w.time_lapse = 1;
        w.halve_time_lapse();
        assert_eq!(w.time_lapse, 1);
        w.time_lapse = TIME_LAPSE_MAX;
        w.double_time_lapse();
        assert_eq!(w.time_lapse, TIME_LAPSE_MAX);
        w.time_lapse = TIME_LAPSE_DEFAULT;
        w.double_time_lapse();
        assert_eq!(w.time_lapse, 200_000);
        w.halve_time_lapse();
        assert_eq!(w.time_lapse, 100_000);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test walker`
Expected: COMPILE ERROR — `WalkController` not found (this is the red state).

- [ ] **Step 3: Write the implementation**

Prepend to `src/walker.rs` (above the test module):

```rust
// Walk controller — moves the walker group (the dino) as one rigid body.
//
// Speeds are stored honestly in cells per world-tick (fractions of c).
// At strongly subluminal speeds (1e-6 c) motion is imperceptible in real
// time — ~9 hours per cell — so each sim step advances `time_lapse`
// world-ticks of world time: displacement = speed × time_lapse.
// The light field is quasi-static at this ratio (adiabatic regime), so
// light transport keeps running once per sim step unchanged.

use glam::Vec3;

/// Dino speed in cells per world-tick (fraction of c). Strongly subluminal.
pub const DINO_SPEED_C: f32 = 1e-6;
/// World-ticks that elapse per sim step (default ×100,000 time-lapse).
pub const TIME_LAPSE_DEFAULT: u64 = 100_000;
/// Upper clamp for time-lapse adjustment (2^20).
pub const TIME_LAPSE_MAX: u64 = 1_048_576;
/// Max cells of drift from spawn along the walk axis before turning around.
/// Keeps the feet clear of the rock (needs Z-offset > +8) and the tail
/// roughly over the floor.
pub const WALK_SPAN: f32 = 6.0;

pub struct WalkController {
    /// Speed in cells per world-tick (≤ 1e-6 c).
    pub speed_c: f32,
    /// Current walk direction (unit vector, ±Z in v1).
    pub direction: Vec3,
    /// World-ticks per sim step.
    pub time_lapse: u64,
    /// Cumulative displacement from spawn.
    pub offset: Vec3,
    /// Max |offset.z| before the direction flips.
    pub span: f32,
}

impl WalkController {
    pub fn new() -> Self {
        Self {
            speed_c: DINO_SPEED_C,
            direction: Vec3::Z,
            time_lapse: TIME_LAPSE_DEFAULT,
            offset: Vec3::ZERO,
            span: WALK_SPAN,
        }
    }

    /// Advance one sim step. Returns the displacement to apply to every
    /// walker entity this step; flips direction once the span is reached.
    pub fn step(&mut self) -> Vec3 {
        let delta = self.direction * (self.speed_c * self.time_lapse as f32);
        self.offset += delta;
        if self.offset.z.abs() >= self.span {
            self.direction = -self.direction;
        }
        delta
    }

    pub fn halve_time_lapse(&mut self) {
        self.time_lapse = (self.time_lapse / 2).max(1);
    }

    pub fn double_time_lapse(&mut self) {
        self.time_lapse = (self.time_lapse * 2).min(TIME_LAPSE_MAX);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test walker`
Expected: `test result: ok. 5 passed` (plus the existing consumption tests untouched).

- [ ] **Step 5: Verify the whole project still builds**

Run: `cargo build --release`
Expected: success, one possible `dead_code` warning for the not-yet-used controller (goes away in Task 2).

- [ ] **Step 6: Commit**

```bash
git add src/walker.rs src/main.rs
git commit -m "feat: add WalkController — subluminal speed + time-lapse world clock"
```

---

### Task 2: Walker flag, rigid motion, and animation-anchor fixes

**Files:**
- Modify: `src/field.rs:71-166` (Entity struct + `Entity::new`)
- Modify: `src/field.rs:180-211` (DiffField struct), `src/field.rs:275-300` (`DiffField::new`)
- Modify: `src/field.rs:1100-1108` (end of dino spawn in `spawn_demo_scene`)
- Modify: `src/field.rs:1569-1582` (Phase 3 movement), `src/field.rs:1638-1665` (tail/jaw animation anchors)
- Test: `#[cfg(test)]` module at the bottom of `src/field.rs`

**Interfaces:**
- Consumes: `crate::walker::WalkController` (Task 1).
- Produces (used by Tasks 3–4):
  - `Entity.is_walker: bool` (public field, default `false`)
  - `DiffField.walker: WalkController` (public field)
  - `DiffField.travel_since_refresh: f32` (private; Task 3 reads/resets it)

- [ ] **Step 1: Write the failing integration test**

Add at the very bottom of `src/field.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_view_proj() -> glam::Mat4 {
        // Same pose as Observer::new(): in front of the dino, looking at it.
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 500.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(256.0, 256.0, 310.0),
            glam::Vec3::new(256.0, 256.0, 256.0),
            glam::Vec3::Y,
        );
        proj * view
    }

    #[test]
    #[ignore] // builds the full 512³ field (~2 GB, slow) — run: cargo test --release -- --ignored
    fn walker_group_moves_rigidly() {
        let mut field = DiffField::new();
        let walkers: Vec<usize> = field.entities.iter().enumerate()
            .filter(|(_, e)| e.is_walker)
            .map(|(i, _)| i)
            .collect();
        assert!(walkers.len() > 100, "expected skeleton + receptors, got {}", walkers.len());
        // Sun, floor, rock, vacuum must NOT walk
        assert!(field.entities.iter().all(|e| {
            !(e.is_walker && matches!(e.group, GROUP_SUN | GROUP_FLOOR | GROUP_VACUUM | GROUP_ROCK))
        }));

        let start: Vec<glam::Vec3> = walkers.iter().map(|&i| field.entities[i].position).collect();
        let vp = test_view_proj();
        for _ in 0..30 {
            field.tick(vp);
        }
        let expected = field.walker.offset;
        assert!(expected.length() > 0.5, "walker barely moved: {:?}", expected);

        // Every walker entity moved by exactly the shared offset
        for (k, &i) in walkers.iter().enumerate() {
            let moved = field.entities[i].position - start[k];
            assert!((moved - expected).length() < 1e-3,
                "entity {} moved {:?}, expected {:?}", i, moved, expected);
        }
        // Rigidity: pairwise distances preserved (spot-check first 10)
        let m = walkers.len().min(10);
        for a in 0..m {
            for b in (a + 1)..m {
                let d0 = start[a].distance(start[b]);
                let d1 = field.entities[walkers[a]].position
                    .distance(field.entities[walkers[b]].position);
                assert!((d0 - d1).abs() < 1e-3, "pair ({},{}) drifted: {} vs {}", a, b, d0, d1);
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -- --ignored walker_group_moves_rigidly`
Expected: COMPILE ERROR — no field `is_walker` on `Entity`, no field `walker` on `DiffField`.

- [ ] **Step 3: Add the `is_walker` flag to Entity**

In `src/field.rs`, in `struct Entity` after the `deposit_radii` field (line ~131), add:

```rust
    /// Member of the rigid walker group (the dino) — moved by WalkController,
    /// never by per-entity velocity/bounce.
    pub is_walker: bool,
```

In `Entity::new` (after `deposit_radii: glam::Vec3::ZERO,`, line ~164), add:

```rust
            is_walker: false,
```

- [ ] **Step 4: Add walker state to DiffField**

In `struct DiffField` (after `show_trie_depth: bool,`, line ~210), add:

```rust
    /// Rigid-body walk controller for the dino group.
    pub walker: crate::walker::WalkController,
    /// Cells of walker travel since the last cross-link refresh (Task 3).
    travel_since_refresh: f32,
```

In `DiffField::new()` struct literal (after `show_trie_depth: false,`, line ~298), add:

```rust
            walker: crate::walker::WalkController::new(),
            travel_since_refresh: 0.0,
```

- [ ] **Step 5: Flag the dino entities at spawn**

In `spawn_demo_scene`, right after the `build_receptor_shell(...)` call (line ~1108) and BEFORE the `// ROCK` section, add:

```rust
        // Mark the whole dino — skeleton, midpoints, receptor shell — as the
        // rigid walker group. All dino parts use groups BODY..=ARM_R; receptors
        // inherit their group from the metaball they sit on. Heat conversion
        // happens later in build_connections and doesn't change groups.
        for e in self.entities.iter_mut() {
            if (GROUP_BODY..=GROUP_ARM_R).contains(&e.group) {
                e.is_walker = true;
            }
        }
```

(`GROUP_BODY = 1` through `GROUP_ARM_R = 15` — exactly the 15 dino groups; rock/sun/floor/vacuum are 16–19.)

- [ ] **Step 6: Apply rigid motion in Phase 3**

In `tick()`, immediately before the Phase 3 comment `// Phase 3: entities deposit to grid (only visible entities)` (line ~1569), add:

```rust
        // Rigid walker motion: one shared displacement for the whole dino
        // group this step (speed_c × time_lapse world-ticks).
        let walk_delta = self.walker.step();
        self.travel_since_refresh += walk_delta.length();
        let walk_offset_z = self.walker.offset.z;
```

Inside the Phase 3 entity loop, change the movement lines (1573–1574) from:

```rust
            // Move entity (all entities, not just visible — keeps positions consistent)
            entity.position += entity.velocity;
```

to:

```rust
            // Move entity (all entities, not just visible — keeps positions consistent).
            // Walkers get the shared rigid displacement; their own velocity stays zero
            // so the per-axis bounce below can never tear the group apart.
            if entity.is_walker {
                entity.position += walk_delta;
            }
            entity.position += entity.velocity;
```

- [ ] **Step 7: Fix the absolute-coordinate animation anchors**

Tail wag (line ~1643): the anchor `center_z` is hardcoded to the spawn position and would compute wrong amplitudes once the dino translates. Change:

```rust
                let center_z = FIELD_SIZE as f32 / 2.0;
```

to:

```rust
                // Anchor follows the walker so the wag taper stays body-relative
                let center_z = FIELD_SIZE as f32 / 2.0 + walk_offset_z;
```

Jaw pivot (line ~1657): change:

```rust
                let pivot_z = center + 8.0;  // back of jaw (base z-offset from center)
```

to:

```rust
                let pivot_z = center + 8.0 + walk_offset_z; // back of jaw, follows walker
```

- [ ] **Step 8: Run the fast tests, then the integration test**

Run: `cargo test`
Expected: all walker + consumption unit tests pass (the ignored test is skipped).

Run: `cargo test --release -- --ignored walker_group_moves_rigidly`
Expected: PASS (takes a while — full field build + 30 ticks).

- [ ] **Step 9: Visual smoke check**

Run: `cargo run --release`
Expected: the dino visibly drifts along Z at ~3 cells/sec, paces back and forth within ±6 cells, body stays coherent (no tearing), tail wag and jaw animation look unchanged relative to the body. Known acceptable artifact at this point: the shadow and body lighting stay anchored near spawn (fixed in Task 3).

- [ ] **Step 10: Commit**

```bash
git add src/field.rs
git commit -m "feat: dino walks as rigid group at 1e-6c under time-lapse clock"
```

---

### Task 3: Cross-link refresh — lighting and shadow follow the walker

**Files:**
- Modify: `src/field.rs:275-320` (`DiffField::new` — store link distances)
- Modify: `src/field.rs:180-215` (DiffField struct — two new fields)
- Modify: `src/field.rs:1272-1296` (top of `tick()` — refresh trigger)
- Modify: `src/field.rs` (new method `refresh_cross_links` next to `build_radiation_links`, ~line 722)
- Test: extend `#[cfg(test)]` module in `src/field.rs`

**Interfaces:**
- Consumes: `Entity.is_walker`, `DiffField.travel_since_refresh` (Task 2); existing `spatial_hash`, `ray_blocked`, `flatten_edges`, `build_reverse_edges`.
- Produces: `fn refresh_cross_links(&mut self)` (private), fields `link_connect_dist: f32`, `link_radiation_dist: f32` (private). New const `LINK_REFRESH_DIST: f32 = 1.0` at file scope near the `GROUP_*` consts.

- [ ] **Step 1: Write the failing integration test**

Add to the `tests` module in `src/field.rs`:

```rust
    #[test]
    #[ignore] // builds the full 512³ field (~2 GB, slow) — run: cargo test --release -- --ignored
    fn cross_links_follow_walker() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        // 90 steps ≈ 9 cells of travel (with one turnaround at +6) → several refreshes
        for _ in 0..90 {
            field.tick(vp);
        }

        // Every walker↔world edge must respect the link distances plus at most
        // LINK_REFRESH_DIST of staleness — links may never lag the walk.
        let max_len = field.link_radiation_dist + LINK_REFRESH_DIST + 0.2;
        let mut cross_edges = 0u32;
        let mut foot_to_floor = 0u32;
        for (i, e) in field.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                let t = field.edge_targets[k];
                if e.is_walker == field.entities[t].is_walker { continue; }
                cross_edges += 1;
                let d = e.position.distance(field.entities[t].position);
                assert!(d <= max_len, "stale cross edge {}→{}: {} cells", i, t, d);
                if e.is_walker
                    && matches!(e.group, GROUP_FOOT_L | GROUP_FOOT_R)
                    && field.entities[t].group == GROUP_FLOOR
                {
                    foot_to_floor += 1;
                }
            }
        }
        assert!(cross_edges > 0, "walker is isolated from the world graph");
        assert!(foot_to_floor > 0, "no foot→floor links at the new position — shadow can't follow");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -- --ignored cross_links_follow_walker`
Expected: COMPILE ERROR — `link_radiation_dist` and `LINK_REFRESH_DIST` don't exist yet. (After Step 3 compiles, the assertions themselves fail on stale edges — either red state is fine.)

- [ ] **Step 3: Store the link distances and refresh constant**

Near the `GROUP_*` constants (line ~37), add:

```rust
/// Cells of walker travel between cross-link refreshes.
pub const LINK_REFRESH_DIST: f32 = 1.0;
```

In `struct DiffField` (after `travel_since_refresh: f32,`), add:

```rust
    /// Link distances captured at build time, reused by refresh_cross_links.
    link_connect_dist: f32,
    link_radiation_dist: f32,
```

In `DiffField::new()`: add to the struct literal (after `travel_since_refresh: 0.0,`):

```rust
            link_connect_dist: 0.0,
            link_radiation_dist: 0.0,
```

and after the two existing build calls (line ~314-315):

```rust
        field.build_connections(connect_dist);
        field.build_radiation_links(radiation_dist, connect_dist);
        field.link_connect_dist = connect_dist;
        field.link_radiation_dist = radiation_dist;
```

- [ ] **Step 4: Implement `refresh_cross_links`**

Add after `flatten_edges` (line ~742) in `src/field.rs`:

```rust
    /// Rebuild only the edges that cross the walker/world boundary.
    ///
    /// Rigid translation keeps dino-internal edges valid forever, and world
    /// entities don't move — so those edge sets (and their in-flight
    /// deposits) are preserved verbatim. Only walker↔world pairs are
    /// re-searched: short-range connection edges (this is how atmosphere
    /// light reaches the receptor shell) and LOS-checked radiation links
    /// (this is how the feet shade the floor). Heat flags, colors, and
    /// consumption state are deliberately NOT touched.
    fn refresh_cross_links(&mut self) {
        let t0 = std::time::Instant::now();
        let connect_dist = self.link_connect_dist;
        let radiation_dist = self.link_radiation_dist;
        let connect_dist_sq = connect_dist * connect_dist;
        let radiation_dist_sq = radiation_dist * radiation_dist;
        let n = self.entities.len();
        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();
        let walker: Vec<bool> = self.entities.iter().map(|e| e.is_walker).collect();

        // Save in-flight pipe contents keyed by (source, target)
        let mut old_deposits: HashMap<(usize, usize), EdgeDeposit> =
            HashMap::with_capacity(self.edge_targets.len());
        for (i, e) in self.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                old_deposits.insert((i, self.edge_targets[k]), self.edge_deposits[k]);
            }
        }

        // Retain same-side edges only (drop all cross edges)
        let mut temp_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, e) in self.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                let t = self.edge_targets[k];
                if walker[i] == walker[t] {
                    temp_edges[i].push(t);
                }
            }
        }

        // New cross connection edges: walker↔world within connect_dist.
        // Iterate walkers only (few hundred) against the world spatial hash.
        let grid = Self::spatial_hash(&positions, connect_dist);
        for i in 0..n {
            if !walker[i] { continue; }
            let pos = positions[i];
            let cx = (pos.x / connect_dist).floor() as i32;
            let cy = (pos.y / connect_dist).floor() as i32;
            let cz = (pos.z / connect_dist).floor() as i32;
            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if walker[j] { continue; }
                                if pos.distance_squared(positions[j]) < connect_dist_sq {
                                    temp_edges[i].push(j);
                                    temp_edges[j].push(i);
                                }
                            }
                        }
                    }
                }
            }
        }

        // New cross radiation links: solid↔solid, LOS-checked, capped like
        // build_radiation_links (closest 10 per entity among the new links).
        let block_cell = connect_dist.max(1.0);
        let block_radius_sq = (connect_dist * 0.3).powi(2);
        let mut block_grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let pos = positions[i];
            let key = (
                (pos.x / block_cell).floor() as i32,
                (pos.y / block_cell).floor() as i32,
                (pos.z / block_cell).floor() as i32,
            );
            block_grid.entry(key).or_default().push(i);
        }
        let rad_grid = Self::spatial_hash(&positions, radiation_dist);
        let max_radiation: usize = 10;
        let mut cross_rad: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for i in 0..n {
            if !walker[i] { continue; }
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let pos = positions[i];
            let cx = (pos.x / radiation_dist).floor() as i32;
            let cy = (pos.y / radiation_dist).floor() as i32;
            let cz = (pos.z / radiation_dist).floor() as i32;
            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = rad_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if walker[j] { continue; }
                                if self.entities[j].is_vacuum || self.entities[j].is_heat { continue; }
                                let dist_sq = pos.distance_squared(positions[j]);
                                if dist_sq >= connect_dist_sq && dist_sq < radiation_dist_sq {
                                    if Self::ray_blocked(
                                        pos, positions[j], i, j,
                                        &block_grid, block_cell, &positions, block_radius_sq,
                                    ) {
                                        continue;
                                    }
                                    cross_rad[i].push((j, dist_sq));
                                    cross_rad[j].push((i, dist_sq));
                                }
                            }
                        }
                    }
                }
            }
        }
        for i in 0..n {
            let cands = &mut cross_rad[i];
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cands.truncate(max_radiation);
            for &(target, _) in cands.iter() {
                temp_edges[i].push(target);
            }
        }

        // Repack SoA (recomputes edge_dirs from the moved positions),
        // restore retained pipe contents, recompute gammas, rebuild reverse.
        self.flatten_edges(temp_edges);
        for i in 0..n {
            let start = self.entities[i].edge_start as usize;
            let count = self.entities[i].edge_count as usize;
            for k in start..start + count {
                if let Some(dep) = old_deposits.get(&(i, self.edge_targets[k])) {
                    self.edge_deposits[k] = *dep;
                }
            }
        }
        for i in 0..n {
            let start = self.entities[i].edge_start as usize;
            let end = start + self.entities[i].edge_count as usize;
            for k in start..end {
                let target = self.edge_targets[k];
                let dist_sq = positions[i].distance_squared(positions[target]);
                self.edge_gammas[k] = 1.0 / dist_sq.max(0.1);
            }
        }
        self.build_reverse_edges();

        log::debug!(
            "Cross-link refresh at offset {:?}: {} total edges, {:.2} ms",
            self.walker.offset,
            self.edge_targets.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
```

- [ ] **Step 5: Trigger the refresh from `tick()`**

At the very top of `tick()` (line ~1272, before `compute_active_set`), add:

```rust
        // Refresh walker↔world light links once the dino has drifted a cell
        // from where they were last built — shadow and lighting follow.
        if self.travel_since_refresh >= LINK_REFRESH_DIST {
            self.travel_since_refresh = 0.0;
            self.refresh_cross_links();
        }
```

- [ ] **Step 6: Run all tests**

Run: `cargo test`
Expected: fast tests pass.

Run: `cargo test --release -- --ignored`
Expected: both `walker_group_moves_rigidly` and `cross_links_follow_walker` PASS.

- [ ] **Step 7: Visual + performance check**

Run: `RUST_LOG=debug cargo run --release` (PowerShell: `$env:RUST_LOG='debug'; cargo run --release`)
Expected: the shadow and body lighting now travel with the dino; `Cross-link refresh` debug lines appear ~3×/sec with single-digit-millisecond timings; no visible hitch or flicker beyond a subtle one-step ripple near the feet at refresh moments; FPS comparable to before.

- [ ] **Step 8: Commit**

```bash
git add src/field.rs
git commit -m "feat: refresh walker↔world light links as the dino walks"
```

---

### Task 4: Time-lapse controls and HUD

**Files:**
- Modify: `src/renderer.rs:453-462` (add three methods after `increase_render_depth`)
- Modify: `src/main.rs:140-147` (key handling), `src/main.rs:187-193` (title bar)

**Interfaces:**
- Consumes: `DiffField.walker` (Task 2), `WalkController::{halve,double}_time_lapse`, `walker::DINO_SPEED_C` (Task 1).
- Produces: `Renderer::time_lapse() -> u64`, `Renderer::halve_time_lapse()`, `Renderer::double_time_lapse()`.

- [ ] **Step 1: Add Renderer pass-through methods**

In `src/renderer.rs`, after `increase_render_depth` (line ~461), add:

```rust
    pub fn time_lapse(&self) -> u64 {
        self.diff_field.walker.time_lapse
    }

    pub fn halve_time_lapse(&mut self) {
        self.diff_field.walker.halve_time_lapse();
        log::info!("Time lapse: ×{}", self.diff_field.walker.time_lapse);
    }

    pub fn double_time_lapse(&mut self) {
        self.diff_field.walker.double_time_lapse();
        log::info!("Time lapse: ×{}", self.diff_field.walker.time_lapse);
    }
```

- [ ] **Step 2: Bind the keys**

In `src/main.rs`, in the pressed-key `match` after the `BracketRight` arm (line ~145), add:

```rust
                            KeyCode::Minus => {
                                state.renderer.halve_time_lapse();
                            }
                            KeyCode::Equal => {
                                state.renderer.double_time_lapse();
                            }
```

- [ ] **Step 3: Extend the title bar**

In `src/main.rs`, replace the `set_title` call (lines ~187-193):

```rust
                    state.window.set_title(&format!(
                        "Causal Cone Engine v0.7 — {:.0} FPS — tick {} — observer v={:.3}c",
                        state.current_fps,
                        state.tick_count,
                        state.observer.speed()
                    ));
```

with:

```rust
                    state.window.set_title(&format!(
                        "Causal Cone Engine v0.7 — {:.0} FPS — tick {} — observer v={:.3}c — dino v={:e}c — lapse ×{}",
                        state.current_fps,
                        state.tick_count,
                        state.observer.speed(),
                        walker::DINO_SPEED_C,
                        state.renderer.time_lapse()
                    ));
```

(`{:e}` renders `1e-6`. `walker::` is already in scope via `mod walker;` from Task 1.)

- [ ] **Step 4: Build and verify manually**

Run: `cargo build --release`
Expected: clean build.

Run: `cargo run --release`
Expected: title reads `… — dino v=1e-6c — lapse ×100000` (updates once per second). Pressing `-` repeatedly halves the lapse (log lines confirm) and the dino visibly slows, freezing entirely near ×1 — that IS 1e-6 c in real time. Pressing `=` accelerates it back up to a brisk walk at ×1,048,576 (~3.1 cells/sec × 10 ≈ 31 cells/sec). Turnaround still happens at ±6 cells regardless of lapse.

- [ ] **Step 5: Commit**

```bash
git add src/renderer.rs src/main.rs
git commit -m "feat: time-lapse keys (-/=) and dino speed HUD"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md:23-34` (controls table), `README.md:64-73` (Demo Scene)
- Modify: `ARCHITECTURE.md:69-83` (Per-Tick Pipeline table), `ARCHITECTURE.md:117-122` (Observer section), `ARCHITECTURE.md:124-139` (Demo Scene)

**Interfaces:** none (prose only). Numbers must match the constants from Task 1 verbatim.

- [ ] **Step 1: Update README controls table**

Add two rows to the table (after the `[` / `]` row, README.md line ~33):

```markdown
| `-` / `=` | Halve / double the time-lapse factor (×1 = literal real-time 1e-6 c) |
```

- [ ] **Step 2: Update README Demo Scene paragraph**

In README.md line ~70, change the sentence

```markdown
It stands on a 40×40 dirt/grass floor beside a rock, lit by a sun disc, with an
```

to

```markdown
It paces slowly back and forth on a 40×40 dirt/grass floor beside a rock —
moving at a true 1e-6 c, rendered visible through a ×100,000 time-lapse
world clock — lit by a sun disc, with an
```

- [ ] **Step 3: Update ARCHITECTURE.md**

1. In the Per-Tick Pipeline table (line ~75), add a row before **Active set**:

```markdown
| **Cross-link refresh** | When the walker (dino) has drifted ≥1 cell since links were last built, walker↔world connection and radiation edges are re-searched and the SoA edge arrays repacked; internal edges and their in-flight deposits are preserved. |
```

2. In the same table's **Phase 3** row, change the description from

```markdown
| **Phase 3 — deposit** | Entities move (and bounce off bounds). Heat and too-deep entities are skipped; vacuum entities scatter into the grid; visible solids deposit color/density into cells. Dirty slabs and the new AABB are recorded. |
```

to

```markdown
| **Phase 3 — deposit** | The walker group (dino) translates rigidly by `speed × time_lapse` and paces ±6 cells along Z; other entities move by velocity (and bounce off bounds). Heat and too-deep entities are skipped; vacuum entities scatter into the grid; visible solids deposit color/density into cells. Dirty slabs and the new AABB are recorded. |
```

3. After the Observer section (line ~122), add a new section:

```markdown
## Time-Lapse World Clock (`walker.rs`)

Two clocks are separated. A **sim step** is the 30 Hz wall-clock unit: one
graph hop of light, one decay + deposit pass. A **world tick** is the physics
unit where c = 1 cell/tick. Each sim step advances `time_lapse` world-ticks
(default ×100,000, keys `-`/`=`, clamped to [1, 2²⁰]).

The dino's speed is stored honestly as `DINO_SPEED_C = 1e-6` cells per
world-tick — strongly subluminal. Per sim step it moves
`1e-6 × 100,000 = 0.1 cells` (30×/sec, never jumping), i.e. ~3 cells/sec on
screen. Light transport still runs once per sim step: at this speed ratio
the light field is **quasi-static** (adiabatic regime) — light crosses the
whole field in a negligible fraction of a step of world time, so the
residual graph-convergence lag is invisible. The observer stays in the
wall-clock frame; its 0.5c cap and aberration vignette are unchanged.
```

- [ ] **Step 4: Verify docs render and numbers match**

Run: `cargo test walker` (constants unchanged: 1e-6, 100_000, 1_048_576, 6.0) and re-read the three edited doc sections checking every number against `src/walker.rs`.
Expected: all numbers match.

- [ ] **Step 5: Commit**

```bash
git add README.md ARCHITECTURE.md
git commit -m "docs: document time-lapse clock, walking dino, and -/= controls"
```

---

## Final Verification (after all tasks)

- [ ] `cargo test` — all fast tests pass.
- [ ] `cargo test --release -- --ignored` — both field integration tests pass.
- [ ] `cargo run --release` — dino paces at ~3 cells/sec with shadow and lighting following; `-` freezes it near ×1; `=` speeds it up; body never tears; FPS in the usual range (~170 at idle per previous measurements).
