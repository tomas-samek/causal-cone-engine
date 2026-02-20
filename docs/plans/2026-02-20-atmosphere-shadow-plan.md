# v0.6 Atmosphere & Shadow Tuning — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Concentrate atmospheric entities into a column around the dino, tune shader ambient, and bring the observer closer for better shadow visibility.

**Architecture:** Replace the broad vacuum grid with a tight cylindrical column centered on the dino AABB. Add per-tick scatter/magnitude modulation based on distance from the live AABB center. Lower the shader ambient floor from 0.25 to 0.10.

**Tech Stack:** Rust, wgpu, WGSL shaders, glam math

---

### Task 1: Move Observer Closer

**Files:**
- Modify: `src/observer.rs:40`

**Step 1: Change start position**

In `Observer::new()`, change the z component from 380.0 to 310.0:

```rust
position: Vec3::new(256.0, 256.0, 310.0),
```

**Step 2: Build and verify**

Run: `cargo build`
Expected: Compiles without errors.

**Step 3: Commit**

```bash
git add src/observer.rs
git commit -m "v0.6: Move observer closer (z=380 -> z=310)"
```

---

### Task 2: Lower Shader Ambient Floor

**Files:**
- Modify: `shaders/field_sample.wgsl:171`

**Step 1: Change ambient constant**

In `fs_main`, change the ambient floor from 0.25 to 0.10:

```wgsl
let ambient = 0.10; // ambient fill so shadows aren't pitch black
```

**Step 2: Build and verify**

Run: `cargo build`
Expected: Compiles without errors. (Shader is loaded at runtime, so compilation won't catch WGSL errors, but this change is a single constant.)

**Step 3: Commit**

```bash
git add shaders/field_sample.wgsl
git commit -m "v0.6: Lower shader ambient floor (0.25 -> 0.10)"
```

---

### Task 3: Replace Vacuum Grid with Atmospheric Column

**Files:**
- Modify: `src/field.rs:960-1023` (the vacuum generation section in `spawn_demo_scene`)

**Step 1: Add base_scatter and base_magnitude constants to Entity**

Add two new fields to the `Entity` struct that store the generation-time base values for per-tick modulation:

In `src/field.rs`, add to `Entity` struct (after `scatter` field, around line 88):

```rust
/// Base scatter value set at generation (for per-tick modulation)
pub base_scatter: f32,
/// Base deposit magnitude set at generation (for per-tick modulation)
pub base_magnitude: f32,
```

In `Entity::new()`, add defaults (after `scatter: 0.0`):

```rust
base_scatter: 0.0,
base_magnitude: 0.0,
```

**Step 2: Replace vacuum generation code**

Replace lines 960-1013 in `spawn_demo_scene` (from `let before_vacuum = self.entities.len();` through the end of the vacuum while loops) with:

```rust
let before_vacuum = self.entities.len();

// ATMOSPHERIC COLUMN — concentrated relay network around the dino.
// Cylindrical column centered on solid entity AABB. Vacuum entities relay
// light from sun to dino and scatter a fraction into the grid (atmosphere).
// Density peaked at column center, fading radially outward.

// Compute solid entity AABB for column centering
let mut solid_min = glam::Vec3::splat(FIELD_SIZE as f32);
let mut solid_max = glam::Vec3::ZERO;
for e in &self.entities {
    if !e.is_vacuum && !e.is_heat {
        solid_min = solid_min.min(e.position);
        solid_max = solid_max.max(e.position);
    }
}
let solid_center = (solid_min + solid_max) * 0.5;
let solid_half = (solid_max - solid_min) * 0.5;
let column_radius = solid_half.x.max(solid_half.z) * 1.5;

let vac_spacing = 3.0;
let col_y_min = solid_min.y - 5.0;
let col_y_max = sun_y;

let mut vx = solid_center.x - column_radius;
while vx <= solid_center.x + column_radius {
    let mut vy = col_y_min;
    while vy <= col_y_max {
        let mut vz = solid_center.z - column_radius;
        while vz <= solid_center.z + column_radius {
            let pos = glam::Vec3::new(vx, vy, vz);

            // Radial distance from column center (XZ plane only)
            let dx = pos.x - solid_center.x;
            let dz = pos.z - solid_center.z;
            let horiz_dist = (dx * dx + dz * dz).sqrt();
            let radial_frac = (horiz_dist / column_radius).clamp(0.0, 1.0);

            // Skip entities outside the column radius
            if radial_frac >= 1.0 {
                vz += vac_spacing;
                continue;
            }

            let falloff = 1.0 - radial_frac * radial_frac;

            let mut e = Entity::new(
                pos,
                glam::Vec3::ZERO,
                0.0,
                [0.0, 0.0, 0.0],
            );
            e.pass_through = 0.95;
            e.is_vacuum = true;
            e.group = GROUP_VACUUM;

            // Density profile: peaked at center, fading radially
            let base_scatter_val = 0.0001 * falloff;
            let base_mag_val = 2.0 * falloff;
            e.scatter = base_scatter_val;
            e.deposit_magnitude = base_mag_val;
            e.base_scatter = base_scatter_val;
            e.base_magnitude = base_mag_val;

            // Vertical color gradient (warmer near ground, bluer up high)
            if vy < sun_y {
                let height_frac = (vy - col_y_min) / (sun_y - col_y_min);
                let bottom_weight = 1.0 - height_frac;
                e.color = [
                    0.4 + 0.4 * bottom_weight,
                    0.5 + 0.35 * bottom_weight,
                    0.9 + 0.1 * height_frac,
                ];
            }

            self.entities.push(e);

            vz += vac_spacing;
        }
        vy += vac_spacing;
    }
    vx += vac_spacing;
}
```

**Step 3: Build and verify**

Run: `cargo build`
Expected: Compiles without errors.

**Step 4: Commit**

```bash
git add src/field.rs
git commit -m "v0.6: Replace vacuum grid with atmospheric column around dino"
```

---

### Task 4: Add Per-Tick Property Modulation

**Files:**
- Modify: `src/field.rs` — `tick()` method, insert before Phase 0 (around line 1030)

**Step 1: Add atmospheric modulation pass**

Insert this block at the start of `tick()`, after `self.compute_active_set(view_proj);` and before the Phase 0 decay section:

```rust
// Atmospheric column modulation: update vacuum scatter/magnitude based on
// distance from current AABB center. Gives "follow AABB" behavior.
{
    let aabb_center = (self.aabb_min + self.aabb_max) * 0.5;
    let aabb_half = (self.aabb_max - self.aabb_min) * 0.5;
    let column_radius = aabb_half.x.max(aabb_half.z) * 1.5;
    let inv_radius = if column_radius > 0.1 { 1.0 / column_radius } else { 0.0 };

    for entity in &mut self.entities {
        if !entity.is_vacuum || entity.base_scatter <= 0.0 { continue; }
        let dx = entity.position.x - aabb_center.x;
        let dz = entity.position.z - aabb_center.z;
        let horiz_dist = (dx * dx + dz * dz).sqrt();
        let radial_frac = (horiz_dist * inv_radius).clamp(0.0, 1.0);
        let falloff = 1.0 - radial_frac * radial_frac;
        entity.scatter = entity.base_scatter * falloff;
        entity.deposit_magnitude = entity.base_magnitude * falloff;
    }
}
```

**Step 2: Build and verify**

Run: `cargo build`
Expected: Compiles without errors.

**Step 3: Commit**

```bash
git add src/field.rs
git commit -m "v0.6: Per-tick atmospheric modulation by AABB distance"
```

---

### Task 5: Build, Run, and Visual Verification

**Step 1: Full build**

Run: `cargo build --release`
Expected: Compiles without errors or warnings.

**Step 2: Run and visually verify**

Run: `cargo run --release`

Check:
- Observer starts closer to the dino (~38 cells from face)
- Atmospheric haze is visible around the dino body, fading outward
- Ground shadows under/behind the dino are darker than before
- Sun still illuminates the dino's upper surfaces
- Sky gradient still renders correctly
- No visual artifacts from overlapping vacuum/solid entities
- Entity count in startup logs shows fewer vacuum entities than before

**Step 3: Commit all together**

```bash
git add -A
git commit -m "v0.6: Atmosphere column & shadow tuning - visual verification pass"
```

(Only if there are fixes needed from visual testing. If Tasks 1-4 are clean, this commit is skipped.)

---

### Task 6: Update ROADMAP.md

**Files:**
- Modify: `ROADMAP.md:47-50`

**Step 1: Mark v0.6 as complete**

Replace the v0.6 section:

```markdown
## v0.6: Shadow tuning & atmosphere ✅
- Atmospheric column: concentrated vacuum relay network around dino AABB (replaces broad grid, ~60% fewer vacuum entities)
- Radial density profile: scatter/magnitude peaked at dino center, Gaussian-like falloff
- Per-tick atmospheric modulation: scatter/magnitude follow live AABB center
- Shader ambient floor: 0.25 → 0.10 for deeper ground shadows
- Observer start position: z=380 → z=310 for closer view
```

**Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "v0.6: Update roadmap with atmosphere & shadow tuning"
```
