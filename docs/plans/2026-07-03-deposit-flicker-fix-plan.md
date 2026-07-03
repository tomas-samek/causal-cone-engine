# Deposit Flicker Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the jaw red/green strobing and the ~3 Hz whole-body pulse by making footprint clears order-independent and feathering the gaussian deposit cutoff.

**Architecture:** Phase 3 of `DiffField::tick()` splits into Pass 3a (move, animate, clear old footprints, queue deposit positions in a reused scratch buffer) and Pass 3b (replay the queue through the existing deposit kernels). All clears complete before any deposit lands. Separately, the gaussian kernel's hard cutoff at exponent 4.0 gains a smoothstep window over exponent 3→4 so boundary cells fade instead of popping.

**Tech Stack:** Rust 2021, existing `field.rs` machinery only. No new dependencies.

**Spec:** `docs/plans/2026-07-03-deposit-flicker-fix-design.md` (approved).

## Global Constraints

- Semantics preserved exactly except the two fixes: same entities move/clear/deposit under the same skip conditions (render-depth, heat, vacuum, visibility); same clear-box extents; same deposit magnitudes/boosts; oscillation phase still advances exactly once per tick, after `deposit_pos` is computed.
- The scratch buffer is a reused allocation on `DiffField` (`deposit_queue`), cleared each tick — no per-tick `Vec` allocation.
- Feather window: 1.0 for exponent ≤ 3.0, smoothstep down to exactly 0.0 at 4.0; cells with exponent ≥ 4.0 still skip.
- Build/verify with `cargo build --release`; fast tests `cargo test`; ignored field tests `cargo test --release -- --ignored` (need ~2 GB, release).
- Only `src/field.rs` changes.

## File Structure

- **Modify** `src/field.rs` only:
  - `DiffField` struct + `new()`: add private `deposit_queue: Vec<(usize, glam::Vec3)>`.
  - `tick()` Phase 3 (currently `src/field.rs:1813-2068`): split into Pass 3a / Pass 3b.
  - New module-level `fn cutoff_window(exponent: f32) -> f32` (near `evaluate_metaball_field`) + unit tests in the existing `#[cfg(test)] mod tests`.

---

### Task 1: Phase 3 split — order-independent footprint clears

**Files:**
- Modify: `src/field.rs:~215` (DiffField struct), `src/field.rs:~330` (`new()` init), `src/field.rs:1813-2068` (Phase 3)
- Test: existing ignored integration tests (`walker_group_moves_rigidly`, `cross_links_follow_walker`) must pass unchanged

**Interfaces:**
- Consumes: existing fields/helpers only (`visible_set`, `active_set`, `consumption_states`, `in_bounds`, `index`, `dirty_slabs`).
- Produces: private `deposit_queue: Vec<(usize, glam::Vec3)>` on `DiffField`; Pass 3b's gaussian kernel body is where Task 2's feather lands.

- [ ] **Step 1: Add the scratch buffer**

In `struct DiffField`, after `link_radiation_dist: f32,`, add:

```rust
    /// Phase 3 scratch: (entity index, final deposit position) queued by
    /// pass 3a (move/animate/clear) and replayed by pass 3b (deposit).
    /// Reused allocation — cleared each tick.
    deposit_queue: Vec<(usize, glam::Vec3)>,
```

In `DiffField::new()`'s struct literal, after `link_radiation_dist: 0.0,`, add:

```rust
            deposit_queue: Vec::new(),
```

- [ ] **Step 2: Replace Phase 3 with the two-pass version**

Replace everything from the line `// Phase 3: entities deposit to grid (only visible entities)` (field.rs:1813) through the AABB assignment lines (field.rs:2069-2070, `self.aabb_min = ...; self.aabb_max = ...;`) with the code below. The `walk_delta` / `walk_offset_z` block just above (field.rs:1807-1811) stays where it is. Everything inside is today's code, re-arranged — the only new statements are the queue push, the queue loop, and the moved oscillation-phase advance.

```rust
        // Phase 3a: move, animate, and clear old footprints.
        // Every clear must land before ANY deposit: a clear that runs
        // mid-loop wipes same-tick contributions already written by
        // earlier-sorted overlapping entities (mouth under jaw), strobing
        // their colors whenever a base cell shifts. Deposit positions are
        // queued here and replayed in pass 3b.
        let mut aabb_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut aabb_max = glam::Vec3::splat(0.0);
        self.deposit_queue.clear();
        for (ent_idx, entity) in self.entities.iter_mut().enumerate() {
            // Move entity (all entities, not just visible — keeps positions consistent).
            // Walkers get the shared rigid displacement; their own velocity stays zero
            // so the per-axis bounce below can never tear the group apart.
            if entity.is_walker {
                entity.position += walk_delta;
            }
            entity.position += entity.velocity;

            // Bounce
            for i in 0..3 {
                if entity.position[i] < 1.0 || entity.position[i] >= (FIELD_SIZE - 1) as f32 {
                    entity.velocity[i] *= -1.0;
                    entity.position[i] = entity.position[i].clamp(1.0, (FIELD_SIZE - 2) as f32);
                }
            }

            // Heat: interior, light can't escape. Always skip.
            // Progressive rendering: skip entities deeper than cutoff
            if ent_idx < self.consumption_states.len() {
                if let Some(ref state) = self.consumption_states[ent_idx] {
                    if state.depth > self.render_depth_cutoff {
                        continue;
                    }
                }
            }

            if entity.is_heat { continue; }

            // Vacuum with atmosphere: scatter into grid only if active.
            // (Stays in 3a: scatter never clears, and atmosphere cells don't
            // meaningfully overlap the skeleton's clear boxes.)
            if entity.is_vacuum {
                if !self.active_set[ent_idx] { continue; }
                if entity.scatter > 0.0 && entity.incoming.density > 0.1 {
                    let ix = entity.position.x as i32;
                    let iy = entity.position.y as i32;
                    let iz = entity.position.z as i32;
                    if Self::in_bounds(ix, iy, iz) {
                        let idx = Self::index(ix as u32, iy as u32, iz as u32);
                        let s = entity.scatter;
                        let intensity = entity.incoming.density * s;
                        let cell = &mut self.cells[idx];
                        cell.density = (cell.density + intensity).min(50.0);
                        // Scatter uses air's own color (blue Rayleigh), not incoming color
                        cell.color_r = (cell.color_r + entity.color[0] * intensity).min(50.0);
                        cell.color_g = (cell.color_g + entity.color[1] * intensity).min(50.0);
                        cell.color_b = (cell.color_b + entity.color[2] * intensity).min(50.0);
                        self.dirty_slabs[iz as usize] = true;
                    }
                }
                continue;
            }

            // Track AABB from non-vacuum entities (tight box around solid geometry)
            let extent = if entity.deposit_radii != glam::Vec3::ZERO {
                entity.deposit_radii * 2.0
            } else {
                glam::Vec3::splat(1.0)
            };
            aabb_min = aabb_min.min(entity.position - extent);
            aabb_max = aabb_max.max(entity.position + extent);

            // Skip deposit for non-visible solid entities (reactive: only render subscribed chains)
            if !self.visible_set[ent_idx] { continue; }

            // Skin texture: offset deposit along surface normal
            let mut deposit_pos = entity.position;
            if entity.oscillation_amplitude > 0.0 {
                let offset = entity.surface_normal * entity.oscillation_phase.sin() * entity.oscillation_amplitude;
                deposit_pos += offset;
            }

            // Tail wag: shift deposit position in X via sine wave (adds on top of texture).
            // Tip has max amplitude, tapers toward body. Traveling wave along Z.
            if entity.group == GROUP_TAIL || entity.group == GROUP_TAIL_TIP {
                let time = self.tick as f32 / 30.0;
                let frequency = std::f32::consts::PI; // ~2 sec period
                // Anchor follows the walker so the wag taper stays body-relative
                let center_z = FIELD_SIZE as f32 / 2.0 + walk_offset_z;
                // z_frac: 0.0 at body junction (z=center), 1.0 at tail tip (z=center-24)
                let z_frac = ((center_z - entity.position.z) / 24.0).clamp(0.0, 1.0);
                let amplitude = 3.0 * z_frac; // tip swings 3 cells, body junction ~0
                let phase = time * frequency + z_frac * 2.0; // traveling wave
                deposit_pos.x += amplitude * phase.sin();
            }

            // Jaw open/close: rotate jaw downward around pivot at back of jaw.
            // Front of jaw swings down, back stays nearly fixed. Mouth follows.
            if entity.group == GROUP_JAW || entity.group == GROUP_MOUTH {
                let time = self.tick as f32 / 30.0;
                let frequency = std::f32::consts::PI * 0.5; // ~4 sec full cycle
                let center = FIELD_SIZE as f32 / 2.0;
                let pivot_z = center + 8.0 + walk_offset_z; // back of jaw, follows walker

                // z_frac: 0 at pivot (back), 1 at front of jaw
                let z_frac = ((entity.position.z - pivot_z) / 8.0).clamp(0.0, 1.0);

                // Jaw only opens DOWN (abs), never pushes up into head
                let open_amount = (time * frequency).sin().abs();
                // 2.5 cells at the snout tip — enough to clear the head
                // gaussian's tail so the opening reads at the iso-surface
                deposit_pos.y -= z_frac * 2.5 * open_amount;
            }

            // Advance oscillation phase (once per tick, after deposit_pos
            // was computed with the current phase — same timing as before)
            entity.oscillation_phase += entity.oscillation_freq;

            // Clear-box extents match the deposit kernel's extents.
            let use_gaussian = entity.deposit_radii != glam::Vec3::ZERO;
            let (half_x, half_y, half_z) = if use_gaussian {
                ((entity.deposit_radii.x * 2.0).ceil() as i32,
                 (entity.deposit_radii.y * 2.0).ceil() as i32,
                 (entity.deposit_radii.z * 2.0).ceil() as i32)
            } else {
                (1i32, 1i32, 1i32)
            };

            let base_x = deposit_pos.x.floor() as i32;
            let base_y = deposit_pos.y.floor() as i32;
            let base_z = deposit_pos.z.floor() as i32;

            // Clear previous footprint if base cell changed
            let new_base_idx = if Self::in_bounds(base_x, base_y, base_z) {
                Self::index(base_x as u32, base_y as u32, base_z as u32) as i32
            } else { -1 };
            if entity.prev_deposit_idx >= 0 && entity.prev_deposit_idx != new_base_idx {
                let prev = entity.prev_deposit_idx as usize;
                let pz = (prev / (FIELD_SIZE * FIELD_SIZE) as usize) as i32;
                let py = ((prev % (FIELD_SIZE * FIELD_SIZE) as usize) / FIELD_SIZE as usize) as i32;
                let px = (prev % FIELD_SIZE as usize) as i32;
                for dz in -half_z..=half_z {
                    for dy in -half_y..=half_y {
                        for dx in -half_x..=half_x {
                            let cx = px + dx;
                            let cy = py + dy;
                            let cz = pz + dz;
                            if Self::in_bounds(cx, cy, cz) {
                                let idx = Self::index(cx as u32, cy as u32, cz as u32);
                                self.cells[idx] = FieldCell::default();
                                self.dirty_slabs[cz as usize] = true;
                            }
                        }
                    }
                }
            }
            entity.prev_deposit_idx = new_base_idx;

            self.deposit_queue.push((ent_idx, deposit_pos));
        }
        self.aabb_min = aabb_min.max(glam::Vec3::ZERO);
        self.aabb_max = aabb_max.min(glam::Vec3::splat(FIELD_SIZE as f32));

        // Phase 3b: deposit. Every queued entity writes into a field whose
        // stale footprints are all cleared — deposit order no longer matters.
        for qi in 0..self.deposit_queue.len() {
            let (ent_idx, deposit_pos) = self.deposit_queue[qi];
            let entity = &self.entities[ent_idx];

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
            let tent_radius = 1.5f32; // only used for non-gaussian

            let base_x = deposit_pos.x.floor() as i32;
            let base_y = deposit_pos.y.floor() as i32;
            let base_z = deposit_pos.z.floor() as i32;

            let mag = entity.deposit_magnitude;
            // Consumption mass boost: entities that consume more deposit denser
            let mag = if ent_idx < self.consumption_states.len() {
                if let Some(ref state) = self.consumption_states[ent_idx] {
                    if !state.learning && state.consumed > 0 {
                        mag * (1.0 + (state.consumed as f32).ln().max(0.0) * 0.05)
                    } else { mag }
                } else { mag }
            } else { mag };
            let absorbed = 1.0 - entity.pass_through;

            // Trie depth visualization: override entity color with depth rainbow
            let entity_color = if self.show_trie_depth {
                if ent_idx < self.consumption_states.len() {
                    if let Some(ref state) = self.consumption_states[ent_idx] {
                        crate::consumption::depth_color(state.depth)
                    } else { [0.3, 0.3, 0.3] }
                } else { [0.3, 0.3, 0.3] }
            } else {
                entity.color
            };

            let total_r = entity_color[0] * mag + entity.incoming.r * absorbed * entity_color[0] + entity.reemit_r;
            let total_g = entity_color[1] * mag + entity.incoming.g * absorbed * entity_color[1] + entity.reemit_g;
            let total_b = entity_color[2] * mag + entity.incoming.b * absorbed * entity_color[2] + entity.reemit_b;
            let total_d = mag + entity.incoming.density * absorbed;

            // Decoupled boost: body gets high density boost (opaque surface) with
            // moderate color boost (natural brightness, no overexposure).
            let is_body = use_gaussian;
            let (density_boost, color_boost) = if is_body { (40.0, 10.0) } else { (10.0, 10.0) };
            let total_r = total_r * color_boost;
            let total_g = total_g * color_boost;
            let total_b = total_b * color_boost;
            let total_d = total_d * density_boost;

            if use_gaussian {
                // Skeleton entity: anisotropic gaussian deposit.
                // weight = exp(-((dx/rx)² + (dy/ry)² + (dz/rz)²))
                let rx = entity.deposit_radii.x;
                let ry = entity.deposit_radii.y;
                let rz = entity.deposit_radii.z;
                let inv_rx2 = 1.0 / (rx * rx);
                let inv_ry2 = 1.0 / (ry * ry);
                let inv_rz2 = 1.0 / (rz * rz);
                for dz in -half_z..=half_z {
                    let cz = base_z + dz;
                    if cz < 0 || cz >= FIELD_SIZE as i32 { continue; }
                    let fz = cz as f32 + 0.5 - deposit_pos.z;
                    let ez = fz * fz * inv_rz2;
                    for dy in -half_y..=half_y {
                        let cy = base_y + dy;
                        if cy < 0 || cy >= FIELD_SIZE as i32 { continue; }
                        let fy = cy as f32 + 0.5 - deposit_pos.y;
                        let eyz = fy * fy * inv_ry2 + ez;
                        if eyz > 4.0 { continue; } // exp(-4) ≈ 0.02, skip negligible
                        for dx in -half_x..=half_x {
                            let cx = base_x + dx;
                            if cx < 0 || cx >= FIELD_SIZE as i32 { continue; }
                            let fx = cx as f32 + 0.5 - deposit_pos.x;
                            let exponent = fx * fx * inv_rx2 + eyz;
                            if exponent > 4.0 { continue; }
                            let w = (-exponent).exp();
                            let idx = Self::index(cx as u32, cy as u32, cz as u32);
                            let cell = &mut self.cells[idx];
                            cell.density = (cell.density + total_d * w).min(50.0);
                            cell.color_r = (cell.color_r + total_r * w).min(50.0);
                            cell.color_g = (cell.color_g + total_g * w).min(50.0);
                            cell.color_b = (cell.color_b + total_b * w).min(50.0);
                            self.dirty_slabs[cz as usize] = true;
                        }
                    }
                }
            } else {
                // Tent kernel for floor/rock entities
                for dz in -half_z..=half_z {
                    let cz_f = base_z as f32 + dz as f32 + 0.5;
                    let wz = (tent_radius - (cz_f - deposit_pos.z).abs()).max(0.0);
                    for dy in -half_y..=half_y {
                        let cy_f = base_y as f32 + dy as f32 + 0.5;
                        let wy = (tent_radius - (cy_f - deposit_pos.y).abs()).max(0.0);
                        for dx in -half_x..=half_x {
                            let cx_f = base_x as f32 + dx as f32 + 0.5;
                            let wx = (tent_radius - (cx_f - deposit_pos.x).abs()).max(0.0);
                            let w = wx * wy * wz;
                            if w < 0.001 { continue; }
                            let cx = base_x + dx;
                            let cy = base_y + dy;
                            let cz = base_z + dz;
                            if Self::in_bounds(cx, cy, cz) {
                                let idx = Self::index(cx as u32, cy as u32, cz as u32);
                                let cell = &mut self.cells[idx];
                                cell.density = (cell.density + total_d * w).min(50.0);
                                cell.color_r = (cell.color_r + total_r * w).min(50.0);
                                cell.color_g = (cell.color_g + total_g * w).min(50.0);
                                cell.color_b = (cell.color_b + total_b * w).min(50.0);
                                self.dirty_slabs[cz as usize] = true;
                            }
                        }
                    }
                }
            }
        }
```

Note: the old `// Advance oscillation phase` line at the end of the old loop (field.rs:2066-2067) is covered by the moved advance in pass 3a — make sure it does not survive twice. The trie-diagnostics block and `self.tick += 1;` after the old AABB assignment stay unchanged, now following pass 3b.

- [ ] **Step 3: Build and run fast tests**

Run: `cargo build --release` — expected: clean (5 pre-existing dead-code warnings only).
Run: `cargo test` — expected: `13 passed; 0 failed; 2 ignored`.

- [ ] **Step 4: Run the ignored integration tests**

Run: `cargo test --release -- --ignored`
Expected: both `walker_group_moves_rigidly` and `cross_links_follow_walker` PASS (movement semantics untouched; takes a few minutes, ~2 GB).

- [ ] **Step 5: Commit**

```bash
git add src/field.rs
git commit -m "fix: order-independent footprint clears (Phase 3 split into clear + deposit passes)"
```

---

### Task 2: Feathered gaussian cutoff

**Files:**
- Modify: `src/field.rs` — new module-level function near `evaluate_metaball_field` (~line 231); one-line change in the Pass 3b gaussian kernel from Task 1
- Test: new unit tests in the existing `#[cfg(test)] mod tests` at the bottom of `src/field.rs`

**Interfaces:**
- Consumes: Pass 3b gaussian kernel from Task 1 (the `let w = (-exponent).exp();` line).
- Produces: `fn cutoff_window(exponent: f32) -> f32` (module-private, unit-tested).

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/field.rs`:

```rust
    #[test]
    fn cutoff_window_is_one_inside_and_zero_at_cutoff() {
        assert_eq!(cutoff_window(0.0), 1.0);
        assert_eq!(cutoff_window(3.0), 1.0);
        assert_eq!(cutoff_window(4.0), 0.0);
        assert!((cutoff_window(3.5) - 0.5).abs() < 1e-6); // smoothstep midpoint
    }

    #[test]
    fn cutoff_window_fades_monotonically_and_continuously() {
        // Monotone decreasing across the feather band
        let mut prev = cutoff_window(3.0);
        for i in 1..=20 {
            let e = 3.0 + i as f32 * 0.05;
            let cur = cutoff_window(e);
            assert!(cur <= prev, "window not monotone at exponent {}", e);
            prev = cur;
        }
        // Continuous at the cutoff: value just inside 4.0 is already tiny,
        // so a cell crossing the boundary cannot pop
        assert!(cutoff_window(3.999) < 1e-4);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test cutoff_window`
Expected: COMPILE ERROR — `cutoff_window` not found (red state).

- [ ] **Step 3: Implement the window and wire it into the kernel**

Add at module level in `src/field.rs`, right before `impl DiffField` (~line 274):

```rust
/// Feather window for the gaussian deposit cutoff: 1.0 below exponent 3.0,
/// smoothstep down to exactly 0.0 at 4.0 (where the kernel stops sampling).
/// Without it, boundary cells carry ~exp(-4)×boosted-magnitude — far above
/// the iso threshold — and pop between "deposited" and "decaying" as a
/// gaussian drifts sub-cell, which reads as edge shimmer.
fn cutoff_window(exponent: f32) -> f32 {
    let t = (4.0 - exponent).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
```

In Pass 3b's gaussian kernel (Task 1 code), change:

```rust
                            if exponent > 4.0 { continue; }
                            let w = (-exponent).exp();
```

to:

```rust
                            if exponent >= 4.0 { continue; }
                            // Feathered cutoff: fade to zero over exponent 3→4
                            // so cells stop popping as the gaussian drifts
                            let w = (-exponent).exp() * cutoff_window(exponent);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test cutoff_window`
Expected: `2 passed`.
Run: `cargo test` — expected: `15 passed; 0 failed; 2 ignored`.
Run: `cargo build --release` — expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/field.rs
git commit -m "fix: feather gaussian deposit cutoff (smoothstep over exponent 3..4)"
```

---

## Final Verification (after both tasks)

- [ ] `cargo test` — 15 passed.
- [ ] `cargo test --release -- --ignored` — both field integration tests pass.
- [ ] `cargo run --release` — smoke-run ~10 s: app ticks, FPS in the usual ~150 range (the split adds no per-cell work).
- [ ] Human visual check (this session cannot capture the framebuffer): jaw opens without red/green strobing; no ~3 Hz whole-body pulse while walking; silhouette edges stop shimmering.
