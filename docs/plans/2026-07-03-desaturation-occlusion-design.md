# Desaturation Retune (B1) + Density-Attenuated Pipes (B2′) — Design

**Date:** 2026-07-03
**Status:** Approved for planning (B2′ revision — replaces the earlier
LOS-toggling B2)

## Problems

1. **Saturation flattens everything.** Skeleton deposits equilibrate at
   `deposit/(1−0.85)` and pin the 50-cap deep into every blob's 2×-radii
   box, while the iso threshold is 0.3. Consequences: the visible surface
   sits at ~2× the nominal radius (blobby body), the jaw swings inside
   solid density (opening invisible), and lit-vs-shaded contrast on the
   floor compresses.
2. **Occlusion is frozen at spawn — and is an oracle, not physics.**
   `build_radiation_links` ray-tests LOS once at startup and permanently
   discards blocked pairs. World↔world links blocked by the dino at spawn
   stay dark there forever. Philosophically the shadow should be a
   **deficit of deposits arriving**, caused by the deposited matter
   itself — not by an external ray-test toggling links.
3. *(Observed, accepted for now)* **Clear-dip transient.** Freshly cleared
   footprint cells restart from 1× deposit instead of the ~6.7× decay
   equilibrium, dipping the iso edge for ~0.5 s — a faint residual shadow
   flicker.

## B1a — Instrumentation (key `H`)

On `H`, one on-demand pass over the active AABB logs:
- Percentiles (50/90/99/max) of cell density and each color channel, plus
  the % of cells pinned at the 50.0 cap.
- Floor contrast probe: for `GROUP_FLOOR` entities, average `incoming`
  density and deposited cell color for tiles under the dino's AABB
  footprint vs tiles >20 cells away — identifies which transport path
  carries the ground pattern and how much contrast exists pre-render.

`log::info!` output; no per-tick cost.

## B1b — Live calibration keys

Two runtime multipliers on `DiffField`, defaulting to 1.0 and applied on
top of the existing hardcoded boosts (body 40/10, floor 10/10):
`tune_density` and `tune_color`. Keys: `1`/`2` halve/double
`tune_density`; `3`/`4` halve/double `tune_color`; each press logs all
tuning values. Density scaling is equivalent to moving the shader iso
threshold, so the shader is untouched. Clamp all tuning multipliers to
[1/1024, 1024].

**Checkpoint:** the user runs a calibration session (jaw opening reads,
silhouette tightens toward 1× radii, floor contrast visible, thin parts —
feet/arms at ~1-cell radii and heat-darkened entities at ×0.15 magnitude —
must not vanish). The chosen multipliers are then folded into the boost
constants; the keys remain as a permanent debug facility.

## B2′ — Density-attenuated pipes (deposits themselves occlude)

Radiation edges are pipes through space, and space is the diff field — so
a pipe crossing deposited matter should carry less. Occlusion becomes an
**arrival deficit driven by the actual field contents**, graded and
soft-edged, valid wherever the dino walks.

- **Rule:** every edge **longer than `link_connect_dist`** (exactly the
  radiation links; short conduction/skin edges are exempt) gets a
  per-edge `edge_atten: Vec<f32>` factor (default 1.0) multiplied
  alongside `edge_gammas` in the Phase 2 push:
  `transmittance = exp(−k · Σ max(0, density − ATTEN_THRESHOLD) · Δt)`
  sampled from the grid at ~1 sample/cell along the segment, **excluding
  1.5-cell margins at both ends** (an edge legitimately starts inside its
  own endpoint's surface density; light leaving a surface must not be
  strangled by its own source).
- **Constants/tunables:** `ATTEN_THRESHOLD = 0.6` (2× iso — floor-surface
  deposits and atmosphere haze below it don't attenuate; solid body
  interiors far above it do). `k` (`ATTEN_K`, default 0.5) is
  live-tunable via keys `5`/`6` (halve/double, same clamp and logging as
  the B1b keys) since its right value depends on the B1 calibration
  outcome.
- **`ray_blocked` is removed from radiation-link building** (startup and
  cross-link refresh both): links are built by distance alone and the
  field attenuates them. One mechanism, no oracle. The rock's occlusion
  becomes graded too. Side effect to watch at calibration: pairs that LOS
  used to discard now occupy slots under the 10-per-entity radiation cap;
  if body lighting dims, raising `max_radiation` (10 → ~14) is the
  expected remedy.
- **Cadence:** the grid is empty when the graph is built, so attenuation
  initializes at **tick 15** (field ≈ equilibrium) with a full pass over
  all long edges, then recomputes for all long edges at **every
  cross-link refresh** (existing ≥1 cell / ≥5 tick cadence; ~70k edges ×
  ~10 samples ≈ 1M grid reads, low single-digit ms). Between refreshes the
  field around a stationary dino is static enough that stale attenuation
  is negligible.
- **Purity for testing:** the sampler is a module-level
  `fn segment_attenuation(a: Vec3, b: Vec3, k: f32, threshold: f32,
  sample: impl Fn(i32, i32, i32) -> f32) -> f32` — pure over an injected
  density lookup, unit-testable without a 2 GB field.
- While touching `field.rs`: extract the shared `deposit_extents(radii:
  glam::Vec3) -> (i32, i32, i32)` helper used by pass 3a clears and pass
  3b deposits (final-review maintenance item — the two copies must never
  drift).

**Roadmap note (Level 2, out of scope):** the fully emergent version —
no long-range shortcuts at all, light hopping only node-to-node through
the vacuum medium at c, occlusion purely from graph topology and
absorption — is the engine's eventual destination. B2′ moves the occluder
from an external ray oracle into the deposited field itself, which is the
stepping stone.

## B1c — Clear-dip mitigation (optional, last)

When an entity's base cell changed this tick (it just cleared), scale its
own pass-3b deposit by `1/(1−0.85)` for that tick so its contribution
lands at equilibrium instantly. During the walk all dino entities clear on
the same tick, so the body reaches equilibrium together; contributions
from *static* neighbors into cleared cells still dip (partial fix,
accepted). Full fixes (per-cell cleared stamps, or a regenerative
zero-and-rewrite field) are future work. Ship only if the calibration
session shows the residual flicker still bothers at the new tuning.

## Non-goals

- No shader changes; no decay-constant change.
- No changes to connection-edge (≤ `link_connect_dist`) transport.
- The mouth-cavity carver (Option A) stays shelved unless B1 calibration
  proves insufficient for the jaw.
- Level 2 (medium-only transport) is roadmap, not this branch.

## Verification

- Pure unit tests: `segment_attenuation` with synthetic density closures —
  empty space → 1.0; a dense wall mid-segment → ≪1; density below
  threshold → 1.0; end-margin exclusion (dense only at endpoints → 1.0);
  monotone in k. Histogram percentile helper likewise.
- Ignored integration test: build the field, tick past the attenuation
  init (~20 ticks), then walk ~80 ticks; assert long edges whose segments
  cross the dino's current solid AABB have `edge_atten` well below 1.0
  while distant floor↔floor long edges stay ≈ 1.0 — at the *walked*
  position, not spawn.
- Existing 15 fast + 2 ignored tests stay green.
- Human: calibration session (B1b + `k` via keys `5`/`6`), then a final
  look — jaw opening visible, shadow present, soft-edged, and following
  the dino.

## Sequencing

One branch: B1a → B1b → **user calibration checkpoint** → bake constants →
B2′ (+`deposit_extents` helper) → recheck shadow with `k` tuning → B1c
only if still needed.
