# Strongly Subluminal Dino — Design

**Date:** 2026-07-03
**Status:** Approved for planning

## Goal

Make the demo-scene dinosaur move at a strongly subluminal speed — at most
**1e-6 c** (deposits/light travel at c = 1 cell/world-tick) — while the motion
remains clearly perceivable to the observer in real time.

Today the dino is fully static: every dino entity is spawned with
`velocity = Vec3::ZERO`. Motion machinery exists (`entity.position +=
entity.velocity` plus per-axis bounce in Phase 3 of `field.rs`) but nothing
uses it.

## Core Idea: Time-Lapse World Clock

At 1e-6 c a body moves 1e-6 cells per world-tick. Rendered in real time
(1 tick = 1/30 s) it would take ~9 hours to cross one cell — imperceptible.
Real animals move at ~1e-9 c; perceiving such motion **requires** watching in
time-lapse. This is also the physically honest regime: at strongly subluminal
speeds the light field is quasi-static relative to the motion.

Two clocks are separated:

- **Sim step** — unchanged. 30 steps/sec wall-clock; one graph hop of light
  transport, one field decay + deposit pass per step.
- **World tick** — the physics unit where c = 1 cell/tick. A new field
  `time_lapse: u64` (default **100,000**, runtime-adjustable) declares how many
  world-ticks of world time elapse per sim step.

The dino's speed is stored honestly in cells per world-tick:
`DINO_SPEED_C = 1e-6`. Per sim step it moves
`DINO_SPEED_C × time_lapse = 0.1 cells`, i.e. **3 cells/sec on screen** —
the ~30-cell dino crosses its own body length in ~10 s.

**The dino moves every sim step (30×/sec) by 0.1 cells.** The time-lapse
factor is accounting, not scheduling — nothing waits N steps and jumps.
Intermediate world-tick positions within a step are not simulated; the
accumulated displacement is applied as one sub-cell shift, far below
perceptible stutter.

Light transport stays at one graph hop per sim step. At ×10⁵ lapse, light
crosses the entire 512-cell field in a tiny fraction of one sim step of world
time, so the light field is treated as **quasi-static** (adiabatic
approximation). The residual few-step convergence lag of the graph is
negligible against the 0.1-cell/step scene change.

The **observer stays in the wall-clock frame** — you are the fast being
watching a slow world. Its 0.5c speed cap, `speed()` readout, and FOV
vignette are untouched.

## Rigid Group Motion

- Every dino entity — skeleton metaballs, midpoints, receptor shell,
  heat entities — gets a new flag `is_walker: bool` set at spawn. A flag is
  required (not an index range) because entities are sorted by grid cell
  index after spawn, destroying spawn-order contiguity.
- A new `WalkController` owned by `DiffField` holds one shared walk vector
  (cells per world-tick, magnitude ≤ 1e-6) and walk bounds. Each sim step it
  applies the uniform displacement `walk_vector × time_lapse` to every walker
  entity's position.
- Walker entities keep `velocity = Vec3::ZERO`, so the existing per-entity
  move/bounce code in Phase 3 never touches them. This avoids the rigid body
  tearing itself apart on independent per-axis bounces.
- **v1 walks along ±Z only** (the dino's facing axis): forward until the
  group AABB reaches a floor-edge bound, then the walk vector flips sign —
  the dino paces back and forth without turning.
- Rigid **rotation is out of scope** for v1: cached `edge_dirs` and the
  axis-aligned anisotropic gaussian deposits (`deposit_radii`) would both
  need rotation support. Noted as future work.
- Rigid translation preserves all internal distances, so dino-internal
  connection edges, radiation links, edge gammas, heat classification, the
  receptor shell, and consumption-trie states all remain valid with no
  rework.
- Sub-cell smoothness is free: deposits are gaussians centered at f32
  positions, so 0.1-cell steps shift the density field (and thus the
  iso-surface) smoothly rather than snapping to voxels.

## Cross-Link Refresh (shadow follows the dino)

The light-transport graph is built once at startup from entity distances
(`build_connections`, `build_radiation_links`). Once the dino translates more
than ~3 cells, its links to the world (feet→floor tiles that create the
shadow, atmosphere→receptor shell) are stale and lighting stays anchored at
spawn.

- Track cumulative walker displacement since the last link build. When it
  exceeds **1.0 cell** (~every 10 steps ≈ 3×/sec at default lapse), refresh
  **walker↔world edges only**:
  - Dino-internal and world-internal edges are kept untouched.
  - Re-run the spatial-hash pair search restricted to (walker, non-walker)
    pairs, for both connection edges (`connect_dist`) and radiation links
    (including the line-of-sight blocking test).
  - Repack the SoA edge arrays (`edge_targets`, `edge_deposits`,
    `edge_gammas`, `edge_dirs`, reverse index) and recompute gammas. This is
    a full O(edges) rebuild of the flat arrays (~hundreds of thousands of
    edges, a few ms) — acceptable at this cadence; in-place per-entity
    splicing is not worth the complexity.
  - In-flight `EdgeDeposit`s are carried over for retained edges (keyed by
    src→tgt); new edges start empty and fill within one step. Worst case is
    a one-step lighting ripple near the feet at each refresh.

## Field Hygiene

Phase 0's 0.85/step density decay leaves a ~0.5 s fading trail behind the
moving body. At 0.1 cells/step the footprints overlap heavily, so this reads
as slight motion blur — **accepted for v1** (arguably the honest look for a
diff field). Dirty-slab tracking and outside-AABB clearing already handle the
moving AABB.

## Controls & HUD

- Window title gains: `dino v=1.0e-6c — lapse ×100000`.
- New keys `-` / `=`: halve / double `time_lapse`, clamped to
  [1, 1,048,576]. At ×1 the sim renders literal real-time 1e-6 c — the dino
  is effectively frozen (9 h/cell), which makes the subluminal regime
  experiential: dial the lapse down to feel it, back up to see motion.

## Constants

| Name | Value | Meaning |
|------|-------|---------|
| `DINO_SPEED_C` | `1e-6` | Dino speed in cells per world-tick (fraction of c) |
| `TIME_LAPSE_DEFAULT` | `100_000` | World-ticks per sim step |
| `TIME_LAPSE_MAX` | `1_048_576` | Upper clamp for `=` key (2^20) |
| `LINK_REFRESH_DIST` | `1.0` | Cells of walker travel between cross-link refreshes |

## Verification

- **Rigidity:** after N steps, all walker pairwise distances unchanged
  (within f32 ε); per-step displacement equals `walk_vector × time_lapse`
  exactly.
- **Turnaround:** walk vector flips at the floor bounds; all walker
  positions stay within field bounds.
- **Link refresh:** after walking ~10 cells, the floor tiles nearest the
  feet have edges from the feet, and the spawn-position tiles no longer do.
- **Visual:** shadow follows the dino; no body tearing; FPS holds
  (measure the SoA rebuild cost at refresh cadence).

## Out of Scope (future work)

- Rigid rotation / turning at bounds (needs rotating `edge_dirs` and
  oriented gaussian deposits).
- Gait animation (articulated legs/tail; stresses internal edges, receptor
  shell, and heat classification).
- Directional decay tuning to sharpen or stylize the motion trail.

## Docs to Update After Implementation

- `README.md` — "Demo Scene" says the dino "stands"; controls table gains
  `-`/`=`.
- `ARCHITECTURE.md` — Observer/tick section documents the two-clock model
  and the quasi-static light approximation; Phase 3 documents the
  `WalkController` and cross-link refresh.
