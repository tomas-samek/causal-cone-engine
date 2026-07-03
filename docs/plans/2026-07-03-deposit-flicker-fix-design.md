# Deposit Flicker Fix — Design

**Date:** 2026-07-03
**Status:** Approved for planning

## Problem

The jaw region strobes between tongue red and body green as the jaw animates,
and the whole body pulses faintly at ~3 Hz while walking.

Root cause (confirmed by code reading, `field.rs` Phase 3):

1. **Order-dependent footprint clears.** When an entity's deposit base cell
   changes, it zeroes its *entire* footprint box (`FieldCell::default()` over
   the 2×-radii extent) — in the middle of the per-entity deposit loop.
   Overlapping entities that already deposited this tick (mouth under jaw,
   head over jaw) get their contribution wiped from the overlap cells for one
   tick, then restored next tick. The jaw's 2.5-cell swing crosses cell
   boundaries many times per cycle → red/green strobing. The walk makes all
   ~130 walker entities cross a boundary simultaneously every 10 ticks
   (0.1 cells/step) → synchronized whole-body clear-wipe at ~3 Hz.
2. **Hard gaussian cutoff.** Deposit kernels skip cells where the gaussian
   exponent exceeds 4.0. At the cutoff the contribution is still large
   (weight ≈ 0.018 × boosted magnitudes ≫ iso threshold), so cells pop
   between "deposited" and "decaying" as gaussians drift sub-cell —
   silhouette-edge shimmer.

## Fix 1: Order-independent footprint clears (Phase 3 split)

Split Phase 3 into two passes over the same entity set:

- **Pass 3a — move, animate, clear:** entity movement (walker delta +
  velocity + bounce) and all deposit-position math (oscillation offset, tail
  wag, jaw drop) run here, storing each depositing entity's final
  `deposit_pos` and new base index in a reused scratch buffer
  (`Vec<(entity_idx, deposit_pos, new_base_idx)>` on `DiffField`). Each
  entity whose base cell changed clears its old footprint box here, exactly
  as today (same box extents, same dirty-slab marking, `prev_deposit_idx`
  updated here).
- **Pass 3b — deposit:** iterate the scratch buffer and run the existing
  gaussian/tent deposit kernels using the stored `deposit_pos`.

All clears therefore complete before any deposit lands: the wipe can no
longer eat a same-tick contribution from an earlier-sorted neighbor.
Animation code is computed once (moved, not duplicated). The set of entities
that clear/deposit is unchanged (same visibility / heat / vacuum /
render-depth skip conditions, evaluated in pass 3a). AABB tracking stays in
pass 3a with the movement code; vacuum scatter deposits stay in pass 3a too
(they never clear and don't overlap the skeleton meaningfully).

## Fix 2: Feathered gaussian cutoff

In the gaussian deposit kernel, replace the hard skip at `exponent > 4.0`
with a smooth window: for `exponent` in `[3.0, 4.0]` multiply the weight by
`smoothstep(4.0, 3.0, exponent)` (i.e. `t = 4.0 - exponent` clamped to
`[0,1]`, window `t·t·(3−2t)`); above 4.0 contribute nothing (still skip).
Cell contributions now fade to exactly zero at the cutoff instead of
stepping from ~0.018×boost to nothing, so sub-cell drift no longer pops
cells. The box extents (2×radii) align with exponent 4 on the axes, so the
window also removes the box-face seam. Slight (<½ cell) silhouette
shrinkage is expected and acceptable.

## Non-goals

- No change to tick rates, decay constant, upload cadence, or the light
  graph — the multi-rate "fast substrate" architecture (Approach B) is
  explicitly deferred until this fix is judged by eye.
- No change to which entities deposit or their magnitudes.

## Verification

- Existing fast tests and both `#[ignore]`d release integration tests pass
  unchanged (positions and movement semantics untouched).
- `cargo run --release`: jaw opens without red/green strobing; no ~3 Hz
  whole-body pulse while walking; silhouette edges stop shimmering. (Human
  visual check — this session cannot capture the framebuffer.)
- FPS unchanged within noise (the split adds one scratch-buffer pass, no new
  per-cell work; the feather adds one multiply on boundary cells only).
