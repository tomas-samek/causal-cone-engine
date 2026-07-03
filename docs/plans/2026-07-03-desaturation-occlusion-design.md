# Desaturation Retune (B1) + Occlusion Follow (B2) — Design

**Date:** 2026-07-03
**Status:** Approved for planning

## Problems

1. **Saturation flattens everything.** Skeleton deposits equilibrate at
   `deposit/(1−0.85)` and pin the 50-cap deep into every blob's 2×-radii
   box, while the iso threshold is 0.3. Consequences: the visible surface
   sits at ~2× the nominal radius (blobby body), the jaw swings inside
   solid density (opening invisible), and lit-vs-shaded contrast on the
   floor compresses.
2. **Occlusion is frozen at spawn.** `build_radiation_links` permanently
   discards LOS-blocked pairs at startup. World↔world links blocked by the
   dino's body at spawn stay dark there forever; links under the dino's
   new position stay bright. The cross-link refresh only rebuilds
   walker↔world edges, so the body's occlusion shadow cannot follow the
   walk.
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
`tune_density`; `3`/`4` halve/double `tune_color`; each press logs both
values. Density scaling is equivalent to moving the shader iso threshold,
so the shader is untouched. Clamp both to [1/1024, 1024].

**Checkpoint:** the user runs a calibration session (jaw opening reads,
silhouette tightens toward 1× radii, floor contrast visible, thin parts —
feet/arms at ~1-cell radii and heat-darkened entities at ×0.15 magnitude —
must not vanish). The chosen multipliers are then folded into the boost
constants; the keys remain as a permanent debug facility (they cost
nothing and future retunes will want them).

## B2 — Occlusion follows the walker

- **Startup:** world↔world radiation candidates are LOS-tested in two
  stages. Stage 1 against **static blockers only** (walker entities
  excluded from the block grid): blocked → permanently discarded, as
  today. Stage 2: surviving pairs whose segment intersects the walker's
  **swept volume** (dino solid AABB at spawn extended ±`WALK_SPAN` along
  Z, computed statically) are recorded in `occludable_pairs:
  Vec<(usize, usize)>` and get their *initial* linked/blocked state from a
  walker-only LOS test. Pairs outside the swept volume are ordinary
  permanent links.
- **Refresh:** on every existing cross-link refresh (≥1 cell travel, ≥5
  tick spacing), re-test all occludable pairs against a walker-only block
  grid built from current positions. In the rebuilt edge SoA: currently
  blocked pairs are dropped, clear pairs are (re-)added. Re-added pipes
  start empty and fill in one tick (same accepted ripple). Membership
  checks use a `HashSet<(usize, usize)>` built once at startup.
- Occludable links bypass the per-entity radiation cap bookkeeping (they
  were legitimate candidates at startup; same spirit as the existing
  plan-sanctioned cap divergence for cross links).
- The segment-vs-swept-AABB intersection test is a pure module-level
  function with unit tests.
- While touching `field.rs`: extract the shared `deposit_extents(radii:
  glam::Vec3) -> (i32, i32, i32)` helper used by pass 3a clears and pass
  3b deposits (final-review maintenance item — the two copies must never
  drift).

## B1c — Clear-dip mitigation (optional, last)

When an entity's base cell changed this tick (it just cleared), scale its
own pass-3b deposit by `1/(1−0.85)` for that tick so its contribution
lands at equilibrium instantly. During the walk all dino entities clear on
the same tick, so the body reaches equilibrium together; contributions
from *static* neighbors into cleared cells still dip (partial fix,
accepted). Full fixes (per-cell cleared stamps, or a regenerative
zero-and-rewrite field) are noted as future work. Ship only if the
calibration session shows the residual flicker still bothers at the new
tuning.

## Non-goals

- No shader changes; no decay-constant change; no light-graph topology
  changes beyond the occludable-pair toggling.
- The mouth-cavity carver (Option A) stays shelved unless B1 calibration
  proves insufficient for the jaw.

## Verification

- Pure unit tests: swept-AABB segment intersection (hit, miss, parallel,
  endpoint-inside cases); histogram percentile helper.
- Ignored integration test: record blocked occludable pairs at spawn; tick
  ~80 steps (walk ~6 cells with turnaround); assert the blocked set
  changed and every currently-blocked pair's segment passes through the
  dino's current solid AABB.
- Existing 15 fast + 2 ignored tests stay green.
- Human: calibration session (B1b checkpoint) and a final look — jaw
  opening visible, shadow contrast present and following the dino.

## Sequencing

One branch: B1a → B1b → **user calibration checkpoint** → bake constants →
B2 (+`deposit_extents` helper) → B1c only if still needed.
