# Causal Cone Engine — Architecture

> A personal pet project. This document describes how the engine actually works
> today. For where it might go next, see [ROADMAP.md](ROADMAP.md).

## Core Principle

There are no rays cast into a scene, no triangle meshes, and no scene graph.
There is a **dense 3D field** of accumulated deposits and an **observer** moving
through it. Each frame samples the field; what you see is what has already been
deposited along your line of sight.

Two things hold the world:

- **The grid** — a dense `512³` array of `FieldCell { density, r, g, b }`
  (~134M cells, ~2 GB at f32). This is the observer's *retina*: the only thing
  the GPU ever reads. It is regenerated, not persisted, each tick.
- **The entity graph** — entities (nodes) connected by directed edges (pipes).
  Light does **not** diffuse cell-to-cell through the grid; it flows along the
  graph between entities. The grid only ever receives the *result* of that flow.

```
Entity --emits/relays--> EdgeDeposit on pipe --delivered--> neighbour Entity
Entity --deposits color--> grid cell   (only what the observer can see)
Observer --marches--> grid cell        (one iso-surface hit per pixel)
```

## Data Model

### FieldCell (`field.rs`)
```rust
struct FieldCell { density: f32, color_r: f32, color_g: f32, color_b: f32 }
```
A flat `Vec<FieldCell>` of length `512³`. Uploaded to the GPU as an
`Rgba16Float` 3D texture (density in R, color in GBA).

### Entity (`field.rs`)
A point that participates in light transport. Key fields:
- `position`, `velocity`, `color`, `deposit_magnitude`
- `pass_through` — fraction of incoming light that continues through it
- `reemit` — fraction of absorbed light re-emitted as its own color
- `scatter` / `base_scatter` — atmosphere: fraction bled into the grid
- `is_heat` — interior entity whose light can't escape the absorbing skin;
  conducts through the graph but never deposits to the grid
- `is_vacuum` — invisible relay (sun, atmosphere) that moves light but isn't drawn
- `specular`, oscillation params (for slow skin-texture shimmer), `deposit_radii`
- `edge_start` / `edge_count` — slice into the SoA edge arrays
- `incoming` / `incoming_dir` — what arrived this tick

### Edges — structure-of-arrays (`field.rs`)
Edges are stored as parallel flat arrays for cache-friendly iteration, not as
per-entity `Vec`s:
`edge_targets`, `edge_deposits` (the `EdgeDeposit` in each pipe), `edge_gammas`
(per-edge conductance weight), `edge_dirs` (normalized source→target). A reverse
index (`reverse_*`) lets a target find its incoming edges.

### Consumption trie (`consumption.rs`)
A separate, optional learning layer running parallel to the body entities:
- `DepositToken` — an incoming deposit quantized to 4-bit density + RGB levels.
- `Spectrum` — the set of tokens an entity "recognizes," crystallized from the
  most frequent tokens covering `TARGET_COVERAGE` (50%) of observations.
- `ConsumptionState` / `Seed` / `cascade_process` — each body entity routes its
  incoming token through a trie: tokens in its spectrum are *consumed* (blended
  toward the entity's own color); rejects cascade to a child; persistent rejects
  seed a new child state one level deeper (up to `MAX_TRIE_DEPTH = 20`).
This drives the trie-depth diagnostics (`T`/`I`) and progressive rendering
(`[` / `]` adjust `render_depth_cutoff`).

## Per-Tick Pipeline (CPU, 30 ticks/sec)

`DiffField::tick(view_proj)` runs a fixed-timestep pipeline. The recurring theme
is **do work only where the observer can see** — this is the practical form of
the "causal cone": chains that don't feed a visible pixel are skipped.

| Phase | What happens |
|-------|--------------|
| **Active set** | `compute_active_set` extracts frustum planes (Gribb–Hartmann) and marks each entity `active` (participates in transport) and `visible` (deposits to grid). Emitters like the sun are always active; heat entities never are. |
| **Atmosphere modulation** | Vacuum relay entities' `scatter`/`magnitude` are modulated by distance from the current geometry AABB center, so the atmosphere column follows the subject. |
| **Phase 0 — decay** | Cells inside the AABB are multiplied by 0.85 (tiny values cleared); slabs outside the AABB are cleared. Only *dirty* slabs are touched. |
| **Phase 1 — deliver** | Each edge's `EdgeDeposit` is pushed into its target's accumulator (active targets only). Targets apply incoming, compute incoming direction, update a debounce counter, and build re-emission energy. |
| **Consumption** | Each body entity's incoming is tokenized and run through `cascade_process` (consume / reject / seed / promote). |
| **Phase 2 — push** | Each active, non-debounced entity rewrites its outgoing edge deposits = own emission + pass-through of incoming, weighted by `edge_gamma × distance_factor`, with optional directional bias for vacuum relays. Parallelized with `rayon` (each entity owns a disjoint edge range). |
| **Phase 3 — deposit** | Entities move (and bounce off bounds). Heat and too-deep entities are skipped; vacuum entities scatter into the grid; visible solids deposit color/density into cells. Dirty slabs and the new AABB are recorded. |

## GPU Upload (`renderer.rs`)

Only what changed crosses the bus. When the tick advances, for each **dirty
slab** the renderer converts the AABB sub-rectangle of `FieldCell`s from f32 to
`f16` and `write_texture`s just that sub-region into the 3D texture. Empty space
and unchanged slabs cost nothing.

## GPU Render (`shaders/field_sample.wgsl`)

A single fullscreen triangle; all the work is in the fragment shader. Present
mode is uncapped (`AutoNoVsync`).

```
For each pixel:
  reconstruct world ray from inv_view_proj
  ray ∩ AABB (slab test) ───────────── miss → sky background
  march with growing step (0.5 × 1.02^i, ≤192 steps, clipped to AABB/200)
    first sample with density ≥ iso (0.3)?
      └─ 12-iteration bisection → sub-voxel iso crossing (crisp silhouette)
         shade surface:
           normal = density gradient (central differences)
           if creature (green): procedural reptile skin
             - Voronoi scales at two frequencies + normal perturbation
             - fbm mottling, dorsal stripe, warm belly tint
           Lambert diffuse + ambient floor + rim + specular (fixed sun_dir)
  composite over sky gradient (zenith/horizon/ground + sun glow)
  velocity vignette → ACES tone map → gamma
```

The iso-surface is found by **bisection**, not fog/alpha accumulation, so the
silhouette stays sharp and halo-free — sub-`iso` Gaussian tails are never drawn.

## Observer (`observer.rs`)

Free-fly camera with acceleration + drag. `c = 1 cell/tick = 30 cells/sec` at
30 ticks/sec; the observer is capped at `MAX_SPEED = 0.5c`. Effective FOV narrows
with speed (a linear approximation of relativistic aberration), and the shader
darkens screen edges at speed — fewer diffs reach you per tick the faster you go.

## Demo Scene (`spawn_demo_scene`)

The dinosaur's **density** comes from a *skeleton*: ~16 metaball-source entities
(body, belly, tail, neck, head, jaw, mouth, eyes, legs, feet) plus ~11 midpoint
entities at the joints. Each deposits a wide **anisotropic gaussian** blob
(`deposit_radii`) and the overlapping blobs merge into a continuous, seamless
density field — far cheaper than flood-filling the volume. Its **lighting** comes
from a separate **receptor shell**: lightweight surface entities (placed by BFS
surface-detection of the metaball field) that absorb most incoming light and
re-emit ~30% as color via radiation links. Entities fully enclosed by opaque
neighbors are turned to **heat** (conduct through the graph, never drawn).

The scene is lit by a **sun disc** of vacuum emitters pushing light through the
graph, wrapped in an **atmosphere** column of vacuum relays that scatter a little
blue light into the grid, and stands on a **40×40 dirt/grass floor** beside a
rock. The sky and sun glow are procedural background in the fragment shader.

## Theoretical Basis

Based on tick-frame physics: time is discrete, space is the diff field, photons
are stationary (they *are* the field updates), and mass is what gives you energy
to fight the substrate stream. The observer can't reach `c` — at `c` you're a
photon and no rendering is possible. The reactive active-set is the engine's
literal causal cone: only what can reach the observer is computed.
