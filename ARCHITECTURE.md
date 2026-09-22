# Causal Cone Engine — Architecture

> A personal pet project. This document describes how the engine actually works
> today. For where it might go next, see [ROADMAP.md](ROADMAP.md).

## Core Principle

There are no rays cast into a scene, no triangle meshes, no scene graph, and
nothing cached in space. Light is **delivered**, never gathered.

Two things hold the world:

- **The entity graph** — entities (nodes) connected by directed edges (pipes).
  Light flows along the graph between entities, one hop per tick. Occlusion
  is *dimming along a pipe* by a transmittance `τ`, never a visibility test.
- **The retina** — a persistent `W×H` array of receptors on the image plane
  (`retina.rs`, default 320×180). Receptors are the observer's *state*, not a
  snapshot: each is the running sum of what entities have delivered to it. It
  never decays and is never rebuilt per frame — only deltas ever touch it.

```
Entity --emits/relays--> EdgeDeposit on pipe --delivered--> neighbour Entity
Entity --sends (new − last)--> receptor pipe --accumulates--> Receptor
Observer --reads--> Receptor            (no march; the pixel is already there)
```

## Data Model

### Receptor (`retina.rs`)
```rust
struct Receptor { density: i64, color: [i64; 3], normal: [i64; 3], depth: i64, skin: i64, sharp: i64 }
struct PipeQ    { density: i32, color: [i32; 3], normal: [i32; 3], depth: i32, skin: i32, sharp: i32 }
```
`W×H` of them, in fixed point: `Q_FRAC` = 16 fractional bits (one step ≈ 1.5e-5,
under the f16 step of the upload), `Q_DEPTH_FRAC` = 10 for `depth` (Σ depth·density,
which runs far larger). They **sum, never average**; the renderer converts back
to floats (`density_f()` …) and divides color, depth and skin by `sharp` on
upload, so the shader reads them already normalized.

Every pipe has **two weights**. `density` travels through the broad footprint
weight `w` and draws the silhouette. Everything else — and `sharp`, the density
again, which normalizes it — travels through the *sharp* weight
`(wᵖ + SHARP_FLOOR·w) · max(0, 1 − D/RETINA_ISO)`:
- `wᵖ` is the same gaussian at `1/√p` of the σ, and `p` is **per source**
  (`Source::sharp_power`, from `sharp_power_for`): as sharp as the source's
  lattice can carry. Halfway to its nearest neighbour (`Entity::neighbor_spacing`,
  found once at build) the sharp kernel must still weigh `exp(−SHARP_GAP)`, so
  `p = SHARP_GAP·(2r/spacing)²`, rounded, in `1..=SHARP_POWER_MAX`. The rock
  (0.8-cell lattice) gets 8, the floor (1.5) gets 2 — one global power either
  blurred the rock or broke the floor into dots. A source a
  few cells from the eye projects ~10 receptors wide; averaging colour and
  normal over all of that is what blurred near geometry. `SHARP_FLOOR` (1%)
  keeps the broad average as the fallback where only tails reach, so a
  silhouette receptor is never left with density and nothing to show.
- `D` is the density (`τ·density·w`, as of the relink) that the *same
  receptor's* pipes deliver from sources at least `FRONT_MARGIN` (1 cell) nearer
  the eye — front-to-back compositing, which a plain sum cannot do. It is the
  display rule read front to back: a receptor is drawn once it holds
  `RETINA_ISO` of density, so it shows the sources that make up the *first*
  `RETINA_ISO`, nearest first, and anything behind a full iso is behind a
  surface. It has to be delivered density, the quantity the silhouette is cut
  from, not 3D opacity: the dino skeleton's opacity is 800, and the last sliver
  of such a kernel — well outside the silhouette it draws — hid the rock behind
  it in a halo around the body, while unseen (τ = 0) sources punched black
  holes. It is done on the weights (`link_front`, at relink), not on the sum:
  weights are constants between relinks, so arrival stays delta-only and
  exactly reversible.

A pipe's float contribution is rounded
once, to a `PipeQ`, and each pipe remembers the `PipeQ` it last sent
(`pipe_last`), so the sum is reversible **bit for bit**: relinking subtracts
every pipe's last contribution and lands the array on exactly zero before
rebuilding. Pipes are i32 (one source through a weight ≤ 1, and the array
`arrive` rewrites every tick); receptors are i64 because they sum many pipes.
The adds wrap, so even an overflow would come back out exactly.

### Source (`retina.rs`)
One entity's contribution as the retina sees it, rebuilt each tick and index-
aligned with `DiffField::entities`: `position`, `radii` (the anisotropic
gaussian kernel), `normal`, `opacity` (static density, used for transmittance),
`density` + `color` (what it delivers this tick), and two independent flags —
`drawable` (gets pipes: solid, in frustum, not depth-culled) and `occluder`
(dims segments: solid, in or out of frustum).

### Pipes — structure-of-arrays (`retina.rs`)
Entity → receptor links, one contiguous slice per entity: `pipe_start` /
`pipe_count` index into `pipe_receptor` (which receptor), `pipe_weight` (the
projected gaussian's feathered weight there), and `pipe_last` (what this pipe
last delivered). Per entity, `entity_trans` holds `τ` toward the eye and
`entity_depth` the eye distance at the last relink. `τ` is integrated only over
the stretch of the entity→eye segment that lies inside the scene's geometry
AABB (grown by the widest kernel reach — everything outside it samples empty
space), and the walk stops at a hard `0.0` once `k·∫` passes `ATTEN_TAU_CUTOFF`.
So the cost does not grow with how far the observer has flown.

### Entity (`field.rs`)
A point that participates in light transport. Key fields:
- `position`, `velocity`, `color`, `deposit_magnitude`
- `pass_through` — fraction of incoming light that continues through it
- `reemit` — fraction of absorbed light re-emitted as its own color
- `scatter` / `base_scatter` — marks a relay as part of the atmosphere column
- `is_heat` — interior entity whose light can't escape the absorbing skin;
  conducts through the graph but never becomes a drawable source
- `is_vacuum` — invisible relay (sun, atmosphere) that moves light but isn't drawn
- `specular`, oscillation params (for slow skin-texture shimmer), `deposit_radii`
- `edge_start` / `edge_count` — slice into the SoA edge arrays
- `incoming` / `incoming_dir` — what arrived this tick

### Edges — structure-of-arrays (`field.rs`)
Edges are stored as parallel flat arrays for cache-friendly iteration, not as
per-entity `Vec`s:
`edge_targets`, `edge_deposits` (the `EdgeDeposit` in each pipe), `edge_gammas`
(per-edge conductance weight), `edge_dirs` (normalized source→target), and
`edge_atten` — per-edge transmittance `τ ∈ [0,1]`, the fraction of light that
survives the density between the two endpoints. Only **radiation** edges (longer
than `link_connect_dist`) are attenuated; connection edges stay at `1.0`. A
segment buried in a metaball core underflows to exactly `0` — a fully opaque
pipe. A reverse index (`reverse_*`) lets a target find its incoming edges.

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
| **Cross-link refresh** | When the walker (dino) has drifted ≥1 cell since links were last built, walker↔world connection and radiation edges are re-searched and the SoA edge arrays repacked; internal edges and their in-flight deposits are preserved. |
| **Active set** | `compute_active_set` extracts frustum planes (Gribb–Hartmann) and marks each entity `active` (participates in transport) and `visible` (may become a drawable source). Emitters like the sun are always active; heat entities never are. |
| **Atmosphere modulation** | Vacuum relay entities' `scatter`/`magnitude` are modulated by distance from the current geometry AABB center, so the atmosphere column follows the subject. |
| **Phase 1 — deliver** | Each edge's `EdgeDeposit` is pushed into its target's accumulator (active targets only). Targets apply incoming, compute incoming direction, update a debounce counter, and build re-emission energy. |
| **Consumption** | Each body entity's incoming is tokenized and run through `cascade_process` (consume / reject / seed / promote). |
| **Phase 2 — push** | Each active, non-debounced entity rewrites its outgoing edge deposits = own emission + pass-through of incoming, weighted by `edge_gamma × distance_factor × edge_atten`, with optional directional bias for vacuum relays. Weights are then **renormalized** to sum to 1 — see the shadow note below. Parallelized with `rayon` (each entity owns a disjoint edge range). |
| **Advance entities** | The walker group (dino) translates rigidly by `speed × time_lapse` and paces ±6 cells along Z; other entities move by velocity (and bounce off `FIELD_SIZE`). Each solid becomes a `Source` with its animated position, boosted density/color, and `drawable` flag, and the geometry AABB is recomputed. |
| **Relink** (conditional) | **Full** — if the cross-links refreshed, a tuning key fired, the AABB's projected corners moved ≥ `RELINK_SHIFT` (0.1 receptor), or a source linked as static moved: the retina drops every pipe (subtracting what it last sent, landing on exactly zero), re-projects every drawable source's gaussian footprint into image space (a lattice member whose projected σ exceeds `SHRINK_SIGMA` = 3 receptors draws with a **shrunk kernel** — toward that σ, but no smaller than `SHRINK_RADIUS_PER_SPACING` = 0.6× its lattice spacing, `Source::shrink_min` — and `contribution` scales what it delivers by 1/shrink²: the lattice's summed surface is unchanged, its pipes fall with the square, and a metaball, whose kernel is its shape, never shrinks; near the rock this took 1.5 M pipes to 0.6 M), recomputes `τ` toward the eye (along a segment from the source's *near surface* — `tau_origin`, the kernel ellipsoid's point nearest the eye — so a metaball's τ is not decided by one line through the siblings its centre is buried among), and sweeps the sources front to back, keeping a running per-receptor density, to set the pipes' sharp weights (`link_front` — one sweep per receptor band, the same bands `arrive` uses, in parallel: receptors do not talk to each other, and a row-major footprint's share of a band is one contiguous stretch of its run). A source **all** of whose receptors already sit behind `HIDDEN_AT` (4) isos of density is *hidden*: it could only add density where the picture is drawn without it, so it is linked with no pipes at all (the back and underside of a near rock — a quarter of the pipes there). A static source counts only static density for this, so nothing a mover hides is ever missing when the mover steps aside. **Partial** — if the camera held still and only *dynamic* sources' projected centers moved that far (the walking dino shifts ~0.17 receptors per tick, so this is every tick): the pipe arrays keep the static sources in a leading segment, so the movers' pipes are withdrawn exactly, the arrays cut at the seam, and only the movers linked again; `τ` is recomputed for everyone (the movers changed what stands in front of the static scene too), and `link_front` repairs the sharp weights only in the receptors the movers left or entered — where it sums the same sources in the same order as the full sweep, so a partial relink draws exactly what a full one would. The static scene — most of the pipes, and ~96 % of them beside the rock — stays linked. Pipes are fixed between relinks, which is why a source moving under a motionless camera needs one at all. |
| **Phase 3′ — arrive** | Every source offers its contribution, snapped to the pipes' fixed point. A source whose offer equals last tick's, through weights no relink touched, is **skipped whole** — a settled scene costs its entity count, not its pipe count (the lighting never quite stops twitching in the seventh digit; the snap is what makes "equal" happen, and keeps it exact: same offer, same weights, same pipes). The rest quantise each pipe and send `new − last` only if the integers differ — no epsilon. Parallel over **receptor bands** (`cut_bands`: whole rows, up to twice the worker threads, balanced by pipe count at the full relink; `band_off` records where each band's stretch of every entity's row-major run begins), each thread writing straight into its own band of receptors — no scratch image, no merge, no allocation. After a full relink every pipe sends and `pipe_last` is read as zero (`fresh`) rather than zero-filled. |

Two asymmetries in **Advance entities** are deliberate: `oscillation_phase`
advances for *every* solid, in frustum or not, so skin texture doesn't jump when
an entity re-enters view; and the AABB spans every solid including depth-culled
ones, so the atmosphere column and the relink trigger track the geometry rather
than the render cutoff.

## GPU Upload (`renderer.rs`)

Two `W×H` `Rgba16Float` textures — `dc` = (density, r, g, b) and `nd` =
(nx, ny, nz, depth) — uploaded **whole**, and only when the retina is `dirty`
(some delta arrived since the last upload). Color and depth are divided by
density here, on the CPU, and converted f32 → f16. At 320×180 each texture is
~450 KB — there is no sub-region bookkeeping because nothing is large enough to
need it.

## GPU Render (`shaders/retina.wgsl`)

A single fullscreen triangle; all the work is in the fragment shader. Present
mode is uncapped (`AutoNoVsync`). **No marching** — the receptor under the pixel
already holds the answer.

```
For each pixel:
  sample dc/nd at the pixel's receptor
  density < RETINA_ISO (0.3)? ──────── sky background
  else:
    color = dc.gba, normal = normalize(nd.xyz), depth = nd.w (pre-divided)
    if creature (green): procedural reptile skin
      - Voronoi scales at two frequencies + normal perturbation
      - fbm mottling, dorsal stripe, warm belly tint
    Lambert diffuse + ambient floor + rim + specular (fixed sun_dir)
  composite over sky gradient (zenith/horizon/ground + sun glow)
  velocity vignette → ACES tone map → gamma
```

The silhouette is a **threshold on arrived density**, so it stays sharp and
halo-free: sub-`iso` gaussian tails are never drawn. `RETINA_ISO` is duplicated
in `retina.rs` and `retina.wgsl` and must be kept in sync.

## Observer (`observer.rs`)

Free-fly camera with acceleration + drag. `c = 1 cell/tick = 30 cells/sec` at
30 ticks/sec; the observer is capped at `MAX_SPEED = 0.5c`. Effective FOV narrows
with speed (a linear approximation of relativistic aberration), and the shader
darkens screen edges at speed — fewer diffs reach you per tick the faster you go.

## Time-Lapse World Clock (`walker.rs`)

Two clocks are separated. A **sim step** is the 30 Hz wall-clock unit: one
graph hop of light, one arrival at the retina. A **world tick** is the physics
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
graph, wrapped in an **atmosphere** column of vacuum relays that carries it down
to the subject, and stands on a **40×40 dirt/grass floor** beside a rock. The sky
and sun glow are procedural background in the fragment shader.

### On the shadow

Two different transmittances exist, and only one of them is doing visible work.

The retina's per-entity `τ` (density between an entity and the *eye*) is what
makes the dino occlude what is behind it — that works, and it is what keys
`5`/`6` tune.

`edge_atten` is the other one, and in this scene it does **not** produce a floor
shadow. Only radiation edges (longer than `link_connect_dist`) carry it;
connection edges are exempt at `τ = 1`, and the sun→atmosphere→floor path is
made entirely of connection edges, because radiation links are never built to or
from vacuum relays. At the current constants `τ` is also effectively binary: of
~111k directed edges, only a few dozen are attenuated at all, and those go to
zero. Worse, Phase 2 renormalizes each emitter's pipe weights *after* applying
`τ`, so a blocked pipe's energy is redistributed to that emitter's other pipes
instead of being absorbed — the result is a relative deficit between one
entity's outgoing pipes, not light removed from the scene. Making the shadow
real needs absorption instead of renormalization, and node transmittance for
vacuum relays sitting inside solid density; both are on the
[roadmap](ROADMAP.md).

## Theoretical Basis

Based on tick-frame physics: time is discrete, space is the diff field, photons
are stationary (they *are* the field updates), and mass is what gives you energy
to fight the substrate stream. The observer can't reach `c` — at `c` you're a
photon and no rendering is possible. The reactive active-set is the engine's
literal causal cone: only what can reach the observer is computed.
