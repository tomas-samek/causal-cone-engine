# Receptor Retina — Design

> Status: approved in brainstorming, 2026-08-26. Supersedes the grid retina
> described in `ARCHITECTURE.md`. Parks the desaturation/attenuation plan
> (`docs/plans/2026-07-03-desaturation-occlusion-plan.md`) at its Task 3
> checkpoint; Tasks 1–2 (stats dump, tuning keys) and the transmittance
> integral of Task 5 are reused here.

## Goal

The observer stops sampling a cached volume and becomes the **last node of the
information pipelines**. The image is a persistent array of **receptors** on the
image plane; entities deliver light to receptors along pipes exactly as they
deliver it to each other; **only deltas travel**; occlusion is **pipe
transmittance** and nothing else. The dense `512³` `FieldCell` grid, its decay
and deposit passes, the slab upload, the ray march and `ray_blocked` are all
removed.

## Non-goals

- Per-receptor depth tests / z-buffer of any kind.
- Per-pipe transmittance (Approach 2) and hierarchical receptors (Approach 3).
  The transmittance function takes a segment and returns a factor so Approach 2
  is a later call-site change.
- Atmosphere haze in the image (was a grid scatter effect). Sky and sun glow
  stay procedural in the shader.
- Full-resolution receptors. Resolution is a constant; correctness must not
  depend on it.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| What is replaced | The `FieldCell` grid (the retina), not the entity graph |
| Where the observer sits | Observer **is a receptor array** — a node set in the graph |
| Receptor granularity | Coarse array (`RETINA_W × RETINA_H`, initial 320×180), GPU upsamples + cosmetics |
| Arrival model | Deltas only: receptors are state, pipes carry `new − last_sent` |
| Occlusion | Pipe transmittance τ, one per entity toward the eye (Approach 1) |
| Coexistence | New module `src/retina.rs` + `shaders/retina.wgsl`; grid path stays compilable but unwired as reference until the visual check passes, then deleted |

## Components

### `src/retina.rs` (new)

```rust
pub const RETINA_W: u32 = 320;
pub const RETINA_H: u32 = 180;
pub const RETINA_ISO: f32 = 0.3;        // display threshold, same value as the grid iso
pub const DELTA_EPS: f32 = 1e-4;        // debounce: |δ| below this is not sent
pub const RELINK_SHIFT: f32 = 0.5;      // receptors; max projected shift that triggers a relink
pub const ATTEN_END_MARGIN: f32 = 1.5;  // cells skipped at both segment ends
pub const ATTEN_SAMPLES_PER_CELL: f32 = 1.0;

#[derive(Clone, Copy, Default)]
pub struct Receptor {
    pub density: f32,        // Σ arrivals
    pub color: [f32; 3],     // Σ density·color
    pub normal: glam::Vec3,  // Σ density·surface_normal
    pub depth: f32,          // Σ density·distance (shader cosmetics only — never occlusion)
}

pub struct Retina {
    pub width: u32,
    pub height: u32,
    pub receptors: Vec<Receptor>,
    // pipes entity → receptor, structure-of-arrays, per-entity slices
    pipe_start: Vec<u32>,
    pipe_count: Vec<u32>,
    pipe_receptor: Vec<u32>,
    pipe_weight: Vec<f32>,        // feathered gaussian footprint weight
    pipe_last: Vec<PipeState>,    // last sent — the delta baseline
    entity_trans: Vec<f32>,       // τ per entity toward the eye
    last_view_proj: glam::Mat4,
    pub stats: RetinaStats,       // pipes total / sent, receptors ≥ iso, mean τ, relinks
}

#[derive(Clone, Copy, Default)]
struct PipeState { density: f32, color: [f32; 3], normal: glam::Vec3, depth: f32 }
```

Receptors **never decay and are never cleared**. Their value is by construction
the exact sum of the current `pipe_last` values, maintained incrementally.

### `src/field.rs` (existing, reduced)

Keeps: entities, edge SoA, cross-link refresh, active set, Phase 1 deliver,
consumption, Phase 2 push, walker motion, `tune_density`/`tune_color`,
`scale_tune`, `cutoff_window`, `dump_field_stats` (retargeted to retina stats).

Gains: `edge_atten: Vec<f32>` on radiation edges longer than
`link_connect_dist`, computed with the same transmittance function (this is the
dino's floor shadow — the B2′ mechanism, unchanged in intent).

Loses (after cut-over): `FieldCell`, `cells`, Phase 0 decay, Phase 3 grid
deposit, dirty slabs, AABB tracking used only by the grid, `ray_blocked`,
`as_bytes`.

### `src/renderer.rs` (existing, reduced)

Replaces the 3D texture and slab upload with two `RETINA_W × RETINA_H`
`Rgba16Float` textures: `(density, r, g, b)` and `(nx, ny, nz, depth)`.
Uploaded whole, only on ticks where at least one δ was sent.

### `shaders/retina.wgsl` (new)

Per pixel: bilinear-sample both textures at the pixel UV.
`density < RETINA_ISO` → existing sky/sun-glow background.
Else `color /= density`, `normal = normalize(normal_sum)`, then the existing
shading block verbatim: Lambert + ambient floor + rim + specular against
`sun_dir`, reptile skin (Voronoi scales, fbm, dorsal stripe, belly tint) for
creature color; velocity vignette → ACES → gamma unchanged.

`field_sample.wgsl` remains as reference until cut-over, then is deleted.

## Linking (projection)

Rebuilds all pipes. Runs when:

- the observer moved/rotated so that the maximum projected shift of the scene
  AABB's 8 corners between `last_view_proj` and the current `view_proj`
  is ≥ `RELINK_SHIFT` receptors, or
- the walker cross-link refresh fired this tick (entities moved ≥ 1 cell), or
- the retina was (re)created (resolution change).

Procedure:

1. For every existing pipe (relink is a full rebuild), subtract `pipe_last`
   from its receptor. Receptors are now exactly zero (assert in debug builds).
2. For each drawable entity — `!is_heat && !is_vacuum` and inside the
   frustum per `compute_active_set` — project `position` → receptor-space
   center `(u, v)` and eye distance `d`.
3. Project `deposit_radii` to an image-plane ellipse: per axis
   `radius / d × focal`, oriented by the entity's `surface_normal` frame the
   way the grid deposit orients its anisotropic blob. Entities with zero
   `deposit_radii` use a single receptor. Clamp the ellipse to ≥ 1 receptor
   so distant entities never vanish.
4. Each receptor whose center lies inside the ellipse gets a pipe with
   `pipe_weight = exp(−e) · cutoff_window(e)` where `e` is the squared
   normalized offset — the same feathered cutoff as the grid deposit
   (commit `100f558`). `pipe_last` starts at zero.
5. Recompute `entity_trans` (below).
6. Store `view_proj` as `last_view_proj`; `stats.relinks += 1`.

Vacuum entities (sun disc, atmosphere column) get no pipes.

## Transmittance

```
τ_i = exp(−atten_k · ∫ ρ_{≠i}(s) ds)
```

over the segment from `position_i` toward the eye, skipping `ATTEN_END_MARGIN`
cells at both ends, sampled at `ATTEN_SAMPLES_PER_CELL`. `ρ_{≠i}(s)` is the
analytic sum of every other drawable, non-vacuum entity's
`magnitude · gaussian(s − pos_j, radii_j)` within reach, found through a
spatial hash whose cell size is the maximum deposit radius in the scene.

`atten_k` is the existing tunable (`ATTEN_K_DEFAULT = 0.5`, keys `5`/`6`,
clamped to [1/1024, 1024]). One function serves both uses:

```rust
pub fn segment_transmittance(field: &DiffField, hash: &SpatialHash,
                             from: Vec3, to: Vec3, skip: Option<usize>, k: f32) -> f32
```

— called per entity toward the eye for the retina, and per long radiation edge
for the floor shadow.

Recomputed on relink only; entities move on the walker cadence, so per-tick
recomputation would be waste.

## Per-tick pipeline

| Phase | What happens |
|---|---|
| Cross-link refresh | unchanged |
| Active set | unchanged |
| Phase 1 deliver, consumption, Phase 2 push | unchanged — light still hops the graph to the shell |
| Walker motion | rigid translate + pacing, moved out of the old Phase 3, behaviour unchanged |
| **Relink** (conditional, see above) | drop → subtract; rebuild pipes; recompute τ |
| **Phase 3′ arrive** | per drawable entity: `contrib = (magnitude·tune_density, emission·tune_color, surface_normal, d) · τ_i`; per pipe `new = contrib · weight`, `δ = new − pipe_last`; if `max|δ| > DELTA_EPS` → receptor += δ, `pipe_last = new`, `stats.pipes_sent += 1` |
| Upload | both textures, whole, only if `stats.pipes_sent > 0` this tick |

Phase 3′ runs under rayon over entities. Because many entities feed one
receptor there is no disjoint ownership; each thread accumulates into a scratch
`Vec<Receptor>` and the scratches are reduced into `receptors` afterwards. If
profiling shows the reduce dominates, switch to per-receptor atomics on packed
f32 — not in scope until measured.

## Controls

| Key | Action |
|---|---|
| `H` | retina stats dump: receptors ≥ iso, pipes total / sent this tick, mean τ, relinks so far, resolution |
| `1` / `2` | `tune_density` ÷2 / ×2 (existing) |
| `3` / `4` | `tune_color` ÷2 / ×2 (existing) |
| `5` / `6` | `atten_k` ÷2 / ×2 (from the parked plan) |
| `7` / `8` | receptor resolution ÷2 / ×2 (both axes), rebuild retina, force relink |

Every tuning keypress logs all current tuning values. Existing keys `T`, `I`,
`[`, `]`, `-`, `=` are untouched.

## Testing

Unit tests in `retina.rs`, no GPU:

- **projection** — one entity at a known position/radii with a known
  `view_proj` → expected receptor set; the center receptor's weight is 1.0;
  weights fall off symmetrically.
- **delta exactness** — after N ticks of randomized contribution changes,
  each receptor equals the direct sum of current `contrib · weight` (1e-5).
- **relink exactness** — after a relink that drops and adds pipes, receptors
  equal the direct sum over the new pipe set; the intermediate state after the
  drop step is all-zero.
- **transmittance** — occluder centered on the segment → τ matches
  `exp(−k·∫gaussian)` numerically; occluder well off the segment → τ ≈ 1;
  self is excluded; end margins exclude entities at the endpoints.
- **debounce** — a settled scene sends 0 pipes on the next tick.
- **resolution change** — halving/doubling and relinking leaves the
  per-receptor density integral over the image within a tolerance.

Ignored integration test (`--ignored`, release): spawn the demo scene, run 60
ticks with the default observer, assert: receptors under the dino silhouette
≥ iso; floor receptors beneath the dino darker than open-floor receptors
(shadow present); no NaN/inf anywhere; one extra tick with the dino frozen
sends 0 pipes.

Human verification (cannot be done headlessly): dino silhouette crisp and
1×-radii tight; jaw opening reads; shadow follows the pacing dino; `5`/`6`
change shadow depth on the next relink; `7`/`8` show whether 320×180 is
enough.

## Cut-over sequence

1. Branch `retina` from `main`; cherry-pick `19b8e60` (stats dump) and
   `e67127e` (tuning keys) from `desaturation-attenuation`.
2. `retina.rs` with `segment_transmittance`, linking, Phase 3′, unit tests.
3. `edge_atten` on long radiation edges in `field.rs` using
   `segment_transmittance`.
4. `retina.wgsl`; renderer switched to the two 2D textures.
5. `main.rs` keys; `tick` wired to the retina path; grid path unreferenced.
6. Human visual check. Stop here if the silhouette or shadow is wrong.
7. Delete the grid path: `FieldCell`, decay, grid deposit, slab upload,
   `ray_blocked`, `field_sample.wgsl`, related tests.
8. Docs: `ARCHITECTURE.md` core principle and pipeline table rewritten around
   the receptor retina; `README.md` controls table; `ROADMAP.md` notes
   Approach 2/3 as follow-ups.

## Risks

- **Soft silhouettes** where the body overlaps itself: τ alone attenuates,
  it does not sort. Remedy is `RETINA_ISO` and `atten_k`, then resolution —
  never a depth test.
- **Reduce cost** in Phase 3′ at higher resolutions; measured before optimized.
- **Relink churn** when the observer flies: every relink is a full rebuild
  including τ. If it shows in the frame time, split into observer-only relinks
  (reproject, keep τ) and entity relinks (recompute τ).
