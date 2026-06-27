# Causal Cone Engine

> A personal pet project — an experimental renderer, built for fun and exploration.

A rendering engine where the observer swims through a persistent diff field.

There are no rays. There are no meshes. There are no lights.
Entities deposit diffs into a 3D field and propagate them along a graph of
connections. The observer moves through the field, reading what's there.
What you see is what has already arrived.

## Build & Run

```bash
cargo run --release
```

First build will take a while (downloading + compiling wgpu). Subsequent builds are fast.

`--release` is important — the CPU-side field simulation is heavy and needs optimization.

## Controls

| Key | Action |
|-----|--------|
| WASD | Move horizontally |
| Space | Move up |
| Shift | Move down |
| Mouse | Look around (click window to capture) |
| Escape | Release mouse / quit |
| T | Toggle trie-depth visualization |
| I | Dump trie / entity info to the log |
| `[` / `]` | Decrease / increase render-depth cutoff (progressive rendering by trie depth) |

## What You're Seeing

The screen is produced by sampling the diff field from the observer's position.
For each pixel, the fragment shader marches a direction into a 3D texture and
bisects to the iso-surface — the first deposit dense enough to count as solid.

- **Bright regions**: Dense deposits from entities
- **Dark regions**: Empty field (vacuum)
- **Color**: Derived from entity properties and accumulated light
- **Depth**: March distance before hitting a deposit (lag)
- **Vignette at speed**: Moving fast narrows your effective field of view

## Architecture

```
CPU (30 ticks/sec):
  Entities deposit → light propagates along connection graph →
  field updates → dirty slabs uploaded (f32 → f16) to the GPU 3D texture

GPU (uncapped):
  For each pixel → march into the 3D texture → bisect to iso-surface → shade
```

The field is a **512³** grid (~134M cells). Uploads are restricted to dirty
slabs within the active geometry's AABB and converted to `Rgba16Float`, so only
what actually changed crosses the bus each tick.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the fuller design.

## Demo Scene

A dinosaur whose body is a *skeleton* of overlapping metaball entities — body,
belly, tail, neck, head, jaw, mouth, yellow eyes, legs and feet, plus midpoints
at the joints — each depositing a wide anisotropic gaussian blob that merges into
seamless geometry, wrapped in a procedural reptile-skin texture. A separate
lightweight receptor shell on the surface catches light and re-emits it as color.
It stands on a 40×40 dirt/grass floor beside a rock, lit by a sun disc, with an
atmosphere scatter column relaying light from the sun into the scene. The sky and
sun glow are drawn procedurally by the fragment shader.

## Theoretical Basis

Based on tick-frame physics: time is discrete, space is the diff field,
photons are stationary (they ARE the field updates), and mass is what
gives you energy to fight the substrate stream.

The observer cannot move at c. At c you're a photon — no rendering
possible. Faster movement = narrower field of view = fewer diffs
reaching you per tick. This is relativistic aberration from pure geometry.
