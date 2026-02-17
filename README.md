# Causal Cone Engine v0.1

A rendering engine where the observer swims through a persistent diff field.

There are no rays. There are no meshes. There are no lights.
Entities deposit diffs into a 3D field. The field spreads at c.
The observer moves through the field, reading what's there.
What you see is what has already arrived.

## Build & Run

```bash
cd W:\workspace\causal-cone-engine
cargo run --release
```

First build will take a while (downloading + compiling wgpu). Subsequent builds are fast.

`--release` is important — the CPU-side field spreading is O(n³) and needs optimization.

## Controls

| Key | Action |
|-----|--------|
| WASD | Move horizontally |
| Space | Move up |
| Shift | Move down |
| Mouse | Look around (click window to capture) |
| Escape | Release mouse / quit |

## What You're Seeing

The screen is a slice through the diff field at the observer's position.
Each pixel answers: "what deposit exists along this direction?"

- **Bright regions**: Dense deposits from entities
- **Dark regions**: Empty field (vacuum)
- **Color**: Derived from entity properties
- **Depth**: March distance before hitting a deposit (lag)
- **Vignette at speed**: Moving fast narrows your effective field of view

## Architecture

```
CPU (30 ticks/sec):
  Entities deposit → Field updates → Upload 3D texture to GPU

GPU (vsync):
  For each pixel → march ray into 3D texture → first hit = color
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design.

## Demo Scene (v0.1)

- A flat wall (20×20 entities, warm color)
- A sphere (500 entities, blue, Fibonacci distributed)
- A moving red point (orbiting)
- A checkerboard floor (40×40 entities)
- Total: ~2500 entities in a 128³ field

## Theoretical Basis

Based on tick-frame physics: time is discrete, space is the diff field,
photons are stationary (they ARE the field updates), and mass is what
gives you energy to fight the substrate stream.

The observer cannot move at c. At c you're a photon — no rendering
possible. Faster movement = narrower field of view = fewer diffs
reaching you per tick. This is relativistic aberration from pure geometry.

See: `tick-frame-space/docs/theory/ch006_rendering_theory.md`
