# Causal Cone Engine

> A personal pet project — an experimental renderer, built for fun and exploration.

A rendering engine where light is delivered, not gathered.

There are no rays. There are no meshes. There are no lights.
Entities push light along a graph of connections, one hop per tick, and the
observer is a **receptor array** on the image plane that those same pipes feed.
What you see is what has already arrived at it.

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
| `-` / `=` | Halve / double the time-lapse factor (×1 = literal real-time 1e-6 c) |
| H | Dump retina + floor-probe stats to the log |
| `1` / `2` | Halve / double density tuning |
| `3` / `4` | Halve / double color tuning |
| `5` / `6` | Halve / double occlusion strength (`atten_k`) |
| `7` / `8` | Halve / double receptor resolution |

## What You're Seeing

The screen **is** the receptor array. Each receptor holds a running sum of what
entities delivered to it along their pipes, attenuated by the density sitting
between the entity and the eye. Nothing is sampled, marched, or looked up: the
image is already there when the frame starts, and the fragment shader only
thresholds it, shades it with the normal that arrived, and composites it over a
procedural sky.

- **Bright regions**: receptors many entities reached
- **Dark regions**: receptors nothing reached, or whose pipes are dimmed
- **Color**: entity color and re-emitted light, density-weighted per receptor
- **Depth**: density-weighted eye distance of what arrived
- **Vignette at speed**: moving fast narrows your effective field of view

## Architecture

```
CPU (30 ticks/sec):
  Entities push light along the connection graph → each drawable entity
  sends its change since last tick down its pipes → receptors accumulate

GPU (uncapped):
  Upload two W×H receptor textures (f32 → f16) → threshold, shade, composite
```

Pipes carry **deltas**: an entity whose contribution hasn't changed sends
nothing, so a still scene costs almost no traffic. They are relinked only when
the view or the geometry has shifted by half a receptor.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the fuller design.

## Demo Scene

A dinosaur whose body is a *skeleton* of overlapping metaball entities — body,
belly, tail, neck, head, jaw, mouth, yellow eyes, legs and feet, plus midpoints
at the joints — each carrying a wide anisotropic gaussian kernel, projected onto
the retina so the overlapping blobs merge into seamless geometry, wrapped in a
procedural reptile-skin texture. A separate
lightweight receptor shell on the surface catches light and re-emits it as color.
It paces slowly back and forth on a 40×40 dirt/grass floor beside a rock —
moving at a true 1e-6 c, rendered visible through a ×100,000 time-lapse
world clock — lit by a sun disc, with an
atmosphere scatter column relaying light from the sun into the scene. The sky and
sun glow are drawn procedurally by the fragment shader.

## Theoretical Basis

Based on tick-frame physics: time is discrete, space is the diff field,
photons are stationary (they ARE the field updates), and mass is what
gives you energy to fight the substrate stream.

The observer cannot move at c. At c you're a photon — no rendering
possible. Faster movement = narrower field of view = fewer diffs
reaching you per tick. This is relativistic aberration from pure geometry.
