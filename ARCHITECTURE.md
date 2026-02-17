# Causal Cone Engine — Architecture

## Core Principle

There are no rays. There is no scene graph. There is only a field of accumulated 
diffs and an observer moving through it.

Entities deposit curvature at their position each tick. These deposits sit in the 
field until something reads them. The observer reads deposits by arriving at their 
location. What the observer sees is determined by what has been deposited within 
their causal cone — the region of space from which diffs have had time to reach them.

## Rendering Model

### What the screen shows

Each pixel represents: "What deposit would I encounter if I extended a line from 
the observer through this screen coordinate into the diff field?"

This sounds like raytracing but it's inverted:
- **Raytracing**: Cast ray → find surface → compute lighting
- **Causal cone**: For each deposit in cone → project onto screen → blend by depth

The difference: we iterate over deposits (forward), not over pixels (backward).
This is inherently O(deposits) not O(pixels × scene_complexity).

### Depth ordering

Deposits are bucketed by temporal lag from observer. Lag = how many ticks ago the 
deposit was made relative to the observer's current tick. Higher lag = further in 
the past = further in depth. Render back-to-front by bucket index. O(n), no sorting.

### Splat rendering

Each deposit is rendered as a Gaussian splat — a small oriented disk that blends 
smoothly with neighbors. When deposits are dense enough, splats overlap and form 
continuous surfaces. No mesh. No triangles. Just overlapping deposits.

Splat properties derived from deposit:
- **Position**: Where the entity deposited (3D world coordinate)
- **Size**: Proportional to deposit magnitude (heavier entity = bigger splat)
- **Color**: Derived from deposit properties (velocity, type, age)
- **Opacity**: Falls off with lag (older deposits fade)
- **Orientation**: Aligned to local field gradient (optional, v2)

### Occlusion

Last-write-wins. A deposit closer to the observer (lower lag) overwrites a deposit 
further away (higher lag) at the same screen pixel. This is painter's algorithm 
with bucket ordering. Alpha blending for partial transparency where deposits 
partially overlap.

### Lighting

No light sources. No shadow maps. No ambient occlusion calculation.

Deposit brightness = deposit magnitude × distance falloff. Regions with dense 
deposits are "bright" because many overlapping splats accumulate opacity. Regions 
with sparse deposits are "dark" because splats don't fill the gaps. Shadows are 
simply regions where occluding deposits block deposits behind them. This is 
automatic from the painter's algorithm — no shadow computation needed.

For v2: deposits from entities with high velocity leave "stretched" splats (motion 
blur) and deposits in high-curvature regions appear redshifted (gravitational 
effects on color).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TICK LOOP (CPU)                   │
│                                                     │
│  Entities ──deposit──→ DiffField (sparse HashMap)   │
│                                                     │
│  DiffField ──changed cells──→ GPU Upload Buffer     │
└─────────────────────┬───────────────────────────────┘
                      │ upload changed deposits
                      ▼
┌─────────────────────────────────────────────────────┐
│                 GPU PIPELINE (wgpu)                  │
│                                                     │
│  Storage Buffer: all active deposits                │
│          │                                          │
│          ▼                                          │
│  Compute Pass: causal cone culling                  │
│  - discard deposits outside observer cone           │
│  - output: visible deposit indices                  │
│          │                                          │
│          ▼                                          │
│  Render Pass: Gaussian splat                        │
│  - vertex shader: project deposit → screen quad     │
│  - fragment shader: Gaussian falloff + alpha blend  │
│  - draw order: by lag bucket (back to front)        │
│          │                                          │
│          ▼                                          │
│  Framebuffer → Swapchain → Display                  │
└─────────────────────────────────────────────────────┘
```

## Data Structures

### Deposit
```rust
struct Deposit {
    position: Vec3,      // world space
    magnitude: f32,      // deposit amount (1.0 = standard)
    lag: u32,            // ticks since deposit (temporal depth)
    color: [f32; 3],     // RGB derived from entity properties
    velocity: Vec3,      // entity velocity at deposit time (for motion blur)
}
```

### DiffField
Sparse HashMap<IVec3, Vec<Deposit>> — grid cell → deposits at that cell.
Only cells with deposits exist. Empty space costs nothing.

### Observer
```rust
struct Observer {
    position: Vec3,
    orientation: Quat,
    causal_radius: f32,  // how far diffs have reached (grows at c per tick)
    tick: u64,           // current observer tick
}
```

## Phases

### v0.1 — Proof of concept
- [x] Window + wgpu setup
- [ ] 10,000 entities depositing randomly in 3D
- [ ] Point rendering (1 pixel per deposit, no splats yet)
- [ ] Camera with WASD + mouse
- [ ] Temporal lag as alpha (older = more transparent)
- [ ] Back-to-front bucket ordering
- [ ] FPS counter
- **Goal**: See moving points with depth from temporal lag. Validate O(n) rendering.

### v0.2 — Gaussian splats
- [ ] Replace points with screen-space Gaussian quads
- [ ] Splat size from deposit magnitude
- [ ] Alpha blending back-to-front
- [ ] Surfaces emerge from dense deposit regions
- **Goal**: Dense point clouds look like surfaces.

### v0.3 — Causal cone culling
- [ ] Compute shader for cone intersection test
- [ ] Only render deposits within observer's cone
- [ ] Cone grows at 1 cell/tick (c)
- [ ] Entities beyond horizon cost zero
- **Goal**: Automatic LOD from causal structure.

### v0.4 — Temporal effects
- [ ] Sliding window from Experiment 49
- [ ] Trails: render last N ticks of each entity
- [ ] Motion blur: velocity-weighted splat stretching
- [ ] Holographic horizon: compressed far-field overlay
- **Goal**: Port Experiment 49's temporal effects to GPU.

### v0.5 — Real scenes
- [ ] Load point cloud data (PLY/LAS files)
- [ ] Convert point cloud → deposits
- [ ] Camera flythrough of real scenes
- [ ] Benchmark against conventional renderers
- **Goal**: Render real-world data competitively.

### v1.0 — Product
- [ ] Scene graph for entity management
- [ ] Physics integration (optional — entities can move)
- [ ] Asset pipeline (mesh → point cloud → deposits)
- [ ] Multi-window support
- [ ] API for external applications
- **Goal**: Usable engine that others can build on.
