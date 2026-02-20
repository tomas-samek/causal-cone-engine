# Causal Cone Engine — Roadmap

## v0.2: Dino looks good ✅
- Diff-field rendering with 128³ grid
- Entity graph with connection + radiation edges
- 3-phase light propagation (deliver → push → deposit)
- Material system (pass_through, reemit, scatter, specular, heat, vacuum)
- Sun disc, vacuum relay network, atmospheric scattering
- Automatic shadows from light blocking
- Performance: dirty slab uploads, SoA edges, AABB culling
- wGPU ray-marching with adaptive step sizes
- Radiation links: direct surface-to-surface edges (replaced bounce relay vacuum layer)
- Physically correct grid deposit: only absorbed light is visible (1 - pass_through)
- Inverted atmosphere gradient: dense near floor, thin near sun
- Tuned material properties: low scatter atmosphere, moderate re-emission

## v0.3: Hierarchical entities (macro-patterns) ✅
- Entity group system: 18 GROUP_* constants tagging every entity (body parts, sun, floor, vacuum)
- Edge gamma infrastructure: per-edge weight multiplier for gamma-weighted light distribution
- Tail wag animation: deposit-position offset via sine wave with traveling phase along tail
- Faster grid decay (0.92 → 0.85) for cleaner animation trails

## v0.4: Reactive pipeline & scale ✅ (supersedes planned "Observer-dependent resolution")
- Reactive pipeline: frustum-based active set culling (Gribb-Hartmann). Only entity chains feeding into the observer's view get processed (~30-50% Phase 1-3 reduction)
- Reverse edge index for backward graph traversal
- Debounce: entities with stable incoming skip Phase 2 push (threshold = edge_count)
- AABB-restricted Phase 0 decay: only touches cells near geometry (~15x reduction at 512³)
- AABB-restricted GPU upload: f32→f16 conversion limited to geometry sub-rectangle per slab
- Precomputed edge directions: eliminate ~100K normalize/sqrt per tick
- Rgba16Float texture format: halves GPU VRAM vs Rgba32Float
- Spatial entity sort for cache-friendly Phase 3 deposits
- Field size 128³ → 512³ (134M cells, ~2GB CPU / ~1GB GPU)
- Jaw animation: cyclic open/close on ~4 sec cycle, pivot at back of jaw

## v0.5: Visual quality & metaball geometry ✅
- Metaball body: replaced 16 overlapping ellipsoids with single metaball field pass. Smooth seamless joints — neck/body/legs/tail blend naturally. Kernel: weight × max(0, 1−r²), threshold 1.0. Color/material interpolated, group from strongest contributor.
- Dynamic connector density: per-entity distance factor from observer scales edge gammas in Phase 2. Close = full weight, distant = 0.1× (topology fixed, signal strength varies).
- ACES filmic tone mapping (replaced Reinhard) — better contrast and color preservation
- Gradient normals: 6-sample central-difference density gradient for Lambert diffuse + rim light
- Sky gradient: blue zenith → warm horizon → dark ground, with sun glow hotspot
- Trilinear texture filtering: GPU interpolates between voxels, filling surface gaps
- 3×3×3 tent-weight deposit: wider splat footprint with smooth falloff (replaced 2×2×2 trilinear)
- Subsurface darkening: entities adjacent to a heat interior get 0.2× color, 0.15× magnitude
- Opacity tuning: density × 0.3 (was 0.1) — body surfaces appear solid, atmosphere stays translucent
- Color normalization fix: removed gray fallback, always divide by max(density, 0.05)

## v0.6: Shadow tuning & atmosphere ✅
- Atmospheric column: concentrated vacuum relay network around dino AABB (replaces broad grid, ~60% fewer vacuum entities)
- Radial density profile: scatter/magnitude peaked at dino center, Gaussian-like falloff
- Per-tick atmospheric modulation: scatter/magnitude follow live AABB center
- Shader ambient floor: 0.25 → 0.10 for deeper ground shadows
- Observer start position: z=380 → z=310 for closer view

## v0.7: Multiple objects interacting
- Multiple independent entity groups
- Inter-object interaction rules
- Leverages hierarchical entities from v0.3

## v0.8: Sound propagation
- Field carries additional signal types beyond light
- Acoustic wave simulation through entity graph

## v0.9: Water/reflection
- Reflective surface simulation
- Extends specular material properties
- Dynamic reflection via field re-emission

## v0.10: Scene scale (landscape, multiple creatures)
- Larger worlds beyond 512³
- Hierarchical or streaming field
- Multiple animated creatures

## v1.0: Real Engine alpha release
- Stable API and scene format
- Documentation and examples
- Performance targets for real-time use

---

## Future improvements (deferred)
- **Observer-dependent resolution (LOD)**: Multi-resolution field varying with observer distance. Closer = finer grid, far = coarser. Causal cone driven LOD. May be needed if scene complexity outgrows 512³.
- **Distance-based debounce**: Entities closer to observer use stricter debounce thresholds (update more often), distant entities debounce more aggressively.
