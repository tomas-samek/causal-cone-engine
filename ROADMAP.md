# Causal Cone Engine — Roadmap

## v0.2: Dino looks good ✅ (current)
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

## v0.3: Hierarchical entities (macro-patterns)
- Parent-child entity relationships
- Macro-patterns emerging from grouped sub-entities
- Entity tree structure replacing flat storage

## v0.4: Observer-dependent resolution
- Field resolution varies with observer distance
- Closer observation = finer detail, far = coarser
- Causal cone driven LOD

## v0.5: Dynamic connector density based on distance
- Graph topology becomes dynamic
- Denser connections near observer, sparser far away
- Runtime edge management

## v0.6: Multiple objects interacting
- Multiple independent entity groups
- Inter-object interaction rules
- Leverages hierarchical entities from v0.3

## v0.7: Sound propagation
- Field carries additional signal types beyond light
- Acoustic wave simulation through entity graph

## v0.8: Water/reflection
- Reflective surface simulation
- Extends specular material properties
- Dynamic reflection via field re-emission

## v0.9: Scene scale (landscape, multiple creatures)
- Larger worlds beyond 128³
- Hierarchical or streaming field
- Multiple animated creatures

## v1.0: Real Engine alpha release
- Stable API and scene format
- Documentation and examples
- Performance targets for real-time use
