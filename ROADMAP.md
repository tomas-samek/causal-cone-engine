# Causal Cone Engine — Roadmap

> A personal pet project — this roadmap is a changelog of what's been built and a
> loose sketch of where it might go, not a committed plan. ✅ marks done.

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
- Observer start position: z=380 → z=310 for a closer view
- Gamma surface model: replaced BFS hollow shell (~27K entities) with ~27 skeleton entities using anisotropic gaussian deposits. Each skeleton point deposits an ellipsoidal blob matching its metaball radii — overlapping gaussians merge into continuous density field. ~200× fewer cell writes per tick.
- Receptor shell: lightweight BFS surface entities (spacing 1.0) catch atmosphere light via radiation links. Skeleton handles density, receptors handle lighting — two systems writing to the same grid.
- Midpoint entities: interpolated skeleton points at joints (body↔neck, neck↔head, body↔tail, etc.) for smoother blending between body parts.
- Cleanup: removed BFS builder, grid_pos/is_subsurface fields, metaball field evaluator simplified to surface detection only.

## v0.7: Consumption entities, trie rendering & crisp surfaces ✅
- Consumption-transformation entities (`consumption.rs`): each body entity's incoming deposit is quantized to a `DepositToken` (4-bit density + RGB). A per-entity `Spectrum` crystallizes from the most frequent tokens covering 50% of observations (`TARGET_COVERAGE`).
- Consumption trie via `cascade_process`: recognized tokens are consumed, rejects cascade to a child, persistent rejects seed new child states one level deeper (`MAX_TRIE_DEPTH = 20`). Trie topology wired by BFS over the body graph.
- Consumption-modulated deposits: the consumed fraction blends each deposit toward the entity's own color (rest passes through).
- Trie-depth diagnostics: `T` toggles depth-as-color visualization; `I` dumps per-entity trie/spectrum info to the log.
- Progressive rendering by trie depth: `[` / `]` adjust `render_depth_cutoff`; entities deeper than the cutoff skip their grid deposit.
- Iso-surface bisection march: replaced fog/alpha compositing with first-crossing detection at `iso = 0.3` plus 12-step bisection — crisp, halo-free silhouettes (sub-`iso` Gaussian tails are never drawn).
- Procedural reptile skin: two-frequency Voronoi scales with normal perturbation, fbm mottling, dorsal stripe, warm belly tint, and a waxy specular sheen.

## v0.8: Receptor retina (the grid is gone) ✅
- Receptor retina (`retina.rs`): the observer is a persistent `W×H` array of receptors on the image plane, each the running sum of what entities delivered along entity→receptor pipes. Sums, never averages — the renderer normalizes on upload. (Since 2026-09-21 every pipe has two weights: density through the broad footprint, colour/normal/depth/skin through a sharpened, front-to-back-dimmed one normalized by its own `sharp` sum — near geometry stopped being a blur of every tail that reached it. The sharp power is per source, as much as its lattice spacing can carry: the rock gets 8, the floor 2. The dimming is by *delivered density* in front, read front to back — a receptor shows the sources that make up its first iso, and anything behind a full iso gets zero.)
- Delta pipes: each pipe remembers what it last sent and transmits only `new − last`. A settled scene sends almost nothing; the sum stays exactly reversible, so a relink can subtract every pipe and land on zero. (Since 2026-09-21 pipes are fixed-point integers — i32 Q16 pipes into i64 receptors — so "exactly" is bit for bit and the old `DELTA_EPS` debounce is gone.)
- Relink on movement only: pipes are rebuilt when the scene AABB's projected corners shift ≥ `RELINK_SHIFT` (0.1 receptor), a source's projected centre does, the cross-links refresh, or a tuning key fires — not per frame. (2026-09-22: the relink is **partial** when the camera holds still and only dynamic sources moved — the static scene's pipes stay linked; `arrive` skips any source whose offer and weights are unchanged, so a settled scene costs its entity count, not its pipe count.)
- Fewer pipes where they were pure overlap (2026-09-22): a source all of whose receptors already hold `HIDDEN_AT` isos of density in front of it is *hidden* and linked with no pipes (a near rock's back and underside); a lattice member whose projected σ exceeds `SHRINK_SIGMA` draws with a kernel shrunk toward it — no smaller than 0.6× its lattice spacing — delivering 1/shrink² more, so the summed surface is unchanged while pipes fall with the square (1.5 M → 0.6 M beside the rock).
- Parallel by receptor band: `link_front` and `arrive` each run one thread per band of whole rows, balanced by pipe count, writing straight into their own receptors — no scratch images, no merge.
- Occlusion as transmittance: `segment_transmittance` integrates `exp(−k·∫max(0, ρ−threshold))` through the gaussian density along a segment. Used both per entity toward the eye and per radiation edge (`edge_atten`). Replaced the old binary line-of-sight test. (2026-09-22: toward the eye the segment starts at the kernel's *near surface*, not its centre — a metaball's τ was flipping to zero whenever the one line from its centre crossed a sibling ball, which drew a dark crescent on the dino from some angles.)
- Display shader (`shaders/retina.wgsl`): threshold at `RETINA_ISO`, shade with the arrived normal, composite over the procedural sky — no ray march, no 3D texture.
- Live tuning keys: `H` stats dump, `1`/`2` density, `3`/`4` color, `5`/`6` occlusion strength, `7`/`8` receptor resolution.
- Deleted: the `512³` `FieldCell` grid (~2 GB), Phase 0 decay, the Phase 3 voxel deposit, dirty-slab uploads, and `shaders/field_sample.wgsl`.

## v0.9: Multiple objects interacting
- Multiple independent entity groups
- Inter-object interaction rules
- Leverages hierarchical entities from v0.3

## Extinction test: stored light (experimental — on its own branch; the rules below are a first cut and may be a dead end)
Today every pipeline is instantaneous in steady state: sun → atmosphere → dino → retina, and an entity's re-emission (`reemit_r/g/b`) is a fixed fraction of *this tick's* incoming — switch the source off and the dino goes dark the next tick, apart from the one-hop-per-tick delay of the graph itself. Test and then model the alternative: each node is an absorber **and** an emitter with a **buffer** — what it absorbs charges the buffer, and the buffer emits a slow, decaying deposit of its own, so a surface (the skin, or the skeleton layers under it) keeps glowing faintly after its light source is switched off.
- The test first: turn the sun off in a settled scene and record, per tick, what still arrives at the retina and from which entities. Today the answer should be "nothing after the graph drains" — that number is the baseline the buffer model has to beat.
- **Target state — the buffer *is* the consumption state.** Not a second store beside the trie: the entity's inner state (`ConsumptionState`, its spectrum and trie) is what holds charge, and keeping it running costs light. Each tick, **per incoming packet** (one token per incoming edge — not the summed `incoming`), the packet is **XOR**ed with what the state needs for its upkeep: the bits that match are consumed — they charge the state and pay for its update — and what XOR leaves is that packet's remainder. The remainders of all packets are then merged with **OR**, and that is what the entity **emits**, i.e. its colour. Colour is not a stored `[f32; 3]` any more but the difference between what arrived and what the state ate (`entity.color` survives only as the diet's complement); a well-fed entity shows the light it does not need, a starved one goes dark as its state spends its charge and stops taking. `reemit`, the fixed fraction of this tick's incoming, and the `ln(consumed)` mass boost both dissolve into this: emission is the leftover, density is the state's charge.
- Afterglow follows: with the source off, the state keeps drawing on its charge for a while and keeps emitting the leftover of what it still has, decaying at a rate set by how much upkeep costs — a material property (skin cheap and quick to fade, skeleton slow, rock/floor near zero). The retina needs nothing new — it already sums whatever the sources offer, and a fading source is just a source whose offer changes.
- The width caveat: in the real world a packet is 2^hundreds of "bits", not a 4-bit token, and at that width XOR and OR stop being logic and become statistics — OR over n packets of bit density p leaves 1−(1−p)ⁿ set, XOR against a fixed mask is a per-bit flip, i.e. linear. Our small ints are the regime where discreteness bites hardest (boundary flips, 1/15 steps). We are not building the universe bit by bit, so the branch implements the **mean-field limit** — the same rules on per-bit probabilities in floats, which cannot flicker, need no hysteresis, stay float-shaped like the rest of the field (debounce and retina snap unchanged), and are what the wide version converges to. Per channel, with `p` a packet's density and `m` the state's upkeep mask (the share of that channel it needs): consumed = `p·m`, XOR remainder = `p + m − 2pm`, OR-merge of n remainders = `1 − Π(1 − rᵢ)`. A bit-exact token version is at most a reference to compare against, not the plan. With `m` a fraction per channel the upkeep mask is itself a colour — the entity's diet — so today's material definitions map onto it directly: diet = `1 − color`, and the leftover under white light *is* the colour, rather than its complement.
- What to watch: the settled-scene invariant (an entity in equilibrium eats and emits the same every tick — the frozen-scene test must still land near-silent, which it does not today for the `ln(consumed)` drift), energy bookkeeping (emitted + consumed ≤ arrived + charge spent; a state must not amplify), the token quantisation (4-bit density + RGB — XOR and OR live on that grid, so its resolution is the colour resolution, and OR over many packets saturates toward all-ones: a brightly lit entity tends to white unless the state's XOR is taking most of each packet — which is intended: bleaching is what happens under too many light sources in reality too, and here it is exactly the state's absorbing capacity being exceeded, how much it can remove from the equation per tick), the per-packet cost (Phase 1 sums deliveries per target today; XOR needs each edge's token before the sum — one token per edge, not per entity), and the debounce, which keys on incoming density and would need to see a state's own emission change as a change too.

## v0.10: Sound propagation
- Field carries additional signal types beyond light
- Acoustic wave simulation through entity graph

## v0.11: Water/reflection
- Reflective surface simulation
- Extends specular material properties
- Dynamic reflection via field re-emission

## v0.12: Scene scale (landscape, multiple creatures)
- Larger worlds beyond 512³
- Hierarchical or streaming field
- Multiple animated creatures

## v1.0: Real Engine alpha release
- Stable API and scene format
- Documentation and examples
- Performance targets for real-time use

---

## Future improvements (deferred)
- **Real shadow**: `edge_atten` currently dims a pipe and then Phase 2 renormalizes the emitter's weights, so the blocked energy is rerouted to its other pipes rather than absorbed — a relative deficit, not a missing photon. Absorb instead of renormalizing, and/or give vacuum relays a node transmittance so an atmosphere entity sitting inside solid density passes less on. Today's sun→atmosphere→floor path is all connection edges (`τ = 1`), so it is untouched by attenuation; see the shadow note in [ARCHITECTURE.md](ARCHITECTURE.md).
- **Per-pipe transmittance (retina Approach 2)**: `τ` is currently one value per entity toward the eye, so a partially occluded entity dims uniformly across its whole footprint. Computing `τ` per entity→receptor pipe would let a silhouette edge cut through a single source.
- **Hierarchical receptors (Approach 3)**: a receptor pyramid so distant or low-contrast regions link at a coarser level and only the busy parts of the image pay full pipe count — the relink cost is what bounds scene size today.
- **Observer-dependent resolution (LOD)**: entity detail varying with observer distance — near groups resolved finely, distant ones merged into a single source. Causal-cone driven LOD; largely subsumed by hierarchical receptors above.
- **Distance-based debounce**: Entities closer to observer use stricter debounce thresholds (update more often), distant entities debounce more aggressively.
