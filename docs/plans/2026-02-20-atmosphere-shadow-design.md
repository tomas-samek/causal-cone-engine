# v0.6 Design: Atmosphere Column & Shadow Tuning

## Problem

The current vacuum relay network spans a broad 60x40x50 region (~4400 entities at spacing 3). Most atmospheric entities are far from the dino, contribute nothing visually, and waste computation. Meanwhile, atmosphere scatters too much ambient light, washing out ground shadows.

## Solution

Concentrate atmosphere into a tight column around the dino, increase local scatter density, lower shader ambient floor, and bring the observer closer.

## Changes

### 1. Atmospheric Column (field.rs: spawn_demo_scene)

Replace the broad vacuum grid with a cylindrical column centered on the dino AABB.

- **Horizontal extent:** AABB center XZ +/- (max AABB half-width * 1.5) — adaptive radius
- **Vertical extent:** Floor (AABB min Y - 5) to sun height — light travels sun-to-ground
- **Spacing:** 3.0 (unchanged, maintains graph connectivity)
- **Density profile:** Gaussian-like radial falloff peaked at AABB center
  - `radial_frac = horizontal_dist / column_radius` (0 at center, 1 at edge)
  - `scatter = 0.0001 * (1.0 - radial_frac^2)`
  - `magnitude = 2.0 * (1.0 - radial_frac^2)`
- **Color:** Keep vertical gradient (warmer near ground, bluer up high)
- **Body exclusion:** Removed — vacuum near the body contributes to ambient fill; solid entities block light through the graph naturally
- **Entity reduction:** ~60% fewer vacuum entities (~1300-1800 vs ~4400)

### 2. Per-Tick Property Modulation (field.rs: tick)

Each tick, modulate vacuum entity scatter and deposit_magnitude based on distance from the current AABB center. This gives "follow AABB" behavior without moving entities or rebuilding edge topology.

```
aabb_center = (aabb_min + aabb_max) * 0.5
aabb_half = (aabb_max - aabb_min) * 0.5
column_radius = max(aabb_half.x, aabb_half.z) * 1.5

for each vacuum entity:
    horiz_dist = distance_xz(entity.position, aabb_center)
    radial_frac = (horiz_dist / column_radius).clamp(0, 1)
    falloff = 1.0 - radial_frac^2
    entity.scatter = base_scatter * falloff
    entity.deposit_magnitude = base_magnitude * falloff
```

Cost: minimal — iterates only over the smaller column entity set.

### 3. Shader Ambient Floor (field_sample.wgsl)

Lower ambient from 0.25 to 0.10. Ground shadows darken significantly. Fill light comes from the atmospheric propagation system rather than a shader hack. The 0.10 floor prevents unnaturally pitch-black surfaces facing away from the sun.

### 4. Observer Start Position (observer.rs)

Move from z=380 to z=310. Observer is now ~38 cells from the dino's face instead of ~108, providing a better view of the atmospheric effects and shadow detail.

## Files Changed

| File | Change |
|------|--------|
| `src/field.rs` | Vacuum generation (column), per-tick modulation |
| `shaders/field_sample.wgsl` | Ambient floor 0.25 -> 0.10 |
| `src/observer.rs` | Start position z=380 -> z=310 |

## Trade-offs

- Column topology is fixed at init. If the dino translates far, column won't follow physically (only property modulation adapts). Not an issue currently since the dino doesn't translate.
- Removing body exclusion means vacuum entities overlap with dino entities spatially. The graph's absorption handles shadow formation — no visual regression expected.
- Lower ambient floor may make back-facing surfaces darker than before. Monitor for visual quality.
