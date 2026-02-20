# Instructions for Code: Gamma Gradient Surface Model

## Context

The current approach spawns thousands of surface entities and relies on their tent deposits
overlapping to create a solid-looking body. This creates problems:
- Shell gaps (white lines between entity layers)
- Ring artifacts (regular spacing creates interference patterns)
- Opacity issues (not enough deposit density to look solid)
- Builder complexity (hollow shell calculation, layer counting)

New approach: **fewer, stronger entities at skeleton points. Let gamma spread create the
surface naturally.**

## Core Idea

Instead of approximating a surface with thousands of entities, place ~50-100 strong emitter
entities at the dino's skeleton (joints, spine, head center, tail segments). Each deposits
heavily into the 512³ grid. Gamma spreads outward through neighboring voxels via diffusion.

The ray marcher already reads grid density. It doesn't care if density came from 1 entity
or 1000. A single strong emitter spreading over 10 voxels looks the same as 100 weak
entities packed into that volume — but smoother, because diffusion has no gaps.

## What Changes

### Entity Spawning
Current: ~5000+ body entities at 0.3-0.4 spacing filling the metaball volume
New: ~50-100 skeleton entities at key anatomical points

Skeleton points (approximate — use existing metaball source centers):
- Head: 1-2 entities (head center, jaw)
- Neck: 2-3 entities along neck curve
- Body: 3-5 entities (chest, belly, hip)
- Tail: 5-8 entities along tail curve
- Each leg: 3-4 entities (hip joint, knee, foot)
- Each arm: 2-3 entities (shoulder, elbow, hand)
- Eye: 1 entity (special material — low deposit, high specular)

Each skeleton entity gets:
- HIGH deposit_strength (maybe 50-200× current value per entity)
- Color from the body group it belongs to (green for body, yellow for eye, etc.)
- Material properties (scatter, pass_through) same as current body material

### Gamma Spread = Flesh

Each tick, skeleton entities deposit into the grid. The deposit spreads via the existing
diffusion/decay system. The spread radius and falloff rate determine the "flesh thickness"
around each bone.

Key parameters to tune:
- **deposit_strength**: how much gamma per tick per skeleton entity (controls body density)
- **spread rate**: how fast gamma diffuses to neighbors (controls flesh thickness)  
- **decay rate**: how fast gamma fades (controls surface sharpness — high decay = thin skin,
  low decay = puffy body)

The balance between deposit, spread, and decay creates a steady-state density field.
That field IS the dino's body. The surface is where density drops below a visual threshold.

### What the Ray Marcher Sees

No change needed to the ray marcher. It already:
1. Steps through the 512³ grid
2. Reads density (gamma) at each voxel
3. Accumulates opacity based on density
4. Reads color from the deposit's color channel

The only difference: instead of seeing many small spikes (one per entity tent deposit),
it sees a smooth continuous field (gamma spread from skeleton). Smoother = more solid looking.

### Deposit Color

Each skeleton entity deposits with its group color. When gamma spreads from entity A (green)
and entity B (also green, different skeleton point), the overlapping region blends naturally.
The color field is continuous, not blocky.

For different-colored adjacent parts (green body next to yellow eye), the spread creates a
natural gradient at the boundary. No hard edge — a smooth color transition, like real tissue.

## Algorithm

```
def build_skeleton_entities(metaball_sources):
    """Place strong emitter entities at metaball centers and along curves between them."""
    skeleton = []
    
    for source in metaball_sources:
        # Place entity at each metaball center
        skeleton.append(Entity(
            pos=source.center,
            group=source.group,
            deposit_strength=source.weight * SKELETON_DEPOSIT_MULTIPLIER,
            color=source.group_color,
            material=source.group_material
        ))
    
    # Optionally: interpolate extra entities between connected metaball sources
    # e.g., along the spine between body and tail sources
    for connection in metaball_connections:
        src, dst = connection
        midpoint = (src.center + dst.center) / 2
        skeleton.append(Entity(
            pos=midpoint,
            group=src.group,  # or blend
            deposit_strength=(src.weight + dst.weight) / 2 * SKELETON_DEPOSIT_MULTIPLIER,
            color=lerp(src.group_color, dst.group_color, 0.5),
            material=src.group_material
        ))
    
    return skeleton
```

### Deposit Phase (each tick)

```
for entity in skeleton_entities:
    # Deposit into grid with WIDE radius (not 3x3x3 tent, more like 7x7x7 or larger)
    deposit_to_grid(
        grid=gamma_grid,
        center=entity.grid_pos,
        strength=entity.deposit_strength,
        radius=SKELETON_DEPOSIT_RADIUS,  # 5-10 voxels instead of 1-2
        color=entity.color,
        falloff='gaussian'  # smooth falloff, not tent
    )
```

The wider deposit radius + gaussian falloff gives smooth blobs that merge into continuous body.

### Spread + Decay (each tick, already exists)

The existing gamma spread and decay system does the rest. Each tick:
- Gamma diffuses to neighbor voxels (spread)
- Gamma decays by factor (e.g., 0.85)
- New deposits from skeleton entities replenish

Steady state: deposit rate = decay rate at some radius from each skeleton entity.
That radius = flesh thickness.

## Expected Benefits

- **No gaps**: continuous field has no shell boundaries
- **No rings**: gaussian deposit + diffusion = smooth, no interference
- **Naturally solid**: high deposit strength = high density = opaque
- **Smooth surface**: diffusion blurs everything, no blocky metaballs
- **Fewer entities**: 50-100 skeleton vs 5000+ surface entities
- **Faster building**: no BFS, no neighbor search, just place ~100 entities
- **Better FPS**: fewer entities to tick, same grid resolution
- **Natural LOD**: far from observer, the blur is fine. Close up, add more skeleton points.

## What Stays the Same

- 512³ grid (unchanged)
- Ray marcher (unchanged)  
- Radiation links (now FROM atmosphere TO nearest skeleton entity or grid region)
- Floor entities (unchanged — they work fine)
- Rock entities (unchanged)
- Sky emitters (unchanged)
- Atmosphere shell (still 2-3 layers around the dino)

## What About Radiation Links?

Current system: atmosphere entity → radiation link → surface entity → deposit

New system: atmosphere entities still exist as the 2-3 layer shell. But they connect to
the nearest SKELETON entity instead of thousands of surface entities. Or better: they
deposit directly into the grid based on the light they receive, letting it spread to
merge with the skeleton's gamma field.

This simplifies radiation links dramatically: atmosphere → grid, skeleton → grid. The grid
is the meeting point. No entity-to-entity radiation links needed for the body.

## Tuning Guide

If the dino looks too puffy/blobby:
- Increase decay rate (faster fade = thinner flesh)
- Decrease spread rate
- Decrease deposit radius

If the dino looks too skeletal/thin:
- Decrease decay rate (slower fade = thicker flesh)  
- Increase spread rate
- Increase deposit strength or deposit radius

If colors bleed between body parts:
- Reduce spread rate (less mixing)
- Or tag deposits with group ID and don't blend across groups

If the dino is too transparent:
- Increase deposit strength (more gamma = more density)

## Priority

1. Replace body entity spawning with skeleton entity placement at metaball centers
2. Widen deposit radius from 3×3×3 tent to 7×7×7 or larger gaussian
3. Tune deposit_strength, decay, spread for solid-looking body
4. Keep atmosphere, floor, rock, sky as-is
5. Adjust radiation links to connect atmosphere → skeleton entities
6. Tune per-body-part: eye gets special treatment (low deposit, reflective)

## Connection to Physics

This approach is more physically correct than the surface-entity approach. In the real
Cone Engine model:
- Mass IS a gamma concentration (self-gravitating eddy)
- The "surface" of any object is where its gamma field falls below detection threshold
- There are no hard surfaces in nature — just field gradients
- The skeleton entities are the dino's "quarks" — fundamental point sources whose
  overlapping fields create the extended body

The observer (ray marcher) sees density. Density comes from the field. The field comes
from point sources. The dino is a field, not a mesh.
