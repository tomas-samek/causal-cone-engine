# Instructions for Code: Graph-First Entity Builder

## Context

The current entity builder spawns entities at positions in 3D space (metaball field sampling
at regular spacing), then searches for neighbors within a connection radius using AABB spatial
hash. This is O(n²) even with the hash, and it chokes at spacing 0.3 — too many pairs to check.

The deeper problem: we spawn entities first, then hope they connect. That's backwards. The
graph topology should drive the spawning, not the other way around.

## The Idea: Grow the Graph Like a Crystal

Instead of sampling 3D space uniformly and then connecting:

1. Start from a SEED entity at the dino's center (e.g., body center of mass)
2. The seed checks the 6 lattice directions (+X, -X, +Y, -Y, +Z, -Z) at the chosen spacing
3. For each direction: evaluate the metaball field at that position
   - If above surface threshold → spawn a SURFACE entity, connect it to parent
   - If above interior threshold but below surface → spawn INTERIOR entity (or skip)
   - If below threshold → don't spawn (outside the body)
4. Each new entity repeats step 2-3, growing outward from the seed
5. Stop when all frontier positions are outside the metaball field

This is BFS/flood-fill on the metaball field. The connection IS the spawn — no neighbor
search needed. Each entity is born already connected to its parent.

## Key Properties

### Hollow Dino (Dense Skin, Empty Inside)
The critical optimization: we only need SURFACE entities for rendering. Interior entities
are occluded and never reach the observer via radiation links.

Two options:
- **Option A**: Only spawn entities within N voxels of the surface threshold. Evaluate the
  metaball field — if the value is far above the threshold (deep interior), skip. This gives
  a shell of ~2-3 entity layers thick.
- **Option B**: Spawn everything, then prune interior entities that have all 6 neighbors
  occupied (fully surrounded = invisible). Keep only entities with at least one empty neighbor
  direction (surface-adjacent).

Option A is faster (never spawns interior). Option B is simpler (spawn all, prune after).

### Variable Density
The spacing doesn't have to be uniform:
- Surface-facing entities: 0.3 spacing (dense, visible)
- Interior-adjacent: 0.4 spacing (structural, less visible)
- Deep interior: skip entirely

But for first implementation, uniform spacing + hollow shell is enough.

### Connection Building is Free
Every entity is spawned FROM a parent via a specific direction. That direction IS the
connection. No AABB search, no radius check, no O(n²). Connection count = entity count.

Additional connections (radiation links, cross-body structural links) can be added AFTER
the initial graph is built, using the spatial positions that are now known.

### Group Tagging
During BFS growth, tag each entity by which metaball source it's closest to (head, body,
neck, tail, legs). The metaball field already has per-source weights — the strongest
contributor determines the group. Same as current system, just evaluated during growth
instead of after spawning.

## Algorithm

```
def build_entity_graph(metaball_sources, spacing, surface_threshold):
    # 1. Find seed position (center of largest metaball source)
    seed_pos = metaball_sources[0].center  # body center
    
    # 2. BFS queue
    queue = deque()
    visited = set()  # grid positions already checked
    entities = []
    connections = []
    
    # 3. Quantize seed to grid
    seed_grid = quantize(seed_pos, spacing)
    queue.append((seed_grid, None))  # (position, parent_index)
    visited.add(seed_grid)
    
    # 4. Flood fill
    while queue:
        grid_pos, parent_idx = queue.popleft()
        world_pos = grid_to_world(grid_pos, spacing)
        
        # Evaluate metaball field at this position
        field_value = evaluate_metaball(world_pos, metaball_sources)
        
        if field_value < surface_threshold:
            continue  # outside body, skip
        
        # Optional: skip deep interior
        # if field_value > interior_cutoff and parent has all 6 neighbors:
        #     continue
        
        # Spawn entity
        entity_idx = len(entities)
        group = strongest_metaball_source(world_pos, metaball_sources)
        entities.append(Entity(pos=world_pos, group=group))
        
        # Connect to parent
        if parent_idx is not None:
            connections.append((parent_idx, entity_idx))
        
        # Explore 6 directions
        for direction in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            neighbor_grid = (grid_pos[0]+direction[0], 
                           grid_pos[1]+direction[1],
                           grid_pos[2]+direction[2])
            if neighbor_grid not in visited:
                visited.add(neighbor_grid)
                queue.append((neighbor_grid, entity_idx))
    
    return entities, connections
```

## Expected Benefits

- **No O(n²) neighbor search** — connections are built during BFS
- **0.3 spacing should be fast** — the flood fill only visits cells inside the metaball field
- **Hollow dino** — skip deep interior, save 50-70% of entities for surface density
- **Natural connectivity** — every entity is reachable from the seed (one connected graph)
- **Group tagging is free** — comes from metaball evaluation during spawn

## What Stays the Same

- Metaball field evaluation (same kernel: weight × max(0, 1-r²), threshold 1.0)
- Entity properties (color, material, group constants)
- Radiation link building (still needs AABB after entities exist, but only for surface entities)
- Grid deposit (tent function, 3x3x3)
- All rendering pipeline (unchanged)

## What Changes

- Entity spawning: uniform sampling → BFS flood fill
- Connection building: AABB search → built during spawn
- Entity count: full volume → surface shell only
- Builder performance: O(n²) → O(n) where n = entities inside metaball field

## Additional Context from Today's Discussion

The ghost dino problem: body is translucent, you can see sky through the torso. The eye is
paradoxically the most solid thing. This is because entity density is too low at 0.4 spacing
for the tent deposit to fill all voxels. Denser spacing (0.3) fixes it but the current
builder chokes.

The floor rings: concentric circle artifacts from regular entity spacing creating deposit
interference patterns. Fix: add small random jitter (±0.5 voxels) to each entity's deposit
position during Phase 3 grid deposit. Don't move the entity — just jitter where it writes
to the grid. Breaks regularity, costs nothing.

The atmosphere was already optimized to just 2-3 layers around the dino (60% fewer entities).
The saved entity budget should go into denser dino surface mesh.

## Priority

1. Implement BFS flood-fill builder (replaces current spatial sampling + AABB connection)
2. Add hollow shell option (skip entities where field_value > 2× threshold or similar)
3. Test at 0.3 spacing — should build fast and produce denser surface
4. Add deposit jitter (±0.3 voxels random offset) to kill floor/body rings
5. Tune body opacity to be solid (match current eye opacity level)
6. Reduce eye opacity to be slightly translucent with specular
