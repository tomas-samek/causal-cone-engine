use std::collections::HashMap;
use rayon::prelude::*;

// Diff Field — the persistent 3D texture that IS the universe.
//
// Entities deposit into it. Light propagates through CONNECTIONS between entities,
// not through grid diffusion. The grid is just the observer's retina.
//
// Entity → deposits own color to grid (rendering)
// Entity → sends deposit along connections to neighbors (propagation)
// Neighbor → accumulates incoming, adds to own deposit next tick (mixing)

/// Field resolution — 512³ (~134M cells, ~2GB at f32)
pub const FIELD_SIZE: u32 = 512;
pub const FIELD_CELLS: usize = (FIELD_SIZE * FIELD_SIZE * FIELD_SIZE) as usize;

// Entity group IDs for scene organization
pub const GROUP_NONE: u16 = 0;
pub const GROUP_BODY: u16 = 1;
pub const GROUP_BELLY: u16 = 2;
pub const GROUP_TAIL: u16 = 3;
pub const GROUP_TAIL_TIP: u16 = 4;
pub const GROUP_NECK: u16 = 5;
pub const GROUP_HEAD: u16 = 6;
pub const GROUP_JAW: u16 = 7;
pub const GROUP_MOUTH: u16 = 8;
pub const GROUP_EYE: u16 = 9;
pub const GROUP_LEG_L: u16 = 10;
pub const GROUP_FOOT_L: u16 = 11;
pub const GROUP_LEG_R: u16 = 12;
pub const GROUP_FOOT_R: u16 = 13;
pub const GROUP_ARM_L: u16 = 14;
pub const GROUP_ARM_R: u16 = 15;
pub const GROUP_SUN: u16 = 16;
pub const GROUP_FLOOR: u16 = 17;
pub const GROUP_VACUUM: u16 = 18;
pub const GROUP_ROCK: u16 = 19;

/// A single deposit in the field
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FieldCell {
    pub density: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
}

impl Default for FieldCell {
    fn default() -> Self {
        Self {
            density: 0.0,
            color_r: 0.0,
            color_g: 0.0,
            color_b: 0.0,
        }
    }
}

/// A deposit sitting on a connector (edge) — the pipe contents.
/// Each tick, source pushes new content in, old content gets delivered to destination.
#[derive(Clone, Copy, Default)]
pub struct EdgeDeposit {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub density: f32,
}

/// An entity that deposits into the field.
pub struct Entity {
    pub position: glam::Vec3,
    pub velocity: glam::Vec3,
    pub deposit_magnitude: f32,
    pub color: [f32; 3],
    /// How much incoming light passes through (0=fully absorbs, 1=transparent)
    pub pass_through: f32,
    /// Interior entities are fully surrounded — their emissions can't escape
    /// through the absorbing skin layers. Their energy is converted to heat.
    /// They still participate in graph propagation (conducting heat inward)
    /// but don't deposit to the grid (observer can't see them).
    pub is_heat: bool,
    /// Vacuum relay — invisible to observer, just passes light through graph.
    pub is_vacuum: bool,
    /// Atmospheric scatter — fraction of incoming light that bleeds into the grid.
    /// Air molecules redirect a tiny % of photons in all directions.
    /// 0.0 = pure vacuum (space), >0 = atmosphere.
    pub scatter: f32,
    /// Base scatter value set at generation (for per-tick modulation)
    pub base_scatter: f32,
    /// Base deposit magnitude set at generation (for per-tick modulation)
    pub base_magnitude: f32,
    /// Specular reflection — fraction of incoming bounced back unfiltered.
    /// Like the waxy cuticle on grass blades or wet surfaces.
    /// This light keeps its original color (mirror-like).
    pub specular: f32,
    /// Re-emission: fraction of absorbed light re-emitted as own deposit.
    /// Solid surfaces become secondary light sources when illuminated.
    /// Each entity represents billions of atoms acting coherently.
    pub reemit: f32,
    /// Accumulated re-emission energy (builds up from incoming light)
    pub reemit_r: f32,
    pub reemit_g: f32,
    pub reemit_b: f32,
    /// Index into DiffField's flat edge arrays
    pub edge_start: u32,
    /// Number of outgoing edges
    pub edge_count: u32,
    /// What arrived this tick from all incoming edges (read-only after delivery)
    pub incoming: EdgeDeposit,
    /// Which body part / scene element this entity belongs to
    pub group: u16,
    /// Outward-pointing surface normal (for skin texture oscillation)
    pub surface_normal: glam::Vec3,
    /// Current oscillation phase (radians)
    pub oscillation_phase: f32,
    /// Oscillation frequency (radians per tick)
    pub oscillation_freq: f32,
    /// Max oscillation offset in voxels
    pub oscillation_amplitude: f32,
    /// Previous grid cell index for clearing (-1 = none)
    pub prev_deposit_idx: i32,
    /// Density-weighted average direction of incoming light (normalized after delivery)
    pub incoming_dir: glam::Vec3,
    /// Debounce: previous incoming density for change detection
    pub prev_incoming_density: f32,
    /// Debounce: consecutive ticks with stable incoming (skip when >= edge_count)
    pub stable_ticks: u8,
}

impl Entity {
    pub fn new(position: glam::Vec3, velocity: glam::Vec3, deposit_magnitude: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            velocity,
            deposit_magnitude,
            color,
            pass_through: 0.5, // default: absorbs half
            is_heat: false,
            is_vacuum: false,
            scatter: 0.0,
            base_scatter: 0.0,
            base_magnitude: 0.0,
            specular: 0.0,
            reemit: 0.2, // surfaces re-emit 30% — absorb most, not mirror-like
            reemit_r: 0.0,
            reemit_g: 0.0,
            reemit_b: 0.0,
            edge_start: 0,
            edge_count: 0,
            incoming: EdgeDeposit::default(),
            group: GROUP_NONE,
            surface_normal: glam::Vec3::ZERO,
            oscillation_phase: 0.0,
            oscillation_freq: 0.0,
            oscillation_amplitude: 0.0,
            prev_deposit_idx: -1,
            incoming_dir: glam::Vec3::ZERO,
            prev_incoming_density: 0.0,
            stable_ticks: 0,
        }
    }
}

/// How oscillation phase is assigned across entities in a group
enum PhaseMode {
    /// All entities get the same base phase (coherent scales)
    Aligned(f32),
    /// Phase varies by position along an axis (ripple effect)
    Gradient(glam::Vec3),
    /// Random phase per entity (smooth blended)
    Random,
}

/// The diff field — CPU-side representation
pub struct DiffField {
    pub cells: Vec<FieldCell>,
    pub entities: Vec<Entity>,
    pub tick: u64,
    deliveries: Vec<EdgeDeposit>,
    delivery_dirs: Vec<glam::Vec3>,
    pub aabb_min: glam::Vec3,
    pub aabb_max: glam::Vec3,
    pub dirty_slabs: [bool; FIELD_SIZE as usize],
    // SoA edge storage — flat contiguous arrays for cache-friendly iteration
    edge_targets: Vec<usize>,
    edge_deposits: Vec<EdgeDeposit>,
    edge_gammas: Vec<f32>,
    edge_dirs: Vec<glam::Vec3>, // precomputed normalized direction per edge (source → target)
    // Reverse edge index — for each target entity, which edges point to it
    reverse_edge_sources: Vec<usize>, // source entity index
    reverse_edge_k: Vec<usize>,      // SoA edge index into edge_deposits/dirs/etc.
    reverse_start: Vec<u32>,          // per-target-entity start into reverse arrays
    reverse_count: Vec<u32>,          // per-target-entity incoming edge count
    // Per-tick active/visible sets (reused allocation)
    active_set: Vec<bool>,
    visible_set: Vec<bool>,
}

impl DiffField {
    pub fn new() -> Self {
        let mut field = Self {
            cells: vec![FieldCell::default(); FIELD_CELLS],
            entities: Vec::new(),
            tick: 0,
            deliveries: Vec::new(),
            delivery_dirs: Vec::new(),
            aabb_min: glam::Vec3::ZERO,
            aabb_max: glam::Vec3::splat(FIELD_SIZE as f32),
            dirty_slabs: [false; FIELD_SIZE as usize],
            edge_targets: Vec::new(),
            edge_deposits: Vec::new(),
            edge_gammas: Vec::new(),
            edge_dirs: Vec::new(),
            reverse_edge_sources: Vec::new(),
            reverse_edge_k: Vec::new(),
            reverse_start: Vec::new(),
            reverse_count: Vec::new(),
            active_set: Vec::new(),
            visible_set: Vec::new(),
        };

        let sp = field.spawn_demo_scene();

        // Sort entities by grid cell index (z-major) for cache-friendly Phase 3 deposits.
        // Must happen before build_connections which assigns edge indices.
        field.entities.sort_by_key(|e| {
            let x = e.position.x as u32;
            let y = e.position.y as u32;
            let z = e.position.z as u32;
            z * FIELD_SIZE * FIELD_SIZE + y * FIELD_SIZE + x
        });

        let connect_dist = (sp * 5.0).min(3.5);
        let radiation_dist = (sp * 15.0).min(10.0);
        field.build_connections(connect_dist);
        field.build_radiation_links(radiation_dist, connect_dist);

        // Skin texture: assign oscillation presets per body region.
        // Frequencies ~1000x slower than tail wag — texture shifts glacially, not per-frame.
        field.set_group_oscillation(GROUP_BODY,     0.0003, 0.3,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_HEAD,     0.0003, 0.3,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_NECK,     0.0003, 0.3,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_BELLY,    0.0005, 0.2,  PhaseMode::Random);
        field.set_group_oscillation(GROUP_LEG_L,    0.0003, 0.15, PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_LEG_R,    0.0003, 0.15, PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_FOOT_L,   0.0003, 0.1,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_FOOT_R,   0.0003, 0.1,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_TAIL,     0.0004, 0.3,  PhaseMode::Gradient(glam::Vec3::Z));
        field.set_group_oscillation(GROUP_TAIL_TIP, 0.0004, 0.3,  PhaseMode::Gradient(glam::Vec3::Z));
        field.set_group_oscillation(GROUP_JAW,      0.0004, 0.2,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_MOUTH,    0.0004, 0.2,  PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_ARM_L,    0.0003, 0.15, PhaseMode::Aligned(0.0));
        field.set_group_oscillation(GROUP_ARM_R,    0.0003, 0.15, PhaseMode::Aligned(0.0));
        // EYE, SUN, FLOOR, VACUUM: freq=0, amplitude=0 by default — no oscillation

        field.deliveries = vec![EdgeDeposit::default(); field.entities.len()];
        field.delivery_dirs = vec![glam::Vec3::ZERO; field.entities.len()];

        // Build reverse edge index for pull-based delivery
        field.build_reverse_edges();

        let n = field.entities.len();
        field.active_set = vec![false; n];
        field.visible_set = vec![false; n];

        // Seed AABB from solid entity positions so Phase 0 is tight from tick 1
        let mut aabb_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut aabb_max = glam::Vec3::ZERO;
        for e in &field.entities {
            if !e.is_vacuum && !e.is_heat {
                aabb_min = aabb_min.min(e.position - 1.0);
                aabb_max = aabb_max.max(e.position + 1.0);
            }
        }
        field.aabb_min = aabb_min.max(glam::Vec3::ZERO);
        field.aabb_max = aabb_max.min(glam::Vec3::splat(FIELD_SIZE as f32));

        field
    }

    fn index(x: u32, y: u32, z: u32) -> usize {
        (z * FIELD_SIZE * FIELD_SIZE + y * FIELD_SIZE + x) as usize
    }

    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && x < FIELD_SIZE as i32
            && y < FIELD_SIZE as i32
            && z < FIELD_SIZE as i32
    }

    /// Bin entity positions into a uniform spatial grid for O(n) neighbor queries.
    fn spatial_hash(positions: &[glam::Vec3], cell_size: f32) -> HashMap<(i32, i32, i32), Vec<usize>> {
        let mut map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, pos) in positions.iter().enumerate() {
            let key = (
                (pos.x / cell_size).floor() as i32,
                (pos.y / cell_size).floor() as i32,
                (pos.z / cell_size).floor() as i32,
            );
            map.entry(key).or_default().push(i);
        }
        map
    }

    /// Build connection edges and detect heat. Entities within connect_dist get bidirectional edges.
    /// Also detects interior (heat) entities that are fully surrounded by absorbing neighbors.
    fn build_connections(&mut self, connect_dist: f32) {
        let connect_dist_sq = connect_dist * connect_dist;
        let n = self.entities.len();

        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();

        // Collect connection edges via spatial hash (O(n) instead of O(n²))
        let grid = Self::spatial_hash(&positions, connect_dist);
        let mut temp_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut connection_count = 0u64;
        for i in 0..n {
            let pos = positions[i];
            let cx = (pos.x / connect_dist).floor() as i32;
            let cy = (pos.y / connect_dist).floor() as i32;
            let cz = (pos.z / connect_dist).floor() as i32;
            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if j <= i { continue; }
                                let dist_sq = positions[i].distance_squared(positions[j]);
                                if dist_sq < connect_dist_sq {
                                    temp_edges[i].push(j);
                                    temp_edges[j].push(i);
                                    connection_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Cap connection edges per entity — keep closest 26 (3x3x3 neighborhood)
        let max_connections: usize = 26;
        for i in 0..n {
            if temp_edges[i].len() > max_connections {
                temp_edges[i].sort_by(|&a, &b| {
                    let da = positions[i].distance_squared(positions[a]);
                    let db = positions[i].distance_squared(positions[b]);
                    da.partial_cmp(&db).unwrap()
                });
                temp_edges[i].truncate(max_connections);
            }
        }

        log::info!(
            "Graph built: {} entities, {} connection edges (avg {:.1} per entity, capped at {})",
            n,
            connection_count,
            (connection_count * 2) as f32 / n as f32,
            max_connections
        );

        // Detect interior (heat) entities.
        // If an entity has absorbing neighbors covering all 6 cardinal directions,
        // its own emissions can't escape — energy turns to heat.
        let mut heat_count = 0;
        for i in 0..n {
            let pos = positions[i];
            let mut has = [false; 6]; // +x -x +y -y +z -z

            for &t in &temp_edges[i] {
                if self.entities[t].pass_through >= 0.5 { continue; }
                let d = positions[t] - pos;
                if d.x > 0.3 { has[0] = true; }
                if d.x < -0.3 { has[1] = true; }
                if d.y > 0.3 { has[2] = true; }
                if d.y < -0.3 { has[3] = true; }
                if d.z > 0.3 { has[4] = true; }
                if d.z < -0.3 { has[5] = true; }
            }

            if has.iter().all(|&h| h) {
                self.entities[i].is_heat = true;
                heat_count += 1;
            }
        }

        log::info!(
            "Heat detection: {} surface (visible), {} interior (turned to heat)",
            n - heat_count,
            heat_count
        );

        // Compute surface normals: each surface entity's normal points away from its neighbors.
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let pos = positions[i];
            let mut neighbor_avg = glam::Vec3::ZERO;
            let mut solid_count = 0u32;
            for &t in &temp_edges[i] {
                if self.entities[t].is_vacuum || self.entities[t].is_heat { continue; }
                neighbor_avg += positions[t];
                solid_count += 1;
            }
            if solid_count > 0 {
                neighbor_avg /= solid_count as f32;
                let normal = (pos - neighbor_avg).normalize_or_zero();
                self.entities[i].surface_normal = normal;
            }
        }

        // Two-layer coloring: darken subsurface entities (those adjacent to heat/interior).
        // Surface entities keep full color; entities touching the interior are the "under-skin"
        // layer and get darkened. Creates visible depth when the surface oscillates.
        let mut subsurface_count = 0;
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let has_heat_neighbor = temp_edges[i].iter().any(|&t| self.entities[t].is_heat);
            if has_heat_neighbor {
                self.entities[i].color[0] *= 0.2;
                self.entities[i].color[1] *= 0.2;
                self.entities[i].color[2] *= 0.2;
                self.entities[i].deposit_magnitude *= 0.15;
                subsurface_count += 1;
            }
        }
        log::info!("Subsurface layer: {} entities darkened", subsurface_count);

        // Flatten into SoA
        self.flatten_edges(temp_edges);

        log::info!(
            "Edge SoA: {} connection edges, {:.1} MB contiguous",
            self.edge_targets.len(),
            (self.edge_targets.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<EdgeDeposit>())) as f64 / 1_048_576.0
        );
    }

    /// Check if any solid entity blocks the line of sight between two positions.
    /// Steps along the ray checking spatial hash cells for nearby blockers.
    fn ray_blocked(
        pos_a: glam::Vec3,
        pos_b: glam::Vec3,
        idx_a: usize,
        idx_b: usize,
        block_grid: &HashMap<(i32, i32, i32), Vec<usize>>,
        cell_size: f32,
        positions: &[glam::Vec3],
        block_radius_sq: f32,
    ) -> bool {
        let ab = pos_b - pos_a;
        let ab_len = ab.length();
        if ab_len < 0.001 { return false; }
        let ab_dir = ab / ab_len;

        let step = cell_size;
        let mut t = step;
        while t < ab_len - step * 0.5 {
            let sample = pos_a + ab_dir * t;
            let cx = (sample.x / cell_size).floor() as i32;
            let cy = (sample.y / cell_size).floor() as i32;
            let cz = (sample.z / cell_size).floor() as i32;

            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = block_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &eidx in bucket {
                                if eidx == idx_a || eidx == idx_b { continue; }
                                let ap = positions[eidx] - pos_a;
                                let proj = ap.dot(ab_dir);
                                if proj <= 0.0 || proj >= ab_len { continue; }
                                let closest = pos_a + ab_dir * proj;
                                let dist_sq = (positions[eidx] - closest).length_squared();
                                if dist_sq < block_radius_sq {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            t += step;
        }
        false
    }

    /// Build radiation links — direct surface-to-surface edges for non-vacuum, non-heat
    /// entities within max_dist. Skips pairs already connected (dist < connect_dist).
    fn build_radiation_links(&mut self, max_dist: f32, connect_dist: f32) {
        let max_dist_sq = max_dist * max_dist;
        let short_dist_sq = connect_dist * connect_dist;
        let n = self.entities.len();

        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();

        // Reconstruct per-entity edge lists from existing SoA
        let mut temp_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, entity) in self.entities.iter().enumerate() {
            let start = entity.edge_start as usize;
            let end = start + entity.edge_count as usize;
            for k in start..end {
                temp_edges[i].push(self.edge_targets[k]);
            }
        }

        // Build blocking grid from solid entities for line-of-sight raycasting
        let block_cell = connect_dist.max(1.0);
        let block_radius_sq = (connect_dist * 0.3).powi(2);
        let mut block_grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let pos = positions[i];
            let key = (
                (pos.x / block_cell).floor() as i32,
                (pos.y / block_cell).floor() as i32,
                (pos.z / block_cell).floor() as i32,
            );
            block_grid.entry(key).or_default().push(i);
        }

        // Add radiation links via spatial hash — collect candidates per entity, sorted by distance
        let max_radiation: usize = 10; // cap per entity
        let rad_grid = Self::spatial_hash(&positions, max_dist);
        let mut radiation_candidates: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut blocked_count = 0u64;
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let pos = positions[i];
            let cx = (pos.x / max_dist).floor() as i32;
            let cy = (pos.y / max_dist).floor() as i32;
            let cz = (pos.z / max_dist).floor() as i32;
            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = rad_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if j <= i { continue; }
                                if self.entities[j].is_vacuum || self.entities[j].is_heat { continue; }
                                let dist_sq = positions[i].distance_squared(positions[j]);
                                if dist_sq >= short_dist_sq && dist_sq < max_dist_sq {
                                    // Line-of-sight check: skip if solid entity blocks the path
                                    if Self::ray_blocked(
                                        pos, positions[j], i, j,
                                        &block_grid, block_cell, &positions, block_radius_sq,
                                    ) {
                                        blocked_count += 1;
                                        continue;
                                    }
                                    radiation_candidates[i].push((j, dist_sq));
                                    radiation_candidates[j].push((i, dist_sq));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Keep only the closest max_radiation links per entity
        let mut count = 0u64;
        for i in 0..n {
            let cands = &mut radiation_candidates[i];
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cands.truncate(max_radiation);
            for &(target, _) in cands.iter() {
                temp_edges[i].push(target);
            }
            count += cands.len() as u64;
        }
        log::info!(
            "Radiation links: {} directed edges (capped at {} per entity, {} blocked by LOS)",
            count, max_radiation, blocked_count
        );

        // Re-flatten into SoA
        self.flatten_edges(temp_edges);

        // Distance-weighted gammas: closer edges carry more energy (1/dist²).
        // Applies to ALL edges (short-range + radiation).
        for (i, entity) in self.entities.iter().enumerate() {
            let start = entity.edge_start as usize;
            let end = start + entity.edge_count as usize;
            for k in start..end {
                let target = self.edge_targets[k];
                let dist_sq = positions[i].distance_squared(positions[target]);
                self.edge_gammas[k] = 1.0 / dist_sq.max(0.1);
            }
        }

        log::info!(
            "Edge SoA: {} total directed edges, {:.1} MB contiguous",
            self.edge_targets.len(),
            (self.edge_targets.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<EdgeDeposit>())) as f64 / 1_048_576.0
        );
    }

    /// Flatten per-entity edge lists into SoA arrays for cache-friendly iteration.
    fn flatten_edges(&mut self, temp_edges: Vec<Vec<usize>>) {
        let total: usize = temp_edges.iter().map(|e| e.len()).sum();
        self.edge_targets = Vec::with_capacity(total);
        self.edge_dirs = Vec::with_capacity(total);
        self.edge_deposits = vec![EdgeDeposit::default(); total];
        self.edge_gammas = vec![1.0; total];
        let mut offset = 0u32;
        for (i, edges) in temp_edges.iter().enumerate() {
            self.entities[i].edge_start = offset;
            self.entities[i].edge_count = edges.len() as u32;
            let pos_i = self.entities[i].position;
            for &target in edges {
                self.edge_targets.push(target);
                self.edge_dirs.push((self.entities[target].position - pos_i).normalize_or_zero());
            }
            offset += edges.len() as u32;
        }
    }

    /// Build reverse edge index: for each entity, which edges (from other entities) point TO it.
    /// Enables pull-based Phase 1 delivery and backward BFS for active set computation.
    fn build_reverse_edges(&mut self) {
        let n = self.entities.len();
        let total = self.edge_targets.len();

        // Count incoming edges per target entity
        let mut counts = vec![0u32; n];
        for &target in &self.edge_targets {
            counts[target] += 1;
        }

        // Prefix sum for reverse_start
        self.reverse_start = vec![0u32; n];
        self.reverse_count = counts.clone();
        let mut offset = 0u32;
        for i in 0..n {
            self.reverse_start[i] = offset;
            offset += counts[i];
        }

        // Fill reverse arrays — need source entity for each edge
        self.reverse_edge_sources = vec![0usize; total];
        self.reverse_edge_k = vec![0usize; total];
        let mut fill_pos: Vec<u32> = self.reverse_start.clone();

        for (i, entity) in self.entities.iter().enumerate() {
            let start = entity.edge_start as usize;
            let end = start + entity.edge_count as usize;
            for k in start..end {
                let target = self.edge_targets[k];
                let pos = fill_pos[target] as usize;
                self.reverse_edge_sources[pos] = i;
                self.reverse_edge_k[pos] = k;
                fill_pos[target] += 1;
            }
        }

        log::info!(
            "Reverse edge index: {} entries for {} entities",
            total, n
        );
    }

    /// Compute active (pipeline chain) and visible (deposits to grid) sets from the observer's
    /// view-projection matrix. Only active entities get Phase 1-2 processing; only visible
    /// entities deposit to the grid in Phase 3.
    fn compute_active_set(&mut self, view_proj: glam::Mat4) {
        let n = self.entities.len();

        // Extract frustum planes from view-proj (Gribb-Hartmann method)
        let m = view_proj;
        let row0 = glam::Vec4::new(m.col(0).x, m.col(1).x, m.col(2).x, m.col(3).x);
        let row1 = glam::Vec4::new(m.col(0).y, m.col(1).y, m.col(2).y, m.col(3).y);
        let row2 = glam::Vec4::new(m.col(0).z, m.col(1).z, m.col(2).z, m.col(3).z);
        let row3 = glam::Vec4::new(m.col(0).w, m.col(1).w, m.col(2).w, m.col(3).w);

        let mut planes = [
            row3 + row0, // left
            row3 - row0, // right
            row3 + row1, // bottom
            row3 - row1, // top
            row3 + row2, // near
            row3 - row2, // far
        ];

        for plane in &mut planes {
            let len = glam::Vec3::new(plane.x, plane.y, plane.z).length();
            if len > 0.0 { *plane /= len; }
        }

        let margin = 10.0; // cells of margin around frustum for light bleed

        // Single pass: classify each entity as visible, active, or neither.
        // Emitters (deposit_magnitude >= 1.0) are always active — they're light sources.
        // In-frustum non-vacuum solids are visible (deposit to grid).
        // In-frustum entities (including vacuum) are active (participate in pipeline).
        for i in 0..n {
            let entity = &self.entities[i];

            // Emitters always active (sun, eyes, etc.)
            if entity.deposit_magnitude >= 1.0 {
                self.active_set[i] = true;
                self.visible_set[i] = false; // emitters may be vacuum (sun)
                continue;
            }

            if entity.is_heat {
                self.active_set[i] = false;
                self.visible_set[i] = false;
                continue;
            }

            let pos = entity.position;
            let mut inside = true;
            for plane in &planes {
                let dist = plane.x * pos.x + plane.y * pos.y + plane.z * pos.z + plane.w;
                if dist < -margin {
                    inside = false;
                    break;
                }
            }

            self.active_set[i] = inside;
            self.visible_set[i] = inside && !entity.is_vacuum;
        }
    }

    /// Assign oscillation parameters to all entities in a group.
    fn set_group_oscillation(&mut self, group: u16, freq: f32, amplitude: f32, phase_mode: PhaseMode) {
        for entity in &mut self.entities {
            if entity.group != group { continue; }
            entity.oscillation_freq = freq;
            entity.oscillation_amplitude = amplitude;
            entity.oscillation_phase = match &phase_mode {
                PhaseMode::Aligned(base) => *base,
                PhaseMode::Gradient(axis) => entity.position.dot(*axis),
                PhaseMode::Random => {
                    // Deterministic hash from position — no rand crate needed
                    let h = entity.position.x * 127.1 + entity.position.y * 311.7 + entity.position.z * 74.7;
                    (h.sin() * 43758.5453).fract() * std::f32::consts::TAU
                }
            };
        }
    }

    /// Fill an ellipsoid with entities
    fn fill_ellipsoid(
        &mut self,
        center: glam::Vec3,
        radii: glam::Vec3, // (rx, ry, rz)
        color: [f32; 3],
        magnitude: f32,
        spacing: f32,
        pass_through: f32,
        group: u16,
    ) {
        let rx = radii.x;
        let ry = radii.y;
        let rz = radii.z;
        let mut x = -rx;
        while x <= rx {
            let mut y = -ry;
            while y <= ry {
                let mut z = -rz;
                while z <= rz {
                    let nx = x / rx;
                    let ny = y / ry;
                    let nz = z / rz;
                    if nx * nx + ny * ny + nz * nz <= 1.0 {
                        let mut e = Entity::new(
                            center + glam::Vec3::new(x, y, z),
                            glam::Vec3::ZERO,
                            magnitude,
                            color,
                        );
                        e.pass_through = pass_through;
                        e.group = group;
                        self.entities.push(e);
                    }
                    z += spacing;
                }
                y += spacing;
            }
            x += spacing;
        }
    }

    fn spawn_demo_scene(&mut self) -> f32 {
        let center = FIELD_SIZE as f32 / 2.0;
        // Dino faces +Z, centered in field
        let base = glam::Vec3::new(center, center - 5.0, center);
        let green = [0.2, 0.6, 0.15];       // body green
        let dark_green = [0.15, 0.45, 0.1]; // darker accents
        let belly = [0.5, 0.65, 0.3];       // lighter belly
        let eye_color = [1.0, 0.8, 0.0];    // yellow eyes
        let mouth = [0.7, 0.2, 0.15];       // reddish mouth
        let sp = 0.4; // tight spacing — surface stays solid after interior becomes heat

        // --- Dino body via metaball field ---
        // Each body part is a metaball source. The combined field produces
        // smooth, seamless geometry — joints blend naturally.
        // Kernel: weight * max(0, 1 - r²)  where r² = (dx/rx)² + (dy/ry)² + (dz/rz)²
        // Entity spawned where combined field > 1.0.
        // Color/material interpolated from all contributors; group from strongest.
        struct Metaball {
            center: glam::Vec3,
            radii: glam::Vec3,
            weight: f32,
            color: [f32; 3],
            magnitude: f32,
            pass_through: f32,
            group: u16,
        }

        let balls = [
            Metaball { center: base + glam::Vec3::new(0.0, 5.0, 0.0),     radii: glam::Vec3::new(5.0, 6.0, 8.0),   weight: 5.0, color: green,      magnitude: 0.05, pass_through: 0.3,  group: GROUP_BODY },
            Metaball { center: base + glam::Vec3::new(0.0, 1.0, 0.0),     radii: glam::Vec3::new(4.5, 4.0, 7.0),   weight: 5.0, color: belly,      magnitude: 0.05, pass_through: 0.45, group: GROUP_BELLY },
            Metaball { center: base + glam::Vec3::new(0.0, 5.5, -12.0),   radii: glam::Vec3::new(2.5, 2.5, 7.0),   weight: 5.0, color: green,      magnitude: 0.05, pass_through: 0.3,  group: GROUP_TAIL },
            Metaball { center: base + glam::Vec3::new(0.0, 5.5, -20.0),   radii: glam::Vec3::new(1.2, 1.2, 4.0),   weight: 5.0, color: dark_green, magnitude: 0.05, pass_through: 0.25, group: GROUP_TAIL_TIP },
            Metaball { center: base + glam::Vec3::new(0.0, 10.0, 8.0),    radii: glam::Vec3::new(3.0, 5.0, 3.0),   weight: 5.0, color: green,      magnitude: 0.05, pass_through: 0.3,  group: GROUP_NECK },
            Metaball { center: base + glam::Vec3::new(0.0, 16.0, 10.0),   radii: glam::Vec3::new(3.5, 3.0, 5.0),   weight: 5.0, color: green,      magnitude: 0.05, pass_through: 0.3,  group: GROUP_HEAD },
            Metaball { center: base + glam::Vec3::new(0.0, 13.5, 12.0),   radii: glam::Vec3::new(2.5, 1.5, 4.0),   weight: 5.0, color: dark_green, magnitude: 0.05, pass_through: 0.25, group: GROUP_JAW },
            Metaball { center: base + glam::Vec3::new(0.0, 14.5, 13.0),   radii: glam::Vec3::new(2.0, 0.8, 3.0),   weight: 5.0, color: mouth,      magnitude: 0.05, pass_through: 0.15, group: GROUP_MOUTH },
            Metaball { center: base + glam::Vec3::new(3.0, 17.0, 12.0),   radii: glam::Vec3::new(0.8, 0.8, 0.8),   weight: 8.0, color: eye_color,  magnitude: 4.0,  pass_through: 0.1,  group: GROUP_EYE },
            Metaball { center: base + glam::Vec3::new(-3.0, 17.0, 12.0),  radii: glam::Vec3::new(0.8, 0.8, 0.8),   weight: 8.0, color: eye_color,  magnitude: 4.0,  pass_through: 0.1,  group: GROUP_EYE },
            Metaball { center: base + glam::Vec3::new(3.0, -3.0, 1.0),    radii: glam::Vec3::new(2.0, 5.0, 2.5),   weight: 5.0, color: dark_green, magnitude: 0.05, pass_through: 0.25, group: GROUP_LEG_L },
            Metaball { center: base + glam::Vec3::new(3.0, -8.0, 2.0),    radii: glam::Vec3::new(2.5, 1.0, 4.0),   weight: 5.0, color: dark_green, magnitude: 0.05, pass_through: 0.25, group: GROUP_FOOT_L },
            Metaball { center: base + glam::Vec3::new(-3.0, -3.0, 1.0),   radii: glam::Vec3::new(2.0, 5.0, 2.5),   weight: 5.0, color: dark_green, magnitude: 0.05, pass_through: 0.25, group: GROUP_LEG_R },
            Metaball { center: base + glam::Vec3::new(-3.0, -8.0, 2.0),   radii: glam::Vec3::new(2.5, 1.0, 4.0),   weight: 5.0, color: dark_green, magnitude: 0.05, pass_through: 0.25, group: GROUP_FOOT_R },
            Metaball { center: base + glam::Vec3::new(4.5, 6.0, 5.0),     radii: glam::Vec3::new(1.0, 2.5, 1.0),   weight: 5.0, color: green,      magnitude: 0.05, pass_through: 0.3,  group: GROUP_ARM_R },
            Metaball { center: base + glam::Vec3::new(-4.5, 6.0, 5.0),    radii: glam::Vec3::new(1.0, 2.5, 1.0),   weight: 5.0, color: green,      magnitude: 0.05, pass_through: 0.3,  group: GROUP_ARM_L },
        ];

        // Bounding box of all metaball influence regions
        let mut bb_min = glam::Vec3::splat(f32::MAX);
        let mut bb_max = glam::Vec3::splat(f32::MIN);
        for b in &balls {
            bb_min = bb_min.min(b.center - b.radii);
            bb_max = bb_max.max(b.center + b.radii);
        }

        // Sample the combined metaball field at entity spacing
        let mut x = bb_min.x;
        while x <= bb_max.x {
            let mut y = bb_min.y;
            while y <= bb_max.y {
                let mut z = bb_min.z;
                while z <= bb_max.z {
                    let pos = glam::Vec3::new(x, y, z);

                    // Accumulate field contributions from all sources
                    let mut total_field = 0.0f32;
                    let mut acc_r = 0.0f32;
                    let mut acc_g = 0.0f32;
                    let mut acc_b = 0.0f32;
                    let mut acc_mag = 0.0f32;
                    let mut acc_pt = 0.0f32;
                    let mut best_group = 0u16;
                    let mut best_c = 0.0f32;

                    for b in &balls {
                        let d = (pos - b.center) / b.radii;
                        let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                        if r2 >= 1.0 { continue; }
                        let c = b.weight * (1.0 - r2);
                        total_field += c;
                        acc_r += b.color[0] * c;
                        acc_g += b.color[1] * c;
                        acc_b += b.color[2] * c;
                        acc_mag += b.magnitude * c;
                        acc_pt += b.pass_through * c;
                        if c > best_c {
                            best_c = c;
                            best_group = b.group;
                        }
                    }

                    if total_field > 1.0 {
                        let inv = 1.0 / total_field;
                        let mut e = Entity::new(
                            pos,
                            glam::Vec3::ZERO,
                            acc_mag * inv,
                            [acc_r * inv, acc_g * inv, acc_b * inv],
                        );
                        e.pass_through = acc_pt * inv;
                        e.group = best_group;
                        self.entities.push(e);
                    }

                    z += sp;
                }
                y += sp;
            }
            x += sp;
        }
        log::info!("Metaball dino: {} entities", self.entities.len());

        // ROCK — small boulder on the ground
        let rock_color = [0.4, 0.35, 0.25];
        self.fill_ellipsoid(
            base + glam::Vec3::new(15.0, -4.5, 25.0),
            glam::Vec3::new(15.0, 2.0, 15.0),
            rock_color, 0.15, 0.8, 0.01, GROUP_ROCK,
        );
        // Bump reemit for rock entities
        for e in self.entities.iter_mut().rev() {
            if e.group != GROUP_ROCK { break; }
            e.reemit = 0.1;
        }

        // SUN — large flat disc high above the scene, like a sky panel.
        // Many emitters sending parallel light downward through vacuum network.
        // Offset slightly in +Z so shadow falls behind (toward -Z).
        let sun_y = center + 30.0;
        let sun_spacing = 3.0;
        for sx in -8..=8 {
            for sz in -8..=8 {
                // Circular disc
                if sx * sx + sz * sz > 64 { continue; }
                let mut light = Entity::new(
                    glam::Vec3::new(
                        center + sx as f32 * sun_spacing,
                        sun_y,
                        center + sz as f32 * sun_spacing,
                    ),
                    glam::Vec3::ZERO,
                    20.0, // directional highlight — sky dome handles ambient fill
                    [1.0, 0.9, 0.5],
                );
                light.pass_through = 1.0; // pure emitter
                light.is_vacuum = true; // sun emits through graph, not visible in grid
                light.group = GROUP_SUN;
                self.entities.push(light);
            }
        }
        log::info!("Sun disc at y={}", sun_y);

        // FLOOR — tight spacing so tiles connect and light propagates across surface
        // Light flows: source → head → body → legs → feet → floor tiles
        // Tiles near feet = bright. Tiles far from feet = dark. Shadow emerges.
        for x in 0..40 {
            for z in 0..40 {
                let mut e = Entity::new(
                    glam::Vec3::new(
                        center + x as f32 * 1.5 - 30.0,
                        center - 13.0,
                        center + z as f32 * 1.5 - 20.0,
                    ),
                    glam::Vec3::ZERO,
                    0.02, // dark until lit
                    if (x + z) % 2 == 0 {
                        [0.45, 0.35, 0.2]  // dirt
                    } else {
                        [0.3, 0.5, 0.15]   // grass
                    },
                );
                e.pass_through = 0.5;
                e.specular = 0.3;     // waxy cuticle
                e.reemit = 0.3;       // low re-emission — shadows stay dark
                e.group = GROUP_FLOOR;
                self.entities.push(e);
            }
        }

        let before_vacuum = self.entities.len();

        // ATMOSPHERIC COLUMN — concentrated relay network around the dino.
        // Cylindrical column centered on solid entity AABB. Vacuum entities relay
        // light from sun to dino and scatter a fraction into the grid (atmosphere).
        // Density peaked at column center, fading radially outward.

        // Compute solid entity AABB for column centering
        let mut solid_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut solid_max = glam::Vec3::ZERO;
        for e in &self.entities {
            if !e.is_vacuum && !e.is_heat {
                solid_min = solid_min.min(e.position);
                solid_max = solid_max.max(e.position);
            }
        }
        let solid_center = (solid_min + solid_max) * 0.5;
        let solid_half = (solid_max - solid_min) * 0.5;
        let column_radius = solid_half.x.max(solid_half.z) * 1.5;

        let vac_spacing = 3.0;
        let col_y_min = solid_min.y - 5.0;
        let col_y_max = sun_y;

        let mut vx = solid_center.x - column_radius;
        while vx <= solid_center.x + column_radius {
            let mut vy = col_y_min;
            while vy <= col_y_max {
                let mut vz = solid_center.z - column_radius;
                while vz <= solid_center.z + column_radius {
                    let pos = glam::Vec3::new(vx, vy, vz);

                    // Radial distance from column center (XZ plane only)
                    let dx = pos.x - solid_center.x;
                    let dz = pos.z - solid_center.z;
                    let horiz_dist = (dx * dx + dz * dz).sqrt();
                    let radial_frac = (horiz_dist / column_radius).clamp(0.0, 1.0);

                    // Skip entities outside the column radius
                    if radial_frac >= 1.0 {
                        vz += vac_spacing;
                        continue;
                    }

                    let falloff = 1.0 - radial_frac * radial_frac;

                    let mut e = Entity::new(
                        pos,
                        glam::Vec3::ZERO,
                        0.0,
                        [0.0, 0.0, 0.0],
                    );
                    e.pass_through = 0.95;
                    e.is_vacuum = true;
                    e.group = GROUP_VACUUM;

                    // Density profile: peaked at center, fading radially
                    let base_scatter_val = 0.0001 * falloff;
                    let base_mag_val = 2.0 * falloff;
                    e.scatter = base_scatter_val;
                    e.deposit_magnitude = base_mag_val;
                    e.base_scatter = base_scatter_val;
                    e.base_magnitude = base_mag_val;

                    // Vertical color gradient (warmer near ground, bluer up high)
                    if vy < sun_y {
                        let height_frac = (vy - col_y_min) / (sun_y - col_y_min);
                        let bottom_weight = 1.0 - height_frac;
                        e.color = [
                            0.4 + 0.4 * bottom_weight,
                            0.5 + 0.35 * bottom_weight,
                            0.9 + 0.1 * height_frac,
                        ];
                    }

                    self.entities.push(e);

                    vz += vac_spacing;
                }
                vy += vac_spacing;
            }
            vx += vac_spacing;
        }

        log::info!(
            "Demo scene: {} entities ({} solid + {} vacuum)",
            self.entities.len(),
            before_vacuum,
            self.entities.len() - before_vacuum
        );

        sp
    }

    /// Run one simulation tick — push-driven pipe propagation
    pub fn tick(&mut self, view_proj: glam::Mat4) {
        // Compute active set: which entity chains feed into what the observer sees
        self.compute_active_set(view_proj);

        // Atmospheric column modulation: update vacuum scatter/magnitude based on
        // distance from current AABB center. Gives "follow AABB" behavior.
        {
            let aabb_center = (self.aabb_min + self.aabb_max) * 0.5;
            let aabb_half = (self.aabb_max - self.aabb_min) * 0.5;
            let column_radius = aabb_half.x.max(aabb_half.z) * 1.5;
            let inv_radius = if column_radius > 0.1 { 1.0 / column_radius } else { 0.0 };

            for entity in &mut self.entities {
                if !entity.is_vacuum || entity.base_scatter <= 0.0 { continue; }
                let dx = entity.position.x - aabb_center.x;
                let dz = entity.position.z - aabb_center.z;
                let horiz_dist = (dx * dx + dz * dz).sqrt();
                let radial_frac = (horiz_dist * inv_radius).clamp(0.0, 1.0);
                let falloff = 1.0 - radial_frac * radial_frac;
                entity.scatter = entity.base_scatter * falloff;
                entity.deposit_magnitude = entity.base_magnitude * falloff;
            }
        }

        // Phase 0: AABB-restricted decay — only touch cells near geometry
        // The AABB from previous tick tells us where deposits exist.
        // Margin of 40 covers vacuum scatter region generously.
        let margin = 40.0f32;
        let fs = FIELD_SIZE as usize;
        let dx_min = (self.aabb_min.x - margin).max(0.0) as usize;
        let dx_max = ((self.aabb_max.x + margin) as usize + 1).min(fs);
        let dy_min = (self.aabb_min.y - margin).max(0.0) as usize;
        let dy_max = ((self.aabb_max.y + margin) as usize + 1).min(fs);
        let dz_min = (self.aabb_min.z - margin).max(0.0) as usize;
        let dz_max = ((self.aabb_max.z + margin) as usize + 1).min(fs);

        for z in 0..fs {
            if !self.dirty_slabs[z] { continue; }

            if z < dz_min || z >= dz_max {
                // Outside AABB z range — stale deposits, clear and mark clean
                let slab_base = z * fs * fs;
                for cell in &mut self.cells[slab_base..slab_base + fs * fs] {
                    *cell = FieldCell::default();
                }
                self.dirty_slabs[z] = false;
                continue;
            }

            // Within AABB z range — decay only the AABB sub-rectangle
            let slab_base = z * fs * fs;
            let mut any_nonzero = false;
            for y in dy_min..dy_max {
                let row_base = slab_base + y * fs;
                for x in dx_min..dx_max {
                    let cell = &mut self.cells[row_base + x];
                    cell.density *= 0.85;
                    cell.color_r *= 0.85;
                    cell.color_g *= 0.85;
                    cell.color_b *= 0.85;
                    if cell.density < 0.001 {
                        cell.density = 0.0;
                        cell.color_r = 0.0;
                        cell.color_g = 0.0;
                        cell.color_b = 0.0;
                    } else {
                        any_nonzero = true;
                    }
                }
            }
            if !any_nonzero { self.dirty_slabs[z] = false; }
        }

        // Phase 1: PUSH-BASED DELIVER with active filtering.
        // Sequential SoA reads for cache-friendly iteration. Skip writes to inactive targets.
        self.deliveries.fill(EdgeDeposit::default());
        self.delivery_dirs.fill(glam::Vec3::ZERO);
        for (_src_idx, entity) in self.entities.iter().enumerate() {
            let start = entity.edge_start as usize;
            let end = start + entity.edge_count as usize;
            for k in start..end {
                let target = self.edge_targets[k];
                if !self.active_set[target] { continue; }
                let dep = &self.edge_deposits[k];
                self.deliveries[target].r += dep.r;
                self.deliveries[target].g += dep.g;
                self.deliveries[target].b += dep.b;
                self.deliveries[target].density += dep.density;
                if dep.density > 0.001 {
                    self.delivery_dirs[target] += self.edge_dirs[k] * dep.density;
                }
            }
        }
        // Apply deliveries + build re-emission + debounce tracking for active entities
        let n = self.entities.len();
        for i in 0..n {
            if !self.active_set[i] { continue; }
            let entity = &mut self.entities[i];
            let new_density = self.deliveries[i].density;
            entity.incoming = self.deliveries[i];
            entity.incoming_dir = self.delivery_dirs[i].normalize_or_zero();

            // Debounce: detect if incoming has changed
            let diff = (new_density - entity.prev_incoming_density).abs();
            let threshold = 0.01 * entity.prev_incoming_density.abs().max(0.01);
            if diff < threshold {
                entity.stable_ticks = entity.stable_ticks.saturating_add(1);
            } else {
                entity.stable_ticks = 0;
            }
            entity.prev_incoming_density = new_density;

            if !entity.is_vacuum && entity.reemit > 0.0 && new_density > 0.01 {
                let re = entity.reemit;
                let cap = 50.0;
                entity.reemit_r = (self.deliveries[i].r * re * entity.color[0]).min(cap);
                entity.reemit_g = (self.deliveries[i].g * re * entity.color[1]).min(cap);
                entity.reemit_b = (self.deliveries[i].b * re * entity.color[2]).min(cap);
            }

            if entity.is_heat && new_density > 1.0 {
                entity.is_heat = false;
            }
        }

        // Phase 2: PUSH — each entity pushes new content into its pipes (parallel).
        // New content = entity's own emission + pass-through of incoming (depleted).
        // This REPLACES what was in the pipe (old content was delivered in Phase 1).
        // Safe to parallelize: each entity writes to its own non-overlapping edge range.
        let cutoff: f32 = 0.01;
        let directionality: f32 = 0.8; // 0=isotropic, 1=fully directional

        // Dynamic connector density: scale gammas by observer distance.
        // Close entities propagate at full weight; distant ones at reduced weight.
        let inv_vp = view_proj.inverse();
        let observer_pos = glam::Vec3::new(inv_vp.col(3).x, inv_vp.col(3).y, inv_vp.col(3).z);
        let distance_factors: Vec<f32> = self.entities.iter().map(|e| {
            let dist = (e.position - observer_pos).length();
            // Full weight within 30 cells, linear falloff to 0.1 at 130+ cells
            (1.0 - (dist - 30.0).max(0.0) / 100.0).clamp(0.1, 1.0)
        }).collect();

        // Collect per-entity edge ranges for parallel slicing
        let edge_ranges: Vec<(usize, usize)> = self.entities.iter().map(|e| {
            (e.edge_start as usize, e.edge_count as usize)
        }).collect();

        let entities = &self.entities;
        let edge_gammas = &self.edge_gammas;
        let edge_dir_arr = &self.edge_dirs;
        let active = &self.active_set;
        let edge_deposits = &mut self.edge_deposits;

        let edge_deposit_slice = edge_deposits.as_mut_slice();
        let edge_base = edge_deposit_slice.as_mut_ptr() as usize;
        let edge_len = edge_deposit_slice.len();

        entities.par_iter().enumerate().zip(edge_ranges.par_iter()).for_each(|((idx, entity), &(start, count))| {
            // Skip inactive entities and debounced entities (stable input for edge_count ticks)
            if count == 0 || !active[idx] || entity.stable_ticks >= entity.edge_count.max(1) as u8 { return; }
            let end = start + count;

            let deposits = unsafe {
                assert!(end <= edge_len);
                let ptr = edge_base as *mut EdgeDeposit;
                std::slice::from_raw_parts_mut(ptr.add(start), count)
            };

            let has_dir = entity.is_vacuum && entity.incoming_dir.length_squared() > 0.01;
            let mut total_weight: f32 = 0.0;
            for (local_k, dep) in deposits.iter_mut().enumerate() {
                let k = start + local_k;
                let mut w = edge_gammas[k] * distance_factors[idx];
                if has_dir {
                    let edge_dir = edge_dir_arr[k];
                    let alignment = edge_dir.dot(entity.incoming_dir);
                    let bias = (1.0 + alignment * directionality) * 0.5;
                    w *= bias.max(0.01);
                }
                dep.density = w;
                total_weight += w;
            }
            if total_weight < 0.001 { return; }

            let mag = entity.deposit_magnitude;
            let own_r = entity.color[0] * mag + entity.reemit_r;
            let own_g = entity.color[1] * mag + entity.reemit_g;
            let own_b = entity.color[2] * mag + entity.reemit_b;
            let own_d = mag + entity.reemit_r + entity.reemit_g + entity.reemit_b;

            let (pass_r, pass_g, pass_b, pass_d) = if entity.incoming.density > cutoff {
                let pt = entity.pass_through;
                if entity.is_vacuum {
                    (
                        entity.incoming.r * pt,
                        entity.incoming.g * pt,
                        entity.incoming.b * pt,
                        entity.incoming.density * pt,
                    )
                } else {
                    let spec = entity.specular;
                    let diff = 1.0 - spec;
                    (
                        (entity.incoming.r * spec + entity.incoming.r * diff * entity.color[0]) * pt,
                        (entity.incoming.g * spec + entity.incoming.g * diff * entity.color[1]) * pt,
                        (entity.incoming.b * spec + entity.incoming.b * diff * entity.color[2]) * pt,
                        entity.incoming.density * pt,
                    )
                }
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            let total_r = own_r + pass_r;
            let total_g = own_g + pass_g;
            let total_b = own_b + pass_b;
            let total_d = own_d + pass_d;
            for dep in deposits.iter_mut() {
                let w = dep.density / total_weight;
                dep.r = total_r * w;
                dep.g = total_g * w;
                dep.b = total_b * w;
                dep.density = total_d * w;
            }
        });

        // Phase 3: entities deposit to grid (only visible entities)
        let mut aabb_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut aabb_max = glam::Vec3::splat(0.0);
        for (ent_idx, entity) in self.entities.iter_mut().enumerate() {
            // Move entity (all entities, not just visible — keeps positions consistent)
            entity.position += entity.velocity;

            // Bounce
            for i in 0..3 {
                if entity.position[i] < 1.0 || entity.position[i] >= (FIELD_SIZE - 1) as f32 {
                    entity.velocity[i] *= -1.0;
                    entity.position[i] = entity.position[i].clamp(1.0, (FIELD_SIZE - 2) as f32);
                }
            }

            // Heat: interior, light can't escape. Always skip.
            if entity.is_heat { continue; }

            // Vacuum with atmosphere: scatter into grid only if active
            if entity.is_vacuum {
                if !self.active_set[ent_idx] { continue; }
                if entity.scatter > 0.0 && entity.incoming.density > 0.1 {
                    let ix = entity.position.x as i32;
                    let iy = entity.position.y as i32;
                    let iz = entity.position.z as i32;
                    if Self::in_bounds(ix, iy, iz) {
                        let idx = Self::index(ix as u32, iy as u32, iz as u32);
                        let s = entity.scatter;
                        let intensity = entity.incoming.density * s;
                        let cell = &mut self.cells[idx];
                        cell.density = (cell.density + intensity).min(50.0);
                        // Scatter uses air's own color (blue Rayleigh), not incoming color
                        cell.color_r = (cell.color_r + entity.color[0] * intensity).min(50.0);
                        cell.color_g = (cell.color_g + entity.color[1] * intensity).min(50.0);
                        cell.color_b = (cell.color_b + entity.color[2] * intensity).min(50.0);
                        self.dirty_slabs[iz as usize] = true;
                    }
                }
                continue;
            }

            // Track AABB from non-vacuum entities (tight box around solid geometry)
            aabb_min = aabb_min.min(entity.position - 1.0);
            aabb_max = aabb_max.max(entity.position + 1.0);

            // Skip deposit for non-visible solid entities (reactive: only render subscribed chains)
            if !self.visible_set[ent_idx] { continue; }

            // Skin texture: offset deposit along surface normal
            let mut deposit_pos = entity.position;
            if entity.oscillation_amplitude > 0.0 {
                let offset = entity.surface_normal * entity.oscillation_phase.sin() * entity.oscillation_amplitude;
                deposit_pos += offset;
            }

            // Tail wag: shift deposit position in X via sine wave (adds on top of texture).
            // Tip has max amplitude, tapers toward body. Traveling wave along Z.
            if entity.group == GROUP_TAIL || entity.group == GROUP_TAIL_TIP {
                let time = self.tick as f32 / 30.0;
                let frequency = std::f32::consts::PI; // ~2 sec period
                let center_z = FIELD_SIZE as f32 / 2.0;
                // z_frac: 0.0 at body junction (z=center), 1.0 at tail tip (z=center-24)
                let z_frac = ((center_z - entity.position.z) / 24.0).clamp(0.0, 1.0);
                let amplitude = 3.0 * z_frac; // tip swings 3 cells, body junction ~0
                let phase = time * frequency + z_frac * 2.0; // traveling wave
                deposit_pos.x += amplitude * phase.sin();
            }

            // Jaw open/close: rotate jaw downward around pivot at back of jaw.
            // Front of jaw swings down, back stays nearly fixed. Mouth follows.
            if entity.group == GROUP_JAW || entity.group == GROUP_MOUTH {
                let time = self.tick as f32 / 30.0;
                let frequency = std::f32::consts::PI * 0.5; // ~4 sec full cycle
                let center = FIELD_SIZE as f32 / 2.0;
                let pivot_z = center + 8.0;  // back of jaw (base z-offset from center)

                // z_frac: 0 at pivot (back), 1 at front of jaw
                let z_frac = ((entity.position.z - pivot_z) / 8.0).clamp(0.0, 1.0);

                // Jaw only opens DOWN (abs), never pushes up into head
                let open_amount = (time * frequency).sin().abs();
                deposit_pos.y -= z_frac * 1.5 * open_amount;
            }

            // 3x3x3 tent-weight splat — wider footprint fills surface gaps.
            let base_x = deposit_pos.x.floor() as i32;
            let base_y = deposit_pos.y.floor() as i32;
            let base_z = deposit_pos.z.floor() as i32;

            // Clear previous 2x2x2 footprint if base cell changed
            let new_base_idx = if Self::in_bounds(base_x, base_y, base_z) {
                Self::index(base_x as u32, base_y as u32, base_z as u32) as i32
            } else { -1 };
            if entity.prev_deposit_idx >= 0 && entity.prev_deposit_idx != new_base_idx {
                let prev = entity.prev_deposit_idx as usize;
                let pz = (prev / (FIELD_SIZE * FIELD_SIZE) as usize) as i32;
                let py = ((prev % (FIELD_SIZE * FIELD_SIZE) as usize) / FIELD_SIZE as usize) as i32;
                let px = (prev % FIELD_SIZE as usize) as i32;
                // 3x3x3 centered clear — catches animation ghosts from tail/jaw swings
                for dz in -1..2i32 {
                    for dy in -1..2i32 {
                        for dx in -1..2i32 {
                            let cx = px + dx;
                            let cy = py + dy;
                            let cz = pz + dz;
                            if Self::in_bounds(cx, cy, cz) {
                                let idx = Self::index(cx as u32, cy as u32, cz as u32);
                                self.cells[idx] = FieldCell::default();
                                self.dirty_slabs[cz as usize] = true;
                            }
                        }
                    }
                }
            }
            entity.prev_deposit_idx = new_base_idx;

            let mag = entity.deposit_magnitude;
            let absorbed = 1.0 - entity.pass_through;

            let total_r = entity.color[0] * mag + entity.incoming.r * absorbed * entity.color[0] + entity.reemit_r;
            let total_g = entity.color[1] * mag + entity.incoming.g * absorbed * entity.color[1] + entity.reemit_g;
            let total_b = entity.color[2] * mag + entity.incoming.b * absorbed * entity.color[2] + entity.reemit_b;
            let total_d = mag + entity.incoming.density * absorbed;

            // Deposit to 3x3x3 with tent weights — wider footprint fills surface gaps.
            // Boost 10x to compensate for wider spread.
            let total_r = total_r * 10.0;
            let total_g = total_g * 10.0;
            let total_b = total_b * 10.0;
            let total_d = total_d * 10.0;
            for dz in -1..2i32 {
                let cz_f = base_z as f32 + dz as f32 + 0.5;
                let wz = (1.5 - (cz_f - deposit_pos.z).abs()).max(0.0);
                for dy in -1..2i32 {
                    let cy_f = base_y as f32 + dy as f32 + 0.5;
                    let wy = (1.5 - (cy_f - deposit_pos.y).abs()).max(0.0);
                    for dx in -1..2i32 {
                        let cx_f = base_x as f32 + dx as f32 + 0.5;
                        let wx = (1.5 - (cx_f - deposit_pos.x).abs()).max(0.0);
                        let w = wx * wy * wz;
                        if w < 0.001 { continue; }
                        let cx = base_x + dx;
                        let cy = base_y + dy;
                        let cz = base_z + dz;
                        if Self::in_bounds(cx, cy, cz) {
                            let idx = Self::index(cx as u32, cy as u32, cz as u32);
                            let cell = &mut self.cells[idx];
                            cell.density = (cell.density + total_d * w).min(50.0);
                            cell.color_r = (cell.color_r + total_r * w).min(50.0);
                            cell.color_g = (cell.color_g + total_g * w).min(50.0);
                            cell.color_b = (cell.color_b + total_b * w).min(50.0);
                            self.dirty_slabs[cz as usize] = true;
                        }
                    }
                }
            }

            // Advance oscillation phase
            entity.oscillation_phase += entity.oscillation_freq;
        }
        self.aabb_min = aabb_min.max(glam::Vec3::ZERO);
        self.aabb_max = aabb_max.min(glam::Vec3::splat(FIELD_SIZE as f32));

        self.tick += 1;
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.cells)
    }
}
