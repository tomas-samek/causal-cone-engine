use std::collections::HashMap;
use rayon::prelude::*;

// Diff Field — entities and the pipes between them.
//
// Light propagates through CONNECTIONS between entities, not through any
// grid. The observer is a receptor array (retina.rs) fed by the same kind
// of pipes. Nothing is cached in space; what you see is what arrived.

/// World extent, in cells — bounce bounds, scene placement, AABB clamp.
/// Nothing is allocated per cell; this is a coordinate range, not a buffer.
pub const FIELD_SIZE: u32 = 512;

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

/// Cells of walker travel between cross-link refreshes.
pub const LINK_REFRESH_DIST: f32 = 1.0;

/// Minimum ticks between cross-link refreshes — caps refresh cost at high
/// time-lapse, where the 1-cell distance trigger would otherwise fire every
/// tick. At the default lapse the distance trigger fires every ~10 ticks,
/// so this changes nothing there.
pub const MIN_REFRESH_SPACING: u64 = 5;

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
    /// but never become a retina source (observer can't see them).
    pub is_heat: bool,
    /// Vacuum relay — invisible to observer, just passes light through graph.
    pub is_vacuum: bool,
    /// Atmospheric scatter — how much of the incoming light air molecules
    /// redirect. 0.0 = pure vacuum (space), >0 = atmosphere. Its live value
    /// is what marks a relay as part of the column that follows the AABB.
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
    /// Density-weighted average direction of incoming light (normalized after delivery)
    pub incoming_dir: glam::Vec3,
    /// Debounce: previous incoming density for change detection
    pub prev_incoming_density: f32,
    /// Debounce: consecutive ticks with stable incoming (skip when >= edge_count)
    pub stable_ticks: u8,
    /// Anisotropic gaussian kernel radii for the retina footprint and for
    /// transmittance; ZERO = point source (unit radii).
    pub deposit_radii: glam::Vec3,
    /// Member of the rigid walker group (the dino) — moved by WalkController,
    /// never by per-entity velocity/bounce.
    pub is_walker: bool,
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
            incoming_dir: glam::Vec3::ZERO,
            prev_incoming_density: 0.0,
            stable_ticks: 0,
            deposit_radii: glam::Vec3::ZERO,
            is_walker: false,
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
    pub entities: Vec<Entity>,
    pub tick: u64,
    deliveries: Vec<EdgeDeposit>,
    delivery_dirs: Vec<glam::Vec3>,
    pub aabb_min: glam::Vec3,
    pub aabb_max: glam::Vec3,
    // SoA edge storage — flat contiguous arrays for cache-friendly iteration
    edge_targets: Vec<usize>,
    edge_deposits: Vec<EdgeDeposit>,
    edge_gammas: Vec<f32>,
    /// Per-edge transmittance τ ∈ [0,1], parallel to `edge_gammas`. Radiation
    /// edges are attenuated by the density they pass through; connection edges
    /// (shorter than `link_connect_dist`) stay 1.0. A segment buried in a
    /// metaball core underflows to exactly 0 — a fully opaque pipe.
    edge_atten: Vec<f32>,
    edge_dirs: Vec<glam::Vec3>, // precomputed normalized direction per edge (source → target)
    // Reverse edge index — for each target entity, which edges point to it
    reverse_edge_sources: Vec<usize>, // source entity index
    reverse_edge_k: Vec<usize>,      // SoA edge index into edge_deposits/dirs/etc.
    reverse_start: Vec<u32>,          // per-target-entity start into reverse arrays
    reverse_count: Vec<u32>,          // per-target-entity incoming edge count
    // Per-tick active/visible sets (reused allocation)
    active_set: Vec<bool>,
    visible_set: Vec<bool>,
    /// Parallel vec: consumption_states[i] corresponds to entities[i] for i < entities.len().
    /// Additional entries (i >= entities.len()) are trie-only metadata nodes.
    pub consumption_states: Vec<Option<crate::consumption::ConsumptionState>>,
    /// Whether to enable consumption/learning logs (memory-intensive).
    pub enable_consumption_logs: bool,
    /// Maximum trie depth to render (u16::MAX = full detail).
    pub render_depth_cutoff: u16,
    /// Debug mode: show trie depth as color instead of entity color.
    pub show_trie_depth: bool,
    /// Rigid-body walk controller for the dino group.
    pub walker: crate::walker::WalkController,
    /// Cells of walker travel since the last cross-link refresh (Task 3).
    travel_since_refresh: f32,
    /// Tick of the last cross-link refresh — throttles refresh cadence so the
    /// 1-cell distance trigger can't fire every tick at high time-lapse.
    last_refresh_tick: u64,
    /// Link distances captured at build time, reused by refresh_cross_links.
    link_connect_dist: f32,
    link_radiation_dist: f32,
    /// Live calibration multipliers (keys 1-4): applied on top of the
    /// hardcoded deposit boosts. Baked into constants after calibration.
    pub tune_density: f32,
    pub tune_color: f32,
    /// The observer's retina — receptor array fed by entity pipes.
    pub retina: crate::retina::Retina,
    /// Transmittance strength for occlusion (keys 5/6).
    pub atten_k: f32,
    /// Set by tuning keys so the next tick relinks even if the view is static.
    pub retina_force_relink: bool,
    /// Test plumbing: park tail/jaw/skin animation so a scene can be frozen.
    pub freeze_animation: bool,
    /// Per-entity retina sources, rebuilt each tick (reused allocation).
    sources: Vec<crate::retina::Source>,
}

// ── Metaball source definitions for skeleton + receptor placement ──────────

struct MetaballSource {
    center: glam::Vec3,
    radii: glam::Vec3,
    color: [f32; 3],
    magnitude: f32,
    pass_through: f32,
    group: u16,
}

struct MetaballSample {
    total_field: f32,
    color: [f32; 3],
    pass_through: f32,
    group: u16,
}

fn evaluate_metaball_field(pos: glam::Vec3, balls: &[MetaballSource]) -> MetaballSample {
    let mut total_field = 0.0f32;
    let mut acc_r = 0.0f32;
    let mut acc_g = 0.0f32;
    let mut acc_b = 0.0f32;
    let mut acc_pt = 0.0f32;
    let mut best_group = GROUP_NONE;
    let mut best_c = 0.0f32;

    for b in balls {
        let d = (pos - b.center) / b.radii;
        let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
        if r2 >= 1.0 { continue; }
        let c = 1.0 - r2;
        total_field += c;
        acc_r += b.color[0] * c;
        acc_g += b.color[1] * c;
        acc_b += b.color[2] * c;
        acc_pt += b.pass_through * c;
        if c > best_c {
            best_c = c;
            best_group = b.group;
        }
    }

    if total_field > 0.0 {
        let inv = 1.0 / total_field;
        MetaballSample {
            total_field,
            color: [acc_r * inv, acc_g * inv, acc_b * inv],
            pass_through: acc_pt * inv,
            group: best_group,
        }
    } else {
        MetaballSample {
            total_field: 0.0,
            color: [0.0; 3],
            pass_through: 0.0,
            group: GROUP_NONE,
        }
    }
}

/// Feather window for the gaussian deposit cutoff: 1.0 below exponent 3.0,
/// smoothstep down to exactly 0.0 at 4.0 (where the kernel stops sampling).
/// Without it, boundary cells carry ~exp(-4)×boosted-magnitude — far above
/// the iso threshold — and pop between "deposited" and "decaying" as a
/// gaussian drifts sub-cell, which reads as edge shimmer.
pub(crate) fn cutoff_window(exponent: f32) -> f32 {
    let t = (4.0 - exponent).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Multiply a live-tuning value (keys 1-6), clamped to [1/1024, 1024].
pub fn scale_tune(v: f32, factor: f32) -> f32 {
    (v * factor).clamp(1.0 / 1024.0, 1024.0)
}

impl DiffField {
    pub fn new() -> Self {
        let mut field = Self {
            entities: Vec::new(),
            tick: 0,
            deliveries: Vec::new(),
            delivery_dirs: Vec::new(),
            aabb_min: glam::Vec3::ZERO,
            aabb_max: glam::Vec3::splat(FIELD_SIZE as f32),
            edge_targets: Vec::new(),
            edge_deposits: Vec::new(),
            edge_gammas: Vec::new(),
            edge_atten: Vec::new(),
            edge_dirs: Vec::new(),
            reverse_edge_sources: Vec::new(),
            reverse_edge_k: Vec::new(),
            reverse_start: Vec::new(),
            reverse_count: Vec::new(),
            active_set: Vec::new(),
            visible_set: Vec::new(),
            consumption_states: Vec::new(),
            enable_consumption_logs: false,
            render_depth_cutoff: u16::MAX,
            show_trie_depth: false,
            walker: crate::walker::WalkController::new(),
            travel_since_refresh: 0.0,
            last_refresh_tick: 0,
            link_connect_dist: 0.0,
            link_radiation_dist: 0.0,
            tune_density: 1.0,
            tune_color: 1.0,
            retina: crate::retina::Retina::new(crate::retina::RETINA_W, crate::retina::RETINA_H),
            atten_k: crate::retina::ATTEN_K_DEFAULT,
            retina_force_relink: false,
            freeze_animation: false,
            sources: Vec::new(),
        };

        let sp = field.spawn_demo_scene();

        // Sort entities by spatial locality (z-major cell order) so neighbours in
        // the graph are neighbours in memory during the per-tick edge sweeps.
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
        // Set before build_radiation_links: compute_edge_atten reads
        // link_connect_dist to tell connection edges from radiation edges.
        field.link_connect_dist = connect_dist;
        field.link_radiation_dist = radiation_dist;
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

        // --- Initialize consumption states (AFTER sort + connection build) ---
        field.consumption_states = field.entities.iter().map(|entity| {
            match entity.group {
                GROUP_BODY | GROUP_BELLY | GROUP_TAIL | GROUP_TAIL_TIP |
                GROUP_NECK | GROUP_HEAD | GROUP_JAW | GROUP_MOUTH |
                GROUP_EYE | GROUP_LEG_L | GROUP_FOOT_L | GROUP_LEG_R |
                GROUP_FOOT_R | GROUP_ARM_L | GROUP_ARM_R => {
                    Some(crate::consumption::ConsumptionState::new(0, 0, field.enable_consumption_logs))
                }
                _ => None,
            }
        }).collect();

        // Wire trie topology via BFS from first GROUP_BODY entity along spatial graph edges.
        // Single-child cascade model: each parent gets at most one trie_child.
        if let Some(root_idx) = field.entities.iter().position(|e| e.group == GROUP_BODY) {
            let mut visited = vec![false; field.entities.len()];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(root_idx);
            visited[root_idx] = true;

            while let Some(parent) = queue.pop_front() {
                let edge_start = field.entities[parent].edge_start as usize;
                let edge_count = field.entities[parent].edge_count as usize;
                for k in edge_start..(edge_start + edge_count) {
                    let target = field.edge_targets[k];
                    if visited[target] { continue; }
                    if field.consumption_states[target].is_none() { continue; }
                    visited[target] = true;

                    // Wire trie: parent -> child (first unvisited body neighbor only)
                    if let Some(ref mut ps) = field.consumption_states[parent] {
                        if ps.trie_child.is_none() {
                            ps.trie_child = Some(target);
                        }
                    }
                    let parent_depth = field.consumption_states[parent]
                        .as_ref().map(|p| p.depth).unwrap_or(0);
                    if let Some(ref mut cs) = field.consumption_states[target] {
                        cs.trie_parent = Some(parent);
                        cs.depth = parent_depth + 1;
                    }
                    queue.push_back(target);
                }
            }
        }

        debug_assert_eq!(field.entities.len(), field.consumption_states.len());

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

    /// Build connection edges and detect heat via spatial hash.
    fn build_connections(&mut self, connect_dist: f32) {
        let connect_dist_sq = connect_dist * connect_dist;
        let n = self.entities.len();

        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();

        let mut temp_edges: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Spatial hash for all edges
        let grid = Self::spatial_hash(&positions, connect_dist);
        let mut edge_count = 0u64;
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
                                    edge_count += 1;
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
            "Graph built: {} entities, {} edges (capped at {} per entity)",
            n, edge_count, max_connections
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

        // Darken entities adjacent to heat (depth shading for rock, etc.)
        let mut darkened_count = 0;
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            if temp_edges[i].iter().any(|&t| self.entities[t].is_heat) {
                self.entities[i].color[0] *= 0.2;
                self.entities[i].color[1] *= 0.2;
                self.entities[i].color[2] *= 0.2;
                self.entities[i].deposit_magnitude *= 0.15;
                darkened_count += 1;
            }
        }
        if darkened_count > 0 {
            log::info!("Heat-adjacent darkening: {} entities", darkened_count);
        }

        // Flatten into SoA
        self.flatten_edges(temp_edges);

        log::info!(
            "Edge SoA: {} connection edges, {:.1} MB contiguous",
            self.edge_targets.len(),
            (self.edge_targets.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<EdgeDeposit>())) as f64 / 1_048_576.0
        );
    }

    /// Does this entity's deposit sit at the same world point every tick?
    /// That is what `Source::is_static` promises the spatial hash, so it has to
    /// rule out all three ways `advance_entities` can move a deposit: the
    /// walker's step, per-entity velocity, and the surface oscillation. The
    /// scene's furniture — floor, rock, sun — clears all three, and it is the
    /// bulk of the hash's cell insertions.
    fn is_static_entity(e: &Entity) -> bool {
        !e.is_walker && e.velocity == glam::Vec3::ZERO && e.oscillation_amplitude == 0.0
    }

    /// Static occluder set for edge attenuation: every solid entity at its
    /// current position. Nothing is drawable here — this is only for τ.
    fn occluder_sources(&self) -> Vec<crate::retina::Source> {
        self.entities.iter().map(|e| crate::retina::Source {
            position: e.position,
            radii: e.deposit_radii,
            normal: glam::Vec3::Y,
            opacity: e.deposit_magnitude * if e.deposit_radii != glam::Vec3::ZERO { 40.0 } else { 10.0 },
            density: 0.0,
            color: [0.0; 3],
            drawable: false,
            occluder: !e.is_heat && !e.is_vacuum,
            is_static: Self::is_static_entity(e),
            skin: false, // τ only — nothing here is ever drawn
        }).collect()
    }

    /// τ for every edge longer than link_connect_dist (radiation links);
    /// connection edges stay 1.0. `only_cross` restricts to walker↔world
    /// edges (what a cross-link refresh rebuilt) — internal edges keep the
    /// values carried over by the caller.
    fn compute_edge_atten(&mut self, only_cross: bool) {
        use crate::retina::{segment_transmittance, SpatialHash};
        let t0 = std::time::Instant::now();
        let sources = self.occluder_sources();
        let hash = SpatialHash::build(&sources);
        let cd_sq = self.link_connect_dist * self.link_connect_dist;
        let k = self.atten_k;
        // (source, edge index, target) for every edge
        let mut edges: Vec<(usize, usize, usize)> = Vec::with_capacity(self.edge_targets.len());
        for (i, e) in self.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for kk in start..start + e.edge_count as usize {
                edges.push((i, kk, self.edge_targets[kk]));
            }
        }
        let walker: Vec<bool> = self.entities.iter().map(|e| e.is_walker).collect();
        let atten: Vec<(usize, f32)> = edges.par_iter().filter_map(|&(i, kk, t)| {
            if only_cross && walker[i] == walker[t] { return None; }
            let a = sources[i].position;
            let b = sources[t].position;
            if a.distance_squared(b) < cd_sq { return Some((kk, 1.0)); }
            // No AABB clip: an edge spans two entities, so it is already inside
            // the geometry box — clipping it would only cost the box test.
            Some((kk, segment_transmittance(&sources, &hash, a, b, &[i, t], k, crate::retina::ATTEN_THRESHOLD, None)))
        }).collect();
        let mut attenuated = 0usize;
        for (kk, a) in atten {
            self.edge_atten[kk] = a;
            if a < 0.999 { attenuated += 1; }
        }
        log::info!("Edge attenuation ({}): {} edges < 1, {:.2} ms",
            if only_cross { "cross" } else { "all" }, attenuated, t0.elapsed().as_secs_f64() * 1000.0);
    }

    /// Recompute τ for every edge (public wrapper for live tuning).
    pub fn compute_edge_atten_public(&mut self) {
        self.compute_edge_atten(false);
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

        // Add radiation links via spatial hash — collect candidates per entity, sorted by distance
        let max_radiation: usize = 10; // cap per entity
        let rad_grid = Self::spatial_hash(&positions, max_dist);
        let mut radiation_candidates: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
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
            "Radiation links: {} directed edges (capped at {} per entity)",
            count, max_radiation
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

        self.compute_edge_atten(false);

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
        self.edge_atten = vec![1.0; total];
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

    /// Rebuild only the edges that cross the walker/world boundary.
    ///
    /// Rigid translation keeps dino-internal edges valid forever, and world
    /// entities don't move — so those edge sets (and their in-flight
    /// deposits) are preserved verbatim. Only walker↔world pairs are
    /// re-searched: short-range connection edges (this is how atmosphere
    /// light reaches the receptor shell) and radiation links, whose
    /// transmittance is then recomputed by `compute_edge_atten(true)` at the
    /// walked position (this is how the body's density dims the pipes that
    /// cross it). Heat flags, colors, and consumption state are deliberately
    /// NOT touched.
    fn refresh_cross_links(&mut self) {
        let t0 = std::time::Instant::now();
        let connect_dist = self.link_connect_dist;
        let radiation_dist = self.link_radiation_dist;
        let connect_dist_sq = connect_dist * connect_dist;
        let radiation_dist_sq = radiation_dist * radiation_dist;
        let n = self.entities.len();
        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();
        let walker: Vec<bool> = self.entities.iter().map(|e| e.is_walker).collect();

        // Retain same-side edges (with their deposits carried over positionally —
        // no hashing) and snapshot ONLY the old cross-edge deposits into a small
        // map (a few thousand entries, not all ~111k edges). Re-created cross pairs
        // recover their in-flight light from this map; truly-new pairs start empty.
        let mut temp_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut temp_deposits: Vec<Vec<EdgeDeposit>> = vec![Vec::new(); n];
        // Retained edges keep their τ positionally too — recomputing every
        // edge's transmittance each refresh would cost far more than the
        // cross-edge subset that actually moved.
        let mut temp_atten: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut old_cross_deposits: HashMap<(usize, usize), EdgeDeposit> = HashMap::new();
        // Entities whose cross edge-set changes this refresh (old or new cross
        // endpoints). Their debounce is reset below so new/removed pipes re-propagate
        // immediately instead of the shadow lagging a few steps behind.
        let mut wake = vec![false; n];
        for (i, e) in self.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                let t = self.edge_targets[k];
                if walker[i] == walker[t] {
                    temp_edges[i].push(t);
                    temp_deposits[i].push(self.edge_deposits[k]);
                    temp_atten[i].push(self.edge_atten[k]);
                } else {
                    old_cross_deposits.insert((i, t), self.edge_deposits[k]);
                    wake[i] = true;
                    wake[t] = true;
                }
            }
        }

        // New cross connection edges: walker↔world within connect_dist. Unlike
        // build_connections (which caps at 26 per entity), these are added uncapped;
        // this stays bounded in practice because cross edges are fully dropped and
        // re-searched each refresh (runtime edge count stays ~111k stable).
        // Iterate walkers only (few hundred) against the world spatial hash.
        let grid = Self::spatial_hash(&positions, connect_dist);
        for i in 0..n {
            if !walker[i] { continue; }
            let pos = positions[i];
            let cx = (pos.x / connect_dist).floor() as i32;
            let cy = (pos.y / connect_dist).floor() as i32;
            let cz = (pos.z / connect_dist).floor() as i32;
            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if walker[j] { continue; }
                                if pos.distance_squared(positions[j]) < connect_dist_sq {
                                    temp_edges[i].push(j);
                                    temp_edges[j].push(i);
                                    wake[i] = true;
                                    wake[j] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // New cross radiation links: solid↔solid, capped like
        // build_radiation_links (closest 10 per entity among the new links).
        // No LOS pruning — occlusion is carried by per-edge τ instead.
        let rad_grid = Self::spatial_hash(&positions, radiation_dist);
        let max_radiation: usize = 10;
        let mut cross_rad: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for i in 0..n {
            if !walker[i] { continue; }
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            let pos = positions[i];
            let cx = (pos.x / radiation_dist).floor() as i32;
            let cy = (pos.y / radiation_dist).floor() as i32;
            let cz = (pos.z / radiation_dist).floor() as i32;
            for dz in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        if let Some(bucket) = rad_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if walker[j] { continue; }
                                if self.entities[j].is_vacuum || self.entities[j].is_heat { continue; }
                                let dist_sq = pos.distance_squared(positions[j]);
                                if dist_sq >= connect_dist_sq && dist_sq < radiation_dist_sq {
                                    cross_rad[i].push((j, dist_sq));
                                    cross_rad[j].push((i, dist_sq));
                                }
                            }
                        }
                    }
                }
            }
        }
        for i in 0..n {
            let cands = &mut cross_rad[i];
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cands.truncate(max_radiation);
            for &(target, _) in cands.iter() {
                temp_edges[i].push(target);
                wake[i] = true;
                wake[target] = true;
            }
        }

        // Repack SoA (recomputes edge_dirs from the moved positions),
        // restore retained pipe contents, recompute gammas, rebuild reverse.
        self.flatten_edges(temp_edges);
        // Restore in-flight pipe contents. The first temp_deposits[i].len() edges in
        // each entity's flattened range are the retained same-side edges, in the same
        // order — restore those positionally (no hashing). The remaining edges are the
        // newly searched cross edges: re-created pairs recover their light from the
        // small old-cross map; truly-new pairs stay empty.
        for i in 0..n {
            let start = self.entities[i].edge_start as usize;
            let count = self.entities[i].edge_count as usize;
            let retained = temp_deposits[i].len();
            for (off, k) in (start..start + count).enumerate() {
                if off < retained {
                    self.edge_deposits[k] = temp_deposits[i][off];
                    self.edge_atten[k] = temp_atten[i][off];
                } else if let Some(dep) = old_cross_deposits.get(&(i, self.edge_targets[k])) {
                    self.edge_deposits[k] = *dep;
                }
            }
        }
        for i in 0..n {
            let start = self.entities[i].edge_start as usize;
            let end = start + self.entities[i].edge_count as usize;
            for k in start..end {
                let target = self.edge_targets[k];
                let dist_sq = positions[i].distance_squared(positions[target]);
                self.edge_gammas[k] = 1.0 / dist_sq.max(0.1);
            }
        }
        self.build_reverse_edges();
        self.compute_edge_atten(true);

        // Deterministic debounce wake: an entity that gained (or lost) cross edges
        // this refresh would otherwise stay debounced for a few steps because its
        // fresh incoming edges deliver zero — delaying the shadow. Reset it so it
        // re-propagates on the next tick.
        for i in 0..n {
            if wake[i] {
                self.entities[i].stable_ticks = 0;
            }
        }

        log::debug!(
            "Cross-link refresh at offset {:?}: {} total edges, {:.2} ms",
            self.walker.offset,
            self.edge_targets.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
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

    /// Compute active (pipeline chain) and visible (drawable on the retina) sets
    /// from the observer's view-projection matrix. Only active entities get
    /// Phase 1-2 processing; only visible entities become drawable sources.
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
        // In-frustum non-vacuum solids are visible (drawable retina sources).
        // In-frustum entities (including vacuum) are active (participate in pipeline).
        for i in 0..n {
            let entity = &self.entities[i];

            // Vacuum emitters always active (sun) — they push light through graph
            // but are never drawn themselves.
            if entity.is_vacuum && entity.deposit_magnitude >= 1.0 {
                self.active_set[i] = true;
                self.visible_set[i] = false;
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

    /// Place lightweight receptor entities at the metaball isosurface via BFS.
    /// Receptors catch light from atmosphere radiation links and re-emit it as color.
    /// They add negligible density — skeleton entities handle body opacity.
    fn build_receptor_shell(
        &mut self,
        balls: &[MetaballSource],
        spacing: f32,
        seed: glam::Vec3,
    ) {
        use std::collections::{VecDeque, HashSet};

        let threshold = 1.0f32;

        let seed_grid = (
            (seed.x / spacing).round() as i32,
            (seed.y / spacing).round() as i32,
            (seed.z / spacing).round() as i32,
        );

        const DIRS: [(i32, i32, i32); 6] = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ];

        // BFS flood fill to discover all "inside" cells
        let mut inside: HashMap<(i32, i32, i32), MetaballSample> = HashMap::new();
        let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::new();
        let mut visited: HashSet<(i32, i32, i32)> = HashSet::new();

        let seed_world = glam::Vec3::new(
            seed_grid.0 as f32 * spacing,
            seed_grid.1 as f32 * spacing,
            seed_grid.2 as f32 * spacing,
        );
        let seed_sample = evaluate_metaball_field(seed_world, balls);
        if seed_sample.total_field < threshold {
            log::warn!("Receptor shell: seed outside metaball field");
            return;
        }

        visited.insert(seed_grid);
        queue.push_back(seed_grid);
        inside.insert(seed_grid, seed_sample);

        while let Some(cell) = queue.pop_front() {
            for &(dx, dy, dz) in &DIRS {
                let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                if visited.contains(&neighbor) { continue; }
                visited.insert(neighbor);
                let world_pos = glam::Vec3::new(
                    neighbor.0 as f32 * spacing,
                    neighbor.1 as f32 * spacing,
                    neighbor.2 as f32 * spacing,
                );
                let sample = evaluate_metaball_field(world_pos, balls);
                if sample.total_field >= threshold {
                    inside.insert(neighbor, sample);
                    queue.push_back(neighbor);
                }
            }
        }

        // Surface cells: at least one 6-neighbor is NOT inside
        let mut surface_cells: Vec<(i32, i32, i32)> = Vec::new();
        for &cell in inside.keys() {
            let is_surface = DIRS.iter().any(|&(dx, dy, dz)| {
                !inside.contains_key(&(cell.0 + dx, cell.1 + dy, cell.2 + dz))
            });
            if is_surface {
                surface_cells.push(cell);
            }
        }
        surface_cells.sort(); // deterministic order

        // Create lightweight receptor entities at surface cells
        for &cell in &surface_cells {
            let sample = &inside[&cell];
            let world_pos = glam::Vec3::new(
                cell.0 as f32 * spacing,
                cell.1 as f32 * spacing,
                cell.2 as f32 * spacing,
            );
            let mut e = Entity::new(
                world_pos,
                glam::Vec3::ZERO,
                0.01, // negligible self-emission — light comes via radiation links
                sample.color,
            );
            e.pass_through = sample.pass_through; // absorbs ~97% of incoming light
            e.group = sample.group;
            e.reemit = 0.3; // re-emit absorbed light as colored deposit
            self.entities.push(e);
        }

        log::info!("Receptor shell: {} surface entities at spacing {}",
            surface_cells.len(), spacing);
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
        let sp = 0.3; // tight spacing — surface stays solid after interior becomes heat

        // --- Dino body via BFS flood-fill of metaball field ---
        // Each body part is a metaball source. The combined field produces
        // smooth, seamless geometry — joints blend naturally.
        // BFS grows from body center, creating only surface + subsurface shell.
        let balls = [
            MetaballSource { center: base + glam::Vec3::new(0.0, 5.0, 0.0),     radii: glam::Vec3::new(5.0, 6.0, 8.0),   color: green,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_BODY },
            MetaballSource { center: base + glam::Vec3::new(0.0, 1.0, 0.0),     radii: glam::Vec3::new(4.5, 4.0, 7.0),   color: belly,      magnitude: 0.2,  pass_through: 0.06, group: GROUP_BELLY },
            MetaballSource { center: base + glam::Vec3::new(0.0, 5.5, -12.0),   radii: glam::Vec3::new(2.5, 2.5, 7.0),   color: green,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_TAIL },
            MetaballSource { center: base + glam::Vec3::new(0.0, 5.5, -20.0),   radii: glam::Vec3::new(1.2, 1.2, 4.0),   color: dark_green, magnitude: 0.2,  pass_through: 0.03, group: GROUP_TAIL_TIP },
            MetaballSource { center: base + glam::Vec3::new(0.0, 10.0, 8.0),    radii: glam::Vec3::new(3.0, 5.0, 3.0),   color: green,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_NECK },
            MetaballSource { center: base + glam::Vec3::new(0.0, 16.0, 10.0),   radii: glam::Vec3::new(3.5, 3.0, 5.0),   color: green,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_HEAD },
            MetaballSource { center: base + glam::Vec3::new(0.0, 13.5, 12.0),   radii: glam::Vec3::new(2.5, 1.5, 4.0),   color: dark_green, magnitude: 0.2,  pass_through: 0.03, group: GROUP_JAW },
            MetaballSource { center: base + glam::Vec3::new(0.0, 14.5, 13.0),   radii: glam::Vec3::new(2.0, 0.8, 3.0),   color: mouth,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_MOUTH },
            MetaballSource { center: base + glam::Vec3::new(3.0, 17.0, 12.0),   radii: glam::Vec3::new(0.8, 0.8, 0.8),   color: eye_color,  magnitude: 4.0,  pass_through: 0.1,  group: GROUP_EYE },
            MetaballSource { center: base + glam::Vec3::new(-3.0, 17.0, 12.0),  radii: glam::Vec3::new(0.8, 0.8, 0.8),   color: eye_color,  magnitude: 4.0,  pass_through: 0.1,  group: GROUP_EYE },
            MetaballSource { center: base + glam::Vec3::new(3.0, -3.0, 1.0),    radii: glam::Vec3::new(2.0, 5.0, 2.5),   color: dark_green, magnitude: 0.2,  pass_through: 0.03, group: GROUP_LEG_L },
            MetaballSource { center: base + glam::Vec3::new(3.0, -8.0, 2.0),    radii: glam::Vec3::new(2.5, 1.0, 4.0),   color: dark_green, magnitude: 0.2,  pass_through: 0.03, group: GROUP_FOOT_L },
            MetaballSource { center: base + glam::Vec3::new(-3.0, -3.0, 1.0),   radii: glam::Vec3::new(2.0, 5.0, 2.5),   color: dark_green, magnitude: 0.2,  pass_through: 0.03, group: GROUP_LEG_R },
            MetaballSource { center: base + glam::Vec3::new(-3.0, -8.0, 2.0),   radii: glam::Vec3::new(2.5, 1.0, 4.0),   color: dark_green, magnitude: 0.2,  pass_through: 0.03, group: GROUP_FOOT_R },
            MetaballSource { center: base + glam::Vec3::new(4.5, 6.0, 5.0),     radii: glam::Vec3::new(1.0, 2.5, 1.0),   color: green,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_ARM_R },
            MetaballSource { center: base + glam::Vec3::new(-4.5, 6.0, 5.0),    radii: glam::Vec3::new(1.0, 2.5, 1.0),   color: green,      magnitude: 0.2,  pass_through: 0.03, group: GROUP_ARM_L },
        ];

        // Skeleton placement: one entity per metaball source with wide gaussian deposit.
        // Each skeleton entity's deposit_radii match its metaball shape — overlapping
        // gaussians merge into a continuous density field with no gaps.
        let skeleton_multiplier = 100.0;
        for b in &balls {
            // Eyes already have high magnitude (4.0) + small radii — don't multiply
            let mag = if b.group == GROUP_EYE {
                b.magnitude
            } else {
                b.magnitude * skeleton_multiplier
            };
            let mut e = Entity::new(
                b.center,
                glam::Vec3::ZERO,
                mag,
                b.color,
            );
            e.pass_through = b.pass_through;
            e.group = b.group;
            e.deposit_radii = b.radii;
            self.entities.push(e);
        }

        // Midpoint entities between connected metaballs for smoother joint blending.
        // Each midpoint gets interpolated radii, color, and magnitude.
        let connections: &[(usize, usize)] = &[
            (0, 4),  // body ↔ neck
            (4, 5),  // neck ↔ head
            (0, 2),  // body ↔ tail
            (2, 3),  // tail ↔ tail tip
            (0, 10), // body ↔ leg L
            (10, 11),// leg L ↔ foot L
            (0, 12), // body ↔ leg R
            (12, 13),// leg R ↔ foot R
            (0, 14), // body ↔ arm R
            (0, 15), // body ↔ arm L
            (5, 6),  // head ↔ jaw
        ];
        for &(a, b_idx) in connections {
            let ba = &balls[a];
            let bb = &balls[b_idx];
            let mid_pos = (ba.center + bb.center) * 0.5;
            let mid_radii = (ba.radii + bb.radii) * 0.5;
            let mid_mag = (ba.magnitude + bb.magnitude) * 0.5 * skeleton_multiplier;
            let mid_color = [
                (ba.color[0] + bb.color[0]) * 0.5,
                (ba.color[1] + bb.color[1]) * 0.5,
                (ba.color[2] + bb.color[2]) * 0.5,
            ];
            let mid_pt = (ba.pass_through + bb.pass_through) * 0.5;
            let mut e = Entity::new(mid_pos, glam::Vec3::ZERO, mid_mag, mid_color);
            e.pass_through = mid_pt;
            // Midpoints inherit the source group — except head↔jaw, which must
            // ride the jaw animation: as GROUP_HEAD it would sit still and fill
            // the mouth with a static blob while the jaw dips below it.
            e.group = if (a, b_idx) == (5, 6) { bb.group } else { ba.group };
            e.deposit_radii = mid_radii;
            self.entities.push(e);
        }

        log::info!("Skeleton dino: {} entities ({} metaball + {} midpoint)",
            balls.len() + connections.len(), balls.len(), connections.len());

        // Receptor shell: lightweight surface entities that catch light from atmosphere.
        // Skeleton handles body density; receptors handle lighting via radiation links.
        let receptor_spacing = 1.0;
        self.build_receptor_shell(&balls, receptor_spacing, balls[0].center);

        // Mark the whole dino — skeleton, midpoints, receptor shell — as the
        // rigid walker group. All dino parts use groups BODY..=ARM_R; receptors
        // inherit their group from the metaball they sit on. Heat conversion
        // happens later in build_connections and doesn't change groups.
        for e in self.entities.iter_mut() {
            if (GROUP_BODY..=GROUP_ARM_R).contains(&e.group) {
                e.is_walker = true;
            }
        }

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
                light.is_vacuum = true; // sun emits through the graph, never drawn
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
        // light from sun to dino; their scatter marks them as atmosphere.
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

    /// On-demand diagnostics (key H): retina occupancy/pipe traffic and a
    /// floor-contrast probe. Answers "what reaches the receptors?" and
    /// "does the ground under the dino get lit differently from far away?".
    pub fn dump_field_stats(&self) {
        self.retina.log_stats(&self.sources);

        // Floor contrast probe: tiles under the walker footprint vs far tiles
        let mut wmin = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut wmax = glam::Vec3::ZERO;
        for e in self.entities.iter().filter(|e| e.is_walker) {
            wmin = wmin.min(e.position);
            wmax = wmax.max(e.position);
        }
        let center = (wmin + wmax) * 0.5;
        let (mut under_inc, mut under_n) = (0.0f64, 0u32);
        let (mut far_inc, mut far_n) = (0.0f64, 0u32);
        for e in self.entities.iter().filter(|e| e.group == GROUP_FLOOR) {
            let under = e.position.x >= wmin.x && e.position.x <= wmax.x
                && e.position.z >= wmin.z && e.position.z <= wmax.z;
            let dx = e.position.x - center.x;
            let dz = e.position.z - center.z;
            if under {
                under_inc += e.incoming.density as f64;
                under_n += 1;
            } else if dx * dx + dz * dz > 400.0 {
                far_inc += e.incoming.density as f64;
                far_n += 1;
            }
        }
        log::info!(
            "Floor probe: under dino n={} avg incoming={:.3} | far (>20 cells) n={} avg incoming={:.3}",
            under_n, under_inc / under_n.max(1) as f64,
            far_n, far_inc / far_n.max(1) as f64,
        );
    }

    /// Light transport for one sim step: cross-link refresh, active set,
    /// atmosphere modulation, Phase 1 deliver, consumption, Phase 2 push.
    /// Returns true if the cross-link refresh ran (the retina relinks then).
    fn transport(&mut self, view_proj: glam::Mat4) -> bool {
        let mut refreshed = false;
        // Refresh walker↔world light links once the dino has drifted a cell
        // from where they were last built — shadow and lighting follow.
        if self.travel_since_refresh >= LINK_REFRESH_DIST
            && self.tick.saturating_sub(self.last_refresh_tick) >= MIN_REFRESH_SPACING
        {
            self.travel_since_refresh = 0.0;
            self.last_refresh_tick = self.tick;
            self.refresh_cross_links();
            refreshed = true;
        }

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

        // --- Consumption-Transformation Pass ---
        // After incoming is accumulated (Phase 1), before entities push outward (Phase 2).
        // Each body entity's incoming deposit is quantized to a token and routed through
        // the consumption trie: consume matching tokens, cascade rejects to children.
        {
            use crate::consumption::{DepositToken, ConsumptionState, cascade_process};
            let density_scale = 1.0_f32;
            let entity_count = self.entities.len();

            // cascade_process expects &mut Vec<ConsumptionState> (flat), but our field
            // stores Vec<Option<ConsumptionState>>. Temporarily extract into a flat vec,
            // tracking which indices were originally None so we can restore them.
            let opt_len = self.consumption_states.len();
            let mut none_slots: Vec<bool> = Vec::with_capacity(opt_len);
            let mut flat: Vec<ConsumptionState> = Vec::with_capacity(opt_len);
            for opt in self.consumption_states.drain(..) {
                match opt {
                    Some(cs) => { none_slots.push(false); flat.push(cs); }
                    None => {
                        none_slots.push(true);
                        // Placeholder; will never be accessed since we skip None-origin indices.
                        flat.push(ConsumptionState::new(0, 0, false));
                    }
                }
            }

            for i in 0..entity_count {
                if i >= flat.len() { break; }
                if none_slots[i] { continue; }
                let incoming = &self.entities[i].incoming;
                if incoming.density < 0.001 { continue; }

                let token = DepositToken::from_deposit(
                    incoming.density, incoming.r, incoming.g, incoming.b,
                    density_scale,
                );

                cascade_process(
                    &mut flat,
                    i,
                    self.tick,
                    token,
                );
            }

            // Restore: original slots get Some/None back; any new entries pushed by
            // cascade_process (trie-only children beyond entity_count) are wrapped in Some.
            self.consumption_states.reserve(flat.len());
            for (idx, cs) in flat.into_iter().enumerate() {
                if idx < none_slots.len() && none_slots[idx] {
                    self.consumption_states.push(None);
                } else {
                    self.consumption_states.push(Some(cs));
                }
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
        // The eye position, recovered from the view-projection matrix. (The
        // old `inv_vp.col(3).xyz` is the unprojected clip origin, not the eye.)
        let observer_pos = crate::retina::eye_from_view_proj(view_proj);
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
        let edge_atten = &self.edge_atten;
        let edge_dir_arr = &self.edge_dirs;
        let active = &self.active_set;
        let consumption_states = &self.consumption_states;
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
                let mut w = edge_gammas[k] * distance_factors[idx] * edge_atten[k];
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

            // Consumption blending: consumed portion → entity's own color (Same), rest passes through
            let (pass_r, pass_g, pass_b) = if idx < consumption_states.len() {
                if let Some(ref state) = consumption_states[idx] {
                    if !state.learning && state.consumed > 0 {
                        let total_seen = (state.consumed + state.rejected) as f32;
                        let cr = state.consumed as f32 / total_seen.max(1.0);
                        (entity.color[0] * cr + pass_r * (1.0 - cr),
                         entity.color[1] * cr + pass_g * (1.0 - cr),
                         entity.color[2] * cr + pass_b * (1.0 - cr))
                    } else { (pass_r, pass_g, pass_b) }
                } else { (pass_r, pass_g, pass_b) }
            } else { (pass_r, pass_g, pass_b) };

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

        refreshed
    }

    /// Move and animate entities, track the AABB, and build this tick's
    /// retina sources.
    ///
    /// Two deliberate asymmetries: every solid advances its
    /// `oscillation_phase`, in frustum or not, so the skin texture never
    /// jumps when an entity comes back into view; and the AABB spans every
    /// solid, including depth-culled ones, so the atmosphere column and the
    /// retina's relink trigger stay tied to the geometry, not to the cutoff.
    fn advance_entities(&mut self) {
        use crate::retina::Source;
        let walk_delta = self.walker.step();
        self.travel_since_refresh += walk_delta.length();
        let walk_offset_z = self.walker.offset.z;
        let tune_density = self.tune_density;
        let tune_color = self.tune_color;
        let freeze = self.freeze_animation;
        let tick = self.tick;

        let n = self.entities.len();
        self.sources.clear();
        self.sources.resize(n, Source {
            position: glam::Vec3::ZERO, radii: glam::Vec3::ZERO, normal: glam::Vec3::Y,
            opacity: 0.0, density: 0.0, color: [0.0; 3], drawable: false, occluder: false,
            is_static: true, skin: false,
        });

        let mut aabb_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut aabb_max = glam::Vec3::splat(0.0);
        for ent_idx in 0..n {
            // Read everything that lives outside `entities` first, so the
            // mutable entity borrow below does not conflict.
            let visible = self.visible_set[ent_idx];
            let show_depth = self.show_trie_depth;
            let (depth_cull, mass_boost, depth_col) =
                match self.consumption_states.get(ent_idx).and_then(|s| s.as_ref()) {
                    Some(state) => (
                        state.depth > self.render_depth_cutoff,
                        if !state.learning && state.consumed > 0 {
                            1.0 + (state.consumed as f32).ln().max(0.0) * 0.05
                        } else { 1.0 },
                        Some(crate::consumption::depth_color(state.depth)),
                    ),
                    None => (false, 1.0f32, None),
                };

            let entity = &mut self.entities[ent_idx];
            if entity.is_walker { entity.position += walk_delta; }
            entity.position += entity.velocity;
            for i in 0..3 {
                if entity.position[i] < 1.0 || entity.position[i] >= (FIELD_SIZE - 1) as f32 {
                    entity.velocity[i] *= -1.0;
                    entity.position[i] = entity.position[i].clamp(1.0, (FIELD_SIZE - 2) as f32);
                }
            }
            if entity.is_heat || entity.is_vacuum { continue; }

            let extent = if entity.deposit_radii != glam::Vec3::ZERO { entity.deposit_radii * 2.0 } else { glam::Vec3::splat(1.0) };
            aabb_min = aabb_min.min(entity.position - extent);
            aabb_max = aabb_max.max(entity.position + extent);

            // Animated deposit position (skin oscillation, tail wag, jaw).
            let mut deposit_pos = entity.position;
            if !freeze && entity.oscillation_amplitude > 0.0 {
                deposit_pos += entity.surface_normal * entity.oscillation_phase.sin() * entity.oscillation_amplitude;
            }
            if !freeze && (entity.group == GROUP_TAIL || entity.group == GROUP_TAIL_TIP) {
                let time = tick as f32 / 30.0;
                let frequency = std::f32::consts::PI;
                let center_z = FIELD_SIZE as f32 / 2.0 + walk_offset_z;
                let z_frac = ((center_z - entity.position.z) / 24.0).clamp(0.0, 1.0);
                let amplitude = 3.0 * z_frac;
                let phase = time * frequency + z_frac * 2.0;
                deposit_pos.x += amplitude * phase.sin();
            }
            if !freeze && (entity.group == GROUP_JAW || entity.group == GROUP_MOUTH) {
                let time = tick as f32 / 30.0;
                let frequency = std::f32::consts::PI * 0.5;
                let center = FIELD_SIZE as f32 / 2.0;
                let pivot_z = center + 8.0 + walk_offset_z;
                let z_frac = ((entity.position.z - pivot_z) / 8.0).clamp(0.0, 1.0);
                let open_amount = (time * frequency).sin().abs();
                deposit_pos.y -= z_frac * 2.5 * open_amount;
            }
            if !freeze { entity.oscillation_phase += entity.oscillation_freq; }

            let use_gaussian = entity.deposit_radii != glam::Vec3::ZERO;
            let (density_boost, color_boost) = if use_gaussian {
                (40.0 * tune_density, 10.0 * tune_color)
            } else {
                (10.0 * tune_density, 10.0 * tune_color)
            };
            let static_boost = if use_gaussian { 40.0 } else { 10.0 };

            // Visible? (frustum + trie-depth cutoff)
            let drawable = visible && !depth_cull;

            // Consumption mass boost: entities that consume more deposit denser
            let mag = entity.deposit_magnitude * mass_boost;
            let absorbed = 1.0 - entity.pass_through;
            let entity_color = if show_depth {
                depth_col.unwrap_or([0.3, 0.3, 0.3])
            } else { entity.color };
            let total_r = (entity_color[0] * mag + entity.incoming.r * absorbed * entity_color[0] + entity.reemit_r) * color_boost;
            let total_g = (entity_color[1] * mag + entity.incoming.g * absorbed * entity_color[1] + entity.reemit_g) * color_boost;
            let total_b = (entity_color[2] * mag + entity.incoming.b * absorbed * entity_color[2] + entity.reemit_b) * color_boost;
            let total_d = (mag + entity.incoming.density * absorbed) * density_boost;

            self.sources[ent_idx] = Source {
                position: deposit_pos,
                radii: entity.deposit_radii,
                normal: if entity.surface_normal.length_squared() > 0.0 { entity.surface_normal } else { glam::Vec3::Y },
                opacity: entity.deposit_magnitude * static_boost,
                density: total_d,
                color: [total_r, total_g, total_b],
                drawable,
                occluder: true,
                is_static: Self::is_static_entity(entity),
                // Every walker group is dino — body, limbs, eyes, mouth. The
                // shader reads this back as "give it reptile scales".
                skin: entity.is_walker,
            };
        }
        self.aabb_min = aabb_min.max(glam::Vec3::ZERO);
        self.aabb_max = aabb_max.min(glam::Vec3::splat(FIELD_SIZE as f32));
    }

    /// Run one simulation tick — push-driven pipe propagation, then arrival
    /// at the observer's retina.
    pub fn tick(&mut self, view_proj: glam::Mat4) {
        let refreshed = self.transport(view_proj);
        self.advance_entities();
        let force = refreshed || self.retina_force_relink;
        self.retina_force_relink = false;
        // Borrow split: sources and retina are separate fields.
        let sources = std::mem::take(&mut self.sources);
        self.retina.tick(&sources, view_proj, self.aabb_min, self.aabb_max, force, self.atten_k);
        self.sources = sources;

        if self.tick % 300 == 0 && self.tick > 0 {
            self.log_trie_diagnostics();
        }
        self.tick += 1;
    }

    /// Periodic trie census: how many states, at what depths, and the
    /// consume/reject totals.
    fn log_trie_diagnostics(&self) {
        let mut depth_counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
        let mut total_consumed = 0u64;
        let mut total_rejected = 0u64;
        for state in self.consumption_states.iter().flatten() {
            *depth_counts.entry(state.depth).or_insert(0) += 1;
            total_consumed += state.consumed;
            total_rejected += state.rejected;
        }
        let n_states = self.consumption_states.iter().filter(|s| s.is_some()).count();
        log::info!("Trie @ t={}: states={}, depths={:?}, consumed={}, rejected={}",
            self.tick, n_states, depth_counts, total_consumed, total_rejected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_view_proj() -> glam::Mat4 {
        // Same pose as Observer::new(): in front of the dino, looking at it.
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 500.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(256.0, 256.0, 310.0),
            glam::Vec3::new(256.0, 256.0, 256.0),
            glam::Vec3::Y,
        );
        proj * view
    }

    #[test]
    fn cutoff_window_is_one_inside_and_zero_at_cutoff() {
        assert_eq!(cutoff_window(0.0), 1.0);
        assert_eq!(cutoff_window(3.0), 1.0);
        assert_eq!(cutoff_window(4.0), 0.0);
        assert!((cutoff_window(3.5) - 0.5).abs() < 1e-6); // smoothstep midpoint
    }

    #[test]
    fn cutoff_window_fades_monotonically_and_continuously() {
        // Monotone decreasing across the feather band
        let mut prev = cutoff_window(3.0);
        for i in 1..=20 {
            let e = 3.0 + i as f32 * 0.05;
            let cur = cutoff_window(e);
            assert!(cur <= prev, "window not monotone at exponent {}", e);
            prev = cur;
        }
        // Continuous at the cutoff: value just inside 4.0 is already tiny,
        // so a cell crossing the boundary cannot pop
        assert!(cutoff_window(3.999) < 1e-4);
    }

    #[test]
    fn scale_tune_stays_in_range() {
        assert_eq!(scale_tune(1.0, 2.0), 2.0);
        assert_eq!(scale_tune(1.0, 0.5), 0.5);
        assert_eq!(scale_tune(1024.0, 2.0), 1024.0);
        let lo = scale_tune(1.0 / 1024.0, 0.5);
        assert!((lo - 1.0 / 1024.0).abs() < 1e-9);
    }

    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn walker_group_moves_rigidly() {
        let mut field = DiffField::new();
        let walkers: Vec<usize> = field.entities.iter().enumerate()
            .filter(|(_, e)| e.is_walker)
            .map(|(i, _)| i)
            .collect();
        assert!(walkers.len() > 100, "expected skeleton + receptors, got {}", walkers.len());
        // Sun, floor, rock, vacuum must NOT walk
        assert!(field.entities.iter().all(|e| {
            !(e.is_walker && matches!(e.group, GROUP_SUN | GROUP_FLOOR | GROUP_VACUUM | GROUP_ROCK))
        }));

        let start: Vec<glam::Vec3> = walkers.iter().map(|&i| field.entities[i].position).collect();
        let vp = test_view_proj();
        for _ in 0..30 {
            field.tick(vp);
        }
        let expected = field.walker.offset;
        assert!(expected.length() > 0.5, "walker barely moved: {:?}", expected);

        // Every walker entity moved by exactly the shared offset
        for (k, &i) in walkers.iter().enumerate() {
            let moved = field.entities[i].position - start[k];
            assert!((moved - expected).length() < 1e-3,
                "entity {} moved {:?}, expected {:?}", i, moved, expected);
        }
        // Rigidity: pairwise distances preserved (spot-check first 10)
        let m = walkers.len().min(10);
        for a in 0..m {
            for b in (a + 1)..m {
                let d0 = start[a].distance(start[b]);
                let d1 = field.entities[walkers[a]].position
                    .distance(field.entities[walkers[b]].position);
                assert!((d0 - d1).abs() < 1e-3, "pair ({},{}) drifted: {} vs {}", a, b, d0, d1);
            }
        }
    }

    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn cross_links_follow_walker() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        // 90 steps ≈ 9 cells of travel (with one turnaround at +6) → several refreshes
        for _ in 0..90 {
            field.tick(vp);
        }

        // Every walker↔world edge must respect the link distances plus at most
        // LINK_REFRESH_DIST of staleness — links may never lag the walk.
        let max_len = field.link_radiation_dist + LINK_REFRESH_DIST + 0.2;
        let mut cross_edges = 0u32;
        let mut foot_to_floor = 0u32;
        for (i, e) in field.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                let t = field.edge_targets[k];
                if e.is_walker == field.entities[t].is_walker { continue; }
                cross_edges += 1;
                let d = e.position.distance(field.entities[t].position);
                assert!(d <= max_len, "stale cross edge {}→{}: {} cells", i, t, d);
                if e.is_walker
                    && matches!(e.group, GROUP_FOOT_L | GROUP_FOOT_R)
                    && field.entities[t].group == GROUP_FLOOR
                {
                    foot_to_floor += 1;
                }
            }
        }
        assert!(cross_edges > 0, "walker is isolated from the world graph");
        assert!(foot_to_floor > 0, "no foot→floor links at the new position — shadow can't follow");
    }

    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn retina_sees_the_dino_and_settles() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        for _ in 0..60 {
            field.tick(vp);
        }
        let r = &field.retina;
        let above = r.receptors.iter().filter(|x| x.density >= crate::retina::RETINA_ISO).count();
        assert!(above > 500, "only {} receptors above iso — dino not on the retina", above);
        assert!(r.receptors.iter().all(|x| x.density.is_finite() && x.color.iter().all(|c| c.is_finite())),
            "non-finite receptor");
        // Walker-group sources are drawable and linked
        let linked_walkers = field.entities.iter().enumerate()
            .filter(|(i, e)| e.is_walker && r.pipes_of(*i).count() > 0).count();
        assert!(linked_walkers > 100, "only {} walker entities have pipes", linked_walkers);
        // Freeze the world: no motion, no animation. Lighting keeps
        // converging (consumption learning nudges mass boosts), so demand
        // "almost silent", not zero — the unit tests prove exact zero.
        field.walker.speed_c = 0.0;
        for e in &mut field.entities { e.velocity = glam::Vec3::ZERO; }
        field.freeze_animation = true;
        // 25 ticks, not 10: now that the dino is no longer occluded by itself
        // its skin actually carries light, so the lighting has ~100 more
        // contributing entities to converge. It reaches 0.6% by tick 20 and
        // stays there; 10 ticks catches it mid-descent at ~1.6%.
        for _ in 0..25 { field.tick(vp); }
        let s = field.retina.stats;
        assert!(s.pipes_sent * 100 < s.pipes_total,
            "frozen scene still sends {} of {} pipes", s.pipes_sent, s.pipes_total);
    }

    /// The dino is made of a receptor shell buried in its own skeleton
    /// metaballs. If an entity's τ is measured against the absolute density
    /// floor, every shell entity is occluded by the body it belongs to and the
    /// dino renders as a hole. The walker group has to *arrive*.
    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn walker_entities_are_not_occluded_by_their_own_body() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        for _ in 0..30 {
            field.tick(vp);
        }
        let r = &field.retina;
        let taus: Vec<f32> = field.entities.iter().enumerate()
            .filter(|(i, e)| e.is_walker && r.pipes_of(*i).count() > 0)
            .map(|(i, _)| r.transmittance(i))
            .collect();
        assert!(taus.len() >= 100, "only {} walker entities linked", taus.len());
        let mean = taus.iter().sum::<f32>() / taus.len() as f32;
        let bright = taus.iter().filter(|&&t| t >= 0.5).count();
        eprintln!("walker τ over {} linked entities: mean {:.3}, {} ≥ 0.5 ({:.0}%)",
            taus.len(), mean, bright, 100.0 * bright as f64 / taus.len() as f64);
        assert!(mean >= 0.3, "mean walker τ {:.3} — the dino occludes itself", mean);
        // The shell is a closed surface, so at most about half of it can ever
        // face the eye — the other half is behind the body and *must* stay
        // dark. The measured split is 51/49, so demand a clear near half.
        assert!(bright * 100 >= taus.len() * 45,
            "only {} of {} walker entities have τ ≥ 0.5", bright, taus.len());
        // And the half nearest the eye is the dino's visible skin: it has to
        // arrive nearly unattenuated, not merely "on average not zero".
        let mut by_depth: Vec<(f32, f32)> = field.entities.iter().enumerate()
            .filter(|(i, e)| e.is_walker && r.pipes_of(*i).count() > 0)
            .map(|(i, _)| (r.depth_of(i), r.transmittance(i)))
            .collect();
        by_depth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let half = by_depth.len() / 2;
        let near = by_depth[..half].iter().map(|x| x.1).sum::<f32>() / half as f32;
        eprintln!("near half mean τ = {:.3}", near);
        assert!(near >= 0.8, "the dino's front skin is dimmed: near-half mean τ {:.3}", near);
    }

    /// A tick must fit in a frame. The bounds are several times what a tick
    /// actually costs, so this fires on a real regression, not on a slow
    /// machine. Three costs on three schedules, so three bounds: a plain tick,
    /// a tick that relinks the retina, and a tick that also rebuilds the
    /// walker's cross links (link graph + edge attenuation — not the retina's
    /// work, but it lands in the same frame).
    ///
    /// The steady-state phase has to freeze the world to find one. While the
    /// dino walks it projects ~0.17 receptors per tick, past `RELINK_SHIFT`, so
    /// every tick relinks *by design* — that is what keeps the animation from
    /// being frozen between relinks. A tick with no relink in it only exists
    /// once the scene stops moving.
    ///
    /// The whole ignored suite runs its tests in parallel over one rayon pool,
    /// so a sample here can be two thirds contention. Cost is what the fastest
    /// sample shows; the rest is other tests. So the budgets are judged on the
    /// best of several samples, and every sample is printed.
    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn tick_stays_inside_the_frame_budget() {
        const STEADY_BUDGET_MS: f64 = 40.0;
        const WORST_STEADY_BUDGET_MS: f64 = 150.0;
        const RELINK_BUDGET_MS: f64 = 80.0;
        const REFRESH_BUDGET_MS: f64 = 150.0;
        /// A relinking walking tick, which is every tick while the dino moves.
        /// Uncontended it lands near 22 ms on the reference machine; 60 leaves
        /// the room a shared rayon pool takes. Judged, like every other budget
        /// here, on the fastest sample — see the doc comment.
        const WALKING_BUDGET_MS: f64 = 60.0;
        /// The first two relinks grow every SoA array (pipes, weights,
        /// pipe_last — ~55 MB together) from nothing and build the static half
        /// of the spatial hash, so they cost roughly twice a settled relink.
        /// That is a process-lifetime cost, not a frame cost, so they are
        /// printed and held to the loose ceiling but never taken as the best.
        const WALKING_WARMUP_TICKS: usize = 2;

        let mut field = DiffField::new();
        let vp = test_view_proj();

        // Phase 1 — the walking scene. Every tick here relinks; one also
        // refreshes cross links, and that sample is measuring the refresh.
        let mut best_walking = f64::MAX;
        for i in 0..12 {
            let relinks_before = field.retina.stats.relinks;
            let refresh_before = field.last_refresh_tick;
            let t0 = std::time::Instant::now();
            field.tick(vp);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let s = field.retina.stats;
            let refreshed = field.last_refresh_tick != refresh_before;
            eprintln!("walking tick {:2}: {:7.2} ms  ({}{}hash {:.2} + relink {:.2} ms [footprints {:.2}, pipes {:.2}, τ {:.2}, rest {:.2}], {} of {} pipes sent)",
                i, ms,
                if s.relinks > relinks_before { "RELINK, " } else { "" },
                if refreshed { "CROSS-LINK REFRESH, " } else { "" },
                s.hash_build_ms, s.relink_ms, s.relink_footprint_ms, s.relink_pipes_ms, s.relink_tau_ms,
                s.relink_ms - s.relink_footprint_ms - s.relink_pipes_ms - s.relink_tau_ms,
                s.pipes_sent, s.pipes_total);
            assert!(ms < REFRESH_BUDGET_MS,
                "walking tick {} {:.2} ms exceeds the {:.0} ms ceiling", i, ms, REFRESH_BUDGET_MS);
            // A cross-link refresh lands on one of these ticks and dominates
            // it: that sample is measuring the refresh, not a walking tick.
            if !refreshed && i >= WALKING_WARMUP_TICKS { best_walking = best_walking.min(ms); }
        }
        eprintln!("walking: best settled tick {:.2} ms", best_walking);
        assert!(best_walking < WALKING_BUDGET_MS,
            "fastest walking tick {:.2} ms exceeds the {:.0} ms budget", best_walking, WALKING_BUDGET_MS);

        // Phase 2 — steady state. Stop the world the way
        // `retina_sees_the_dino_and_settles` does, then let it settle before
        // measuring: the first ticks after the freeze still carry the last
        // relink and the lighting's tail.
        field.walker.speed_c = 0.0;
        for e in &mut field.entities { e.velocity = glam::Vec3::ZERO; }
        field.freeze_animation = true;
        for _ in 0..4 { field.tick(vp); }

        let mut worst = 0.0f64;
        let mut best = f64::MAX;
        let mut sum = 0.0f64;
        let mut steady = 0u32;
        for i in 0..12 {
            let relinks_before = field.retina.stats.relinks;
            let refresh_before = field.last_refresh_tick;
            let t0 = std::time::Instant::now();
            field.tick(vp);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let s = field.retina.stats;
            let relinked = s.relinks > relinks_before;
            let refreshed = field.last_refresh_tick != refresh_before;
            eprintln!("steady tick {:2}: {:7.2} ms  ({}{}relink {:.2} ms, {} of {} pipes sent)",
                i, ms,
                if relinked { "RELINK, " } else { "" },
                if refreshed { "CROSS-LINK REFRESH, " } else { "" },
                s.relink_ms, s.pipes_sent, s.pipes_total);
            if relinked || refreshed {
                assert!(ms < REFRESH_BUDGET_MS,
                    "steady tick {} (relink/refresh) {:.2} ms exceeds the {:.0} ms budget", i, ms, REFRESH_BUDGET_MS);
                continue;
            }
            worst = worst.max(ms);
            best = best.min(ms);
            sum += ms;
            steady += 1;
        }
        assert!(steady >= 8, "only {} of 12 frozen ticks were steady — nothing was measured", steady);
        eprintln!("steady state over {} ticks: best {:.2} ms, mean {:.2} ms, worst {:.2} ms",
            steady, best, sum / steady as f64, worst);
        assert!(best < STEADY_BUDGET_MS,
            "fastest steady-state tick {:.2} ms exceeds the {:.0} ms budget", best, STEADY_BUDGET_MS);
        // Contention inflates a sample, but not without bound: a steady tick
        // that took longer than a full relink is a regression, not noise.
        assert!(worst < WORST_STEADY_BUDGET_MS,
            "slowest steady-state tick {:.2} ms exceeds the {:.0} ms budget", worst, WORST_STEADY_BUDGET_MS);

        // Phase 3 — the relink itself, forced so it lands on a known tick.
        let mut best_relink = f64::MAX;
        for _ in 0..6 {
            let relinks_before = field.retina.stats.relinks;
            let refresh_before = field.last_refresh_tick;
            field.retina_force_relink = true;
            let t0 = std::time::Instant::now();
            field.tick(vp);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            assert!(field.retina.stats.relinks > relinks_before, "forced relink did not fire");
            let s = field.retina.stats;
            eprintln!("forced relink tick: {:.2} ms (hash {:.2} + retina relink {:.2} ms [footprints {:.2}, pipes {:.2}, τ {:.2}, rest {:.2}])",
                ms, s.hash_build_ms, s.relink_ms, s.relink_footprint_ms, s.relink_pipes_ms, s.relink_tau_ms,
                s.relink_ms - s.relink_footprint_ms - s.relink_pipes_ms - s.relink_tau_ms);
            // A cross-link refresh may land on this tick too — that sample is
            // measuring something else, so it only has to clear that budget.
            if field.last_refresh_tick != refresh_before {
                assert!(ms < REFRESH_BUDGET_MS,
                    "forced-relink + refresh tick {:.2} ms exceeds the {:.0} ms budget", ms, REFRESH_BUDGET_MS);
                continue;
            }
            best_relink = best_relink.min(ms);
        }
        assert!(best_relink < f64::MAX,
            "no forced-relink sample avoided the refresh branch — the relink cost was never measured");
        assert!(best_relink < RELINK_BUDGET_MS,
            "fastest forced-relink tick {:.2} ms exceeds the {:.0} ms budget", best_relink, RELINK_BUDGET_MS);
    }

    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn long_edges_are_attenuated_short_edges_are_not() {
        let field = DiffField::new();
        let cd_sq = field.link_connect_dist * field.link_connect_dist;
        let mut long_below_one = 0u32;
        let mut long_total = 0u32;
        for (i, e) in field.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for k in start..start + e.edge_count as usize {
                let t = field.edge_targets[k];
                let d_sq = e.position.distance_squared(field.entities[t].position);
                let a = field.edge_atten[k];
                // τ ∈ [0,1]: an edge buried in the metaball core integrates a
                // density large enough that exp(−k·∫ρ) underflows f32 to
                // exactly 0 — a fully opaque path, not a bug.
                assert!(a >= 0.0 && a <= 1.0, "edge {}→{} atten {}", i, t, a);
                if d_sq < cd_sq {
                    assert_eq!(a, 1.0, "connection edge {}→{} attenuated", i, t);
                } else {
                    long_total += 1;
                    if a < 0.999 { long_below_one += 1; }
                }
            }
        }
        assert!(long_total > 1000, "expected many radiation edges, got {}", long_total);
        assert!(long_below_one > 0, "no radiation edge is attenuated — occluders ignored");
        assert!(long_below_one < long_total, "every radiation edge is attenuated — self-shadowing");
    }

    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn attenuation_shadows_follow_walker() {
        let mut field = DiffField::new();
        let vp = test_view_proj();
        // Walk far enough to trigger several cross-link refreshes. Every
        // walker↔world edge here was created by refresh_cross_links AFTER the
        // dino moved, and flatten_edges reset its τ to 1.0 — nothing carries
        // a build-time value into this set. So a cross edge with τ < 1 can
        // only come from compute_edge_atten(true) running at the walked
        // position: delete that call and every assertion below on `cross`
        // sees a uniform 1.0.
        for _ in 0..100 { field.tick(vp); }
        let mut wmin = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut wmax = glam::Vec3::ZERO;
        for e in field.entities.iter().filter(|e| e.is_walker) {
            wmin = wmin.min(e.position);
            wmax = wmax.max(e.position);
        }
        let center = (wmin + wmax) * 0.5;
        let long_dist_sq = field.link_connect_dist * field.link_connect_dist;

        // Radiation edges that cross the walker/world boundary — the set the
        // refresh rebuilt — versus far floor↔floor edges nowhere near the body.
        let (mut cross_sum, mut cross_n, mut cross_dimmed) = (0.0f64, 0u32, 0u32);
        let (mut far_sum, mut far_n) = (0.0f64, 0u32);
        let mut min_cross = 1.0f32;
        for (i, e) in field.entities.iter().enumerate() {
            let start = e.edge_start as usize;
            for kk in start..start + e.edge_count as usize {
                let t = field.edge_targets[kk];
                let a = e.position;
                let b = field.entities[t].position;
                if a.distance_squared(b) <= long_dist_sq { continue; }
                let tau = field.edge_atten[kk];
                if field.entities[i].is_walker != field.entities[t].is_walker {
                    cross_sum += tau as f64;
                    cross_n += 1;
                    if tau < 0.5 { cross_dimmed += 1; }
                    min_cross = min_cross.min(tau);
                }
                let mid = (a + b) * 0.5;
                let dx = mid.x - center.x;
                let dz = mid.z - center.z;
                if e.group == GROUP_FLOOR
                    && field.entities[t].group == GROUP_FLOOR
                    && dx * dx + dz * dz > 625.0
                {
                    far_sum += tau as f64;
                    far_n += 1;
                }
            }
        }
        assert!(cross_n > 0, "no walker↔world radiation edges after the walk");
        assert!(far_n > 0, "no far floor edges found");
        // The discriminating assertion: τ for the refreshed cross set was
        // computed against the body's density at its NEW position.
        assert!(
            cross_dimmed > 0 && min_cross < 0.5,
            "no walker↔world pipe is dimmed at the walked position: {} of {} below 0.5, min={}",
            cross_dimmed, cross_n, min_cross
        );
        let far_avg = far_sum / far_n as f64;
        let cross_avg = cross_sum / cross_n as f64;
        assert!(
            far_avg > cross_avg,
            "far floor edges ({:.4}) not clearer than walker↔world edges ({:.4})",
            far_avg, cross_avg
        );
    }

    /// Close-range probe: the eye parked inside the walker's own bounding box,
    /// the pose the user's screenshot was taken from. Prints `pipes_total` and
    /// the walking tick time there, plus the widest footprint at the pose —
    /// the diagnostic that exposed the near-plane needle. Not a cost budget:
    /// the numbers are the measurement, and the one assertion pins only that
    /// the near-plane cut is still in force.
    #[test]
    #[ignore] // builds the full demo scene (slow) — run: cargo test --release -- --ignored
    fn close_range_probe_reports_pipe_count() {
        let mut field = DiffField::new();
        let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
        for e in field.entities.iter().filter(|e| e.is_walker) {
            lo = lo.min(e.position);
            hi = hi.max(e.position);
        }
        let centre = (lo + hi) * 0.5;
        let eye = glam::Vec3::new(centre.x + 2.0, lo.y + 3.0, hi.z + 6.0);
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 500.0);
        let vp = proj * glam::Mat4::look_at_rh(eye, centre, glam::Vec3::Y);
        eprintln!("probe pose: eye {:?} → walker centre {:?} (walker aabb {:?}..{:?})", eye, centre, lo, hi);

        let mut best = f64::MAX;
        let mut pipes = 0usize;
        let mut widest = (0usize, 0u32, 0u32); // entity, columns, rows
        for i in 0..5 {
            let t0 = std::time::Instant::now();
            field.tick(vp);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let s = field.retina.stats;
            eprintln!("probe tick {}: {:7.2} ms, pipes_total {}", i, ms, s.pipes_total);
            best = best.min(ms);
            pipes = s.pipes_total;
        }
        // Widest footprint at this pose, in receptor columns.
        let w = field.retina.width;
        for i in 0..field.entities.len() {
            let (mut u0, mut u1, mut v0, mut v1) = (u32::MAX, 0u32, u32::MAX, 0u32);
            let mut n = 0u32;
            for (rc, _) in field.retina.pipes_of(i) {
                u0 = u0.min(rc % w); u1 = u1.max(rc % w);
                v0 = v0.min(rc / w); v1 = v1.max(rc / w);
                n += 1;
            }
            if n == 0 { continue; }
            let cols = u1 - u0 + 1;
            if cols > widest.1 { widest = (i, cols, v1 - v0 + 1); }
        }
        {
            let e = &field.entities[widest.0];
            eprintln!("WIDEST: group {} radii {:?} pos {:?} depth {:.3} dist {:.2}",
                e.group, e.deposit_radii, e.position, field.retina.depth_of(widest.0),
                e.position.distance(eye));
        }
        eprintln!("PROBE RESULT: pipes_total {}, best walking tick {:.2} ms, widest footprint entity {} spans {}×{} receptors (retina {}×{})",
            pipes, best, widest.0, widest.1, widest.2, field.retina.width, field.retina.height);
        // Regression guard on the measurement itself. This pose linked
        // 8.93 M pipes before footprints were bounded by the kernel's own
        // projected extent, most of them from unit kernels beside the camera
        // — off screen entirely — painting bands across the whole image.
        assert!(pipes < 2_000_000,
            "pipes_total {} at the probe pose — footprints are not bounded by \
             the projected kernel extent", pipes);
    }
}

