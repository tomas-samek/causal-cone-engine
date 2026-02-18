// Diff Field — the persistent 3D texture that IS the universe.
//
// Entities deposit into it. Light propagates through CONNECTIONS between entities,
// not through grid diffusion. The grid is just the observer's retina.
//
// Entity → deposits own color to grid (rendering)
// Entity → sends deposit along connections to neighbors (propagation)
// Neighbor → accumulates incoming, adds to own deposit next tick (mixing)

/// Field resolution — 128³ (2M cells, ~32MB at f32)
pub const FIELD_SIZE: u32 = 128;
pub const FIELD_CELLS: usize = (FIELD_SIZE * FIELD_SIZE * FIELD_SIZE) as usize;

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
            specular: 0.0,
            reemit: 0.7, // surfaces re-emit 70% — only way to be visible
            reemit_r: 0.0,
            reemit_g: 0.0,
            reemit_b: 0.0,
            edge_start: 0,
            edge_count: 0,
            incoming: EdgeDeposit::default(),
        }
    }
}

/// The diff field — CPU-side representation
pub struct DiffField {
    pub cells: Vec<FieldCell>,
    pub entities: Vec<Entity>,
    pub tick: u64,
    deliveries: Vec<EdgeDeposit>,
    pub aabb_min: glam::Vec3,
    pub aabb_max: glam::Vec3,
    pub dirty_slabs: [bool; FIELD_SIZE as usize],
    // SoA edge storage — flat contiguous arrays for cache-friendly iteration
    edge_targets: Vec<usize>,
    edge_deposits: Vec<EdgeDeposit>,
}

impl DiffField {
    pub fn new() -> Self {
        let mut field = Self {
            cells: vec![FieldCell::default(); FIELD_CELLS],
            entities: Vec::new(),
            tick: 0,
            deliveries: Vec::new(),
            aabb_min: glam::Vec3::ZERO,
            aabb_max: glam::Vec3::splat(FIELD_SIZE as f32),
            dirty_slabs: [true; FIELD_SIZE as usize],
            edge_targets: Vec::new(),
            edge_deposits: Vec::new(),
        };

        field.spawn_demo_scene();
        field.build_all_edges(3.5, 15.0);
        field.deliveries = vec![EdgeDeposit::default(); field.entities.len()];

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

    /// Build all edges (connections + radiation links) into flat SoA arrays.
    /// Two-phase: collect into temp per-entity vecs, then flatten for cache locality.
    fn build_all_edges(&mut self, connect_dist: f32, radiation_dist: f32) {
        let connect_dist_sq = connect_dist * connect_dist;
        let radiation_dist_sq = radiation_dist * radiation_dist;
        let radiation_min_sq = connect_dist * connect_dist; // radiation starts where connections end
        let n = self.entities.len();

        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();

        // Phase 1: collect edges into temp per-entity vecs
        let mut temp_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut connection_edges = 0u64;
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = positions[i].distance_squared(positions[j]);
                if dist_sq < connect_dist_sq {
                    temp_edges[i].push(j);
                    temp_edges[j].push(i);
                    connection_edges += 1;
                }
            }
        }

        log::info!(
            "Graph built: {} entities, {} connection edges (avg {:.1} per entity)",
            n,
            connection_edges,
            (connection_edges * 2) as f32 / n as f32
        );

        // Detect interior (heat) entities before radiation links.
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

        // Phase 2: radiation links — long-range surface-to-surface
        let mut radiation_edges = 0u64;
        for i in 0..n {
            if self.entities[i].is_vacuum || self.entities[i].is_heat { continue; }
            for j in (i + 1)..n {
                if self.entities[j].is_vacuum || self.entities[j].is_heat { continue; }
                let dist_sq = positions[i].distance_squared(positions[j]);
                if dist_sq >= radiation_min_sq && dist_sq < radiation_dist_sq {
                    temp_edges[i].push(j);
                    temp_edges[j].push(i);
                    radiation_edges += 1;
                }
            }
        }
        log::info!("Radiation links: {} direct surface-to-surface edges", radiation_edges);

        // Phase 3: flatten into SoA
        let total: usize = temp_edges.iter().map(|e| e.len()).sum();
        self.edge_targets = Vec::with_capacity(total);
        self.edge_deposits = vec![EdgeDeposit::default(); total];
        let mut offset = 0u32;
        for (i, edges) in temp_edges.iter().enumerate() {
            self.entities[i].edge_start = offset;
            self.entities[i].edge_count = edges.len() as u32;
            for &target in edges {
                self.edge_targets.push(target);
            }
            offset += edges.len() as u32;
        }

        log::info!(
            "Edge SoA: {} total directed edges ({} connection + {} radiation), {:.1} MB contiguous",
            total,
            connection_edges * 2,
            radiation_edges * 2,
            (total * (std::mem::size_of::<usize>() + std::mem::size_of::<EdgeDeposit>())) as f64 / 1_048_576.0
        );
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
                        self.entities.push(e);
                    }
                    z += spacing;
                }
                y += spacing;
            }
            x += spacing;
        }
    }

    fn spawn_demo_scene(&mut self) {
        let center = FIELD_SIZE as f32 / 2.0;
        // Dino faces +Z, centered in field
        let base = glam::Vec3::new(center, center - 5.0, center);
        let green = [0.2, 0.6, 0.15];       // body green
        let dark_green = [0.15, 0.45, 0.1]; // darker accents
        let belly = [0.5, 0.65, 0.3];       // lighter belly
        let eye_color = [1.0, 0.8, 0.0];    // yellow eyes
        let mouth = [0.7, 0.2, 0.15];       // reddish mouth
        let sp = 0.9; // tight spacing — surface stays solid after interior becomes heat

        // BODY — large horizontal ellipsoid
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 5.0, 0.0),
            glam::Vec3::new(5.0, 6.0, 8.0),
            green, 0.05, sp, 0.3, // nearly dark — visible only from reflected light
        );

        // BELLY — slightly lighter, extends lower to catch floor bounce
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::new(4.5, 4.0, 7.0),
            belly, 0.05, sp, 0.45, // visible only from reflected light
        );

        // TAIL — tapers backward (-Z)
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 5.5, -12.0),
            glam::Vec3::new(2.5, 2.5, 7.0),
            green, 0.05, sp, 0.3,
        );
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 5.5, -20.0),
            glam::Vec3::new(1.2, 1.2, 4.0),
            dark_green, 0.05, sp, 0.25,
        );

        // NECK — tilted upward
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 10.0, 8.0),
            glam::Vec3::new(3.0, 5.0, 3.0),
            green, 0.05, sp, 0.3,
        );

        // HEAD — on top of neck
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 16.0, 10.0),
            glam::Vec3::new(3.5, 3.0, 5.0),
            green, 0.05, sp, 0.3,
        );

        // JAW — below head, slightly forward
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 13.5, 12.0),
            glam::Vec3::new(2.5, 1.5, 4.0),
            dark_green, 0.05, sp, 0.25,
        );

        // MOUTH interior
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 14.5, 13.0),
            glam::Vec3::new(2.0, 0.8, 3.0),
            mouth, 0.05, sp, 0.15,
        );

        // EYES — two small bright spheres (nearly opaque)
        self.fill_ellipsoid(
            base + glam::Vec3::new(3.0, 17.0, 12.0),
            glam::Vec3::new(0.8, 0.8, 0.8),
            eye_color, 4.0, sp, 0.1,
        );
        self.fill_ellipsoid(
            base + glam::Vec3::new(-3.0, 17.0, 12.0),
            glam::Vec3::new(0.8, 0.8, 0.8),
            eye_color, 4.0, sp, 0.1,
        );

        // LEFT LEG
        self.fill_ellipsoid(
            base + glam::Vec3::new(3.0, -3.0, 1.0),
            glam::Vec3::new(2.0, 5.0, 2.5),
            dark_green, 0.05, sp, 0.25,
        );
        // LEFT FOOT
        self.fill_ellipsoid(
            base + glam::Vec3::new(3.0, -8.0, 2.0),
            glam::Vec3::new(2.5, 1.0, 4.0),
            dark_green, 0.05, sp, 0.25,
        );

        // RIGHT LEG
        self.fill_ellipsoid(
            base + glam::Vec3::new(-3.0, -3.0, 1.0),
            glam::Vec3::new(2.0, 5.0, 2.5),
            dark_green, 0.05, sp, 0.25,
        );
        // RIGHT FOOT
        self.fill_ellipsoid(
            base + glam::Vec3::new(-3.0, -8.0, 2.0),
            glam::Vec3::new(2.5, 1.0, 4.0),
            dark_green, 0.05, sp, 0.25,
        );

        // TINY ARMS — classic T-Rex
        self.fill_ellipsoid(
            base + glam::Vec3::new(4.5, 6.0, 5.0),
            glam::Vec3::new(1.0, 2.5, 1.0),
            green, 0.05, sp, 0.3,
        );
        self.fill_ellipsoid(
            base + glam::Vec3::new(-4.5, 6.0, 5.0),
            glam::Vec3::new(1.0, 2.5, 1.0),
            green, 0.05, sp, 0.3,
        );

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
                    50.0, // directional highlight — sky dome handles ambient fill
                    [1.0, 0.9, 0.5],
                );
                light.pass_through = 1.0; // pure emitter
                light.is_vacuum = true; // sun emits through graph, not visible in grid
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
                e.reemit = 0.9;       // near-perfect re-emitter — floor is the main bounce surface
                self.entities.push(e);
            }
        }

        let before_vacuum = self.entities.len();

        // VACUUM — sparse relay network for light to travel through empty space.
        // Each vacuum entity is nearly invisible, nearly transparent.
        // Light propagates through this network. Dino body BLOCKS paths by absorbing.
        // Shadow = where dino interrupts vacuum relay chains between light and floor.
        let vac_spacing = 3.0;
        let vac_start = glam::Vec3::new(center - 30.0, center - 14.0, center - 25.0);
        let vac_end = glam::Vec3::new(center + 30.0, center + 32.0, center + 25.0); // up to sun

        let mut vx = vac_start.x;
        while vx <= vac_end.x {
            let mut vy = vac_start.y;
            while vy <= vac_end.y {
                let mut vz = vac_start.z;
                while vz <= vac_end.z {
                    let pos = glam::Vec3::new(vx, vy, vz);

                    // Skip if inside the dino body (rough bounding check)
                    let rel = pos - base;
                    let in_body = (rel.x / 6.0).powi(2) + ((rel.y - 5.0) / 8.0).powi(2) + (rel.z / 10.0).powi(2) < 1.0;
                    if !in_body {
                        let mut e = Entity::new(
                            pos,
                            glam::Vec3::ZERO,
                            0.01, // nearly invisible — vacuum doesn't glow
                            [0.0, 0.0, 0.0],
                        );
                        e.pass_through = 0.95; // air is nearly transparent
                        e.is_vacuum = true;
                        // Below sun = atmosphere. Upper atmosphere emits (sky dome).
                        // Lower atmosphere just relays with blue scatter.
                        if vy < sun_y {
                            let height_frac = (vy - (center - 13.0)) / (sun_y - (center - 13.0)); // 0=floor, 1=sun
                            if height_frac > 0.7 {
                                // Upper atmosphere — acts as sky dome emitter
                                // Sun has had time to diffuse across the whole sky
                                e.deposit_magnitude = 3.0;
                                e.color = [0.8, 0.85, 1.0]; // warm sky white-blue
                            } else {
                                // Lower atmosphere — just relays with blue scatter
                                e.scatter = 0.002;
                                e.deposit_magnitude = 0.001;
                                e.color = [0.3, 0.4, 1.0]; // Rayleigh blue
                            }
                        }
                        self.entities.push(e);
                    }

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
    }

    /// Run one simulation tick — push-driven pipe propagation
    pub fn tick(&mut self) {
        // depletion is now per-entity via pass_through

        // Phase 0: decay the grid — only dirty slabs
        let slab_size = (FIELD_SIZE * FIELD_SIZE) as usize;
        for z in 0..FIELD_SIZE as usize {
            if !self.dirty_slabs[z] { continue; }
            let start = z * slab_size;
            let end = start + slab_size;
            let mut any_nonzero = false;
            for cell in &mut self.cells[start..end] {
                cell.density *= 0.92;
                cell.color_r *= 0.92;
                cell.color_g *= 0.92;
                cell.color_b *= 0.92;
                if cell.density > 0.001 { any_nonzero = true; }
            }
            if !any_nonzero { self.dirty_slabs[z] = false; }
        }

        // Phase 1: DELIVER — each edge's pipe contents arrive at target.
        // Flat SoA iteration for cache-friendly access.
        self.deliveries.fill(EdgeDeposit::default());
        for entity in &self.entities {
            let start = entity.edge_start as usize;
            let end = start + entity.edge_count as usize;
            for k in start..end {
                let target = self.edge_targets[k];
                self.deliveries[target].r += self.edge_deposits[k].r;
                self.deliveries[target].g += self.edge_deposits[k].g;
                self.deliveries[target].b += self.edge_deposits[k].b;
                self.deliveries[target].density += self.edge_deposits[k].density;
            }
        }
        // Apply deliveries + build re-emission for solid surfaces
        for (i, entity) in self.entities.iter_mut().enumerate() {
            entity.incoming = self.deliveries[i];

            // Solid surfaces absorb incoming light and become secondary emitters.
            // Each entity represents billions of atoms — coherent re-emission,
            // not relay-node pass-through. Color-filtered by surface.
            // Capped to prevent runaway feedback.
            if !entity.is_vacuum && entity.reemit > 0.0 && self.deliveries[i].density > 0.01 {
                let re = entity.reemit;
                let cap = 50.0; // high cap — floor needs to blast light upward to belly
                entity.reemit_r = (self.deliveries[i].r * re * entity.color[0]).min(cap);
                entity.reemit_g = (self.deliveries[i].g * re * entity.color[1]).min(cap);
                entity.reemit_b = (self.deliveries[i].b * re * entity.color[2]).min(cap);
            }

            // Heat nodes wake up if enough light reaches them through neighbors
            if entity.is_heat && self.deliveries[i].density > 1.0 {
                entity.is_heat = false; // light penetrated — no longer invisible
            }
        }

        // Phase 2: PUSH — each entity pushes new content into its pipes.
        // New content = entity's own emission + pass-through of incoming (depleted).
        // This REPLACES what was in the pipe (old content was delivered in Phase 1).
        // CUTOFF: if incoming is below threshold, don't pass it through (signal is dead).
        let cutoff: f32 = 0.01;

        // Split borrow: read entities, write edge_deposits
        let entities = &self.entities;
        let edge_deposits = &mut self.edge_deposits;
        for entity in entities.iter() {
            let n = entity.edge_count as f32;
            if n == 0.0 { continue; }

            let mag = entity.deposit_magnitude;
            // Own emission + re-emission from absorbed light.
            // Re-emission makes illuminated surfaces into secondary light sources.
            let own_r = (entity.color[0] * mag + entity.reemit_r) / n;
            let own_g = (entity.color[1] * mag + entity.reemit_g) / n;
            let own_b = (entity.color[2] * mag + entity.reemit_b) / n;
            let own_d = (mag + entity.reemit_r + entity.reemit_g + entity.reemit_b) / n;

            // Pass-through has two components:
            // 1. Specular: mirror bounce, unfiltered (waxy surface, wet, metallic)
            // 2. Diffuse: color-filtered (the material's absorption spectrum)
            // Vacuum just attenuates uniformly.
            let (pass_r, pass_g, pass_b, pass_d) = if entity.incoming.density > cutoff {
                let pt = entity.pass_through;
                if entity.is_vacuum {
                    (
                        entity.incoming.r * pt / n,
                        entity.incoming.g * pt / n,
                        entity.incoming.b * pt / n,
                        entity.incoming.density * pt / n,
                    )
                } else {
                    let spec = entity.specular;
                    let diff = 1.0 - spec;
                    (
                        (entity.incoming.r * spec + entity.incoming.r * diff * entity.color[0]) * pt / n,
                        (entity.incoming.g * spec + entity.incoming.g * diff * entity.color[1]) * pt / n,
                        (entity.incoming.b * spec + entity.incoming.b * diff * entity.color[2]) * pt / n,
                        entity.incoming.density * pt / n,
                    )
                }
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            // Push into each pipe (flat array, cache-friendly)
            let start = entity.edge_start as usize;
            let end = start + entity.edge_count as usize;
            for k in start..end {
                edge_deposits[k].r = own_r + pass_r;
                edge_deposits[k].g = own_g + pass_g;
                edge_deposits[k].b = own_b + pass_b;
                edge_deposits[k].density = own_d + pass_d;
            }
        }

        // Phase 3: entities deposit to grid (for rendering)
        let mut aabb_min = glam::Vec3::splat(FIELD_SIZE as f32);
        let mut aabb_max = glam::Vec3::splat(0.0);
        for entity in &mut self.entities {
            // Move entity
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

            // Vacuum with atmosphere: scatter a tiny fraction into the grid.
            // This IS air — molecules redirecting photons, creating ambient light.
            if entity.is_vacuum {
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

            let ix = entity.position.x as i32;
            let iy = entity.position.y as i32;
            let iz = entity.position.z as i32;

            if Self::in_bounds(ix, iy, iz) {
                let idx = Self::index(ix as u32, iy as u32, iz as u32);
                let mag = entity.deposit_magnitude;

                // Own deposit + incoming light + re-emission glow
                let total_r = entity.color[0] * mag + entity.incoming.r + entity.reemit_r;
                let total_g = entity.color[1] * mag + entity.incoming.g + entity.reemit_g;
                let total_b = entity.color[2] * mag + entity.incoming.b + entity.reemit_b;
                let total_d = mag + entity.incoming.density;

                let cell = &mut self.cells[idx];
                cell.density = (cell.density + total_d).min(50.0);
                cell.color_r = (cell.color_r + total_r).min(50.0);
                cell.color_g = (cell.color_g + total_g).min(50.0);
                cell.color_b = (cell.color_b + total_b).min(50.0);
                self.dirty_slabs[iz as usize] = true;
            }
        }
        self.aabb_min = aabb_min.max(glam::Vec3::ZERO);
        self.aabb_max = aabb_max.min(glam::Vec3::splat(FIELD_SIZE as f32));

        self.tick += 1;
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.cells)
    }
}
