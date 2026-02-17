// Diff Field — the persistent 3D texture that IS the universe.
//
// Entities deposit into it. Light propagates through CONNECTIONS between entities,
// not through grid diffusion. The grid is just the observer's retina.
//
// Entity → deposits own color to grid (rendering)
// Entity → sends deposit along connections to neighbors (propagation)
// Neighbor → accumulates incoming, adds to own deposit next tick (mixing)

use glam::IVec3;

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

/// An entity that deposits into the field.
pub struct Entity {
    pub position: glam::Vec3,
    pub velocity: glam::Vec3,
    pub deposit_magnitude: f32,
    pub color: [f32; 3],
    /// Indices of connected entities — light travels along these edges
    pub connections: Vec<usize>,
    /// Accumulated incoming deposits from connected neighbors (RGB + density)
    pub incoming_r: f32,
    pub incoming_g: f32,
    pub incoming_b: f32,
    pub incoming_density: f32,
}

impl Entity {
    pub fn new(position: glam::Vec3, velocity: glam::Vec3, deposit_magnitude: f32, color: [f32; 3]) -> Self {
        Self {
            position,
            velocity,
            deposit_magnitude,
            color,
            connections: Vec::new(),
            incoming_r: 0.0,
            incoming_g: 0.0,
            incoming_b: 0.0,
            incoming_density: 0.0,
        }
    }
}

/// The diff field — CPU-side representation
pub struct DiffField {
    pub cells: Vec<FieldCell>,
    pub entities: Vec<Entity>,
    pub tick: u64,
}

impl DiffField {
    pub fn new() -> Self {
        let mut field = Self {
            cells: vec![FieldCell::default(); FIELD_CELLS],
            entities: Vec::new(),
            tick: 0,
        };

        field.spawn_demo_scene();
        field.build_connections(2.5); // connect entities within 2.5 cells of each other

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

    /// Connect entities that are within `max_dist` of each other.
    /// This IS the graph. Light travels along these edges.
    fn build_connections(&mut self, max_dist: f32) {
        let max_dist_sq = max_dist * max_dist;
        let n = self.entities.len();

        // Collect positions first to avoid borrow issues
        let positions: Vec<glam::Vec3> = self.entities.iter().map(|e| e.position).collect();

        let mut total_connections = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = positions[i].distance_squared(positions[j]);
                if dist_sq < max_dist_sq {
                    self.entities[i].connections.push(j);
                    self.entities[j].connections.push(i);
                    total_connections += 1;
                }
            }
        }

        log::info!(
            "Graph built: {} entities, {} edges (avg {:.1} connections/entity)",
            n,
            total_connections,
            (total_connections * 2) as f32 / n as f32
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
                        self.entities.push(Entity::new(
                            center + glam::Vec3::new(x, y, z),
                            glam::Vec3::ZERO,
                            magnitude,
                            color,
                        ));
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
        let sp = 1.2; // spacing between entities

        // BODY — large horizontal ellipsoid
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 5.0, 0.0),
            glam::Vec3::new(5.0, 6.0, 8.0),
            green, 2.0, sp,
        );

        // BELLY — slightly lighter, overlapping lower body
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 2.5, 0.0),
            glam::Vec3::new(4.0, 3.0, 7.0),
            belly, 2.5, sp,
        );

        // TAIL — tapers backward (-Z)
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 5.5, -12.0),
            glam::Vec3::new(2.5, 2.5, 7.0),
            green, 1.8, sp,
        );
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 5.5, -20.0),
            glam::Vec3::new(1.2, 1.2, 4.0),
            dark_green, 1.5, sp,
        );

        // NECK — tilted upward
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 10.0, 8.0),
            glam::Vec3::new(3.0, 5.0, 3.0),
            green, 2.0, sp,
        );

        // HEAD — on top of neck
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 16.0, 10.0),
            glam::Vec3::new(3.5, 3.0, 5.0),
            green, 2.5, sp,
        );

        // JAW — below head, slightly forward
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 13.5, 12.0),
            glam::Vec3::new(2.5, 1.5, 4.0),
            dark_green, 2.0, sp,
        );

        // MOUTH interior
        self.fill_ellipsoid(
            base + glam::Vec3::new(0.0, 14.5, 13.0),
            glam::Vec3::new(2.0, 0.8, 3.0),
            mouth, 1.5, sp,
        );

        // EYES — two small bright spheres
        self.fill_ellipsoid(
            base + glam::Vec3::new(3.0, 17.0, 12.0),
            glam::Vec3::new(0.8, 0.8, 0.8),
            eye_color, 4.0, sp,
        );
        self.fill_ellipsoid(
            base + glam::Vec3::new(-3.0, 17.0, 12.0),
            glam::Vec3::new(0.8, 0.8, 0.8),
            eye_color, 4.0, sp,
        );

        // LEFT LEG
        self.fill_ellipsoid(
            base + glam::Vec3::new(3.0, -3.0, 1.0),
            glam::Vec3::new(2.0, 5.0, 2.5),
            dark_green, 2.0, sp,
        );
        // LEFT FOOT
        self.fill_ellipsoid(
            base + glam::Vec3::new(3.0, -8.0, 2.0),
            glam::Vec3::new(2.5, 1.0, 4.0),
            dark_green, 2.0, sp,
        );

        // RIGHT LEG
        self.fill_ellipsoid(
            base + glam::Vec3::new(-3.0, -3.0, 1.0),
            glam::Vec3::new(2.0, 5.0, 2.5),
            dark_green, 2.0, sp,
        );
        // RIGHT FOOT
        self.fill_ellipsoid(
            base + glam::Vec3::new(-3.0, -8.0, 2.0),
            glam::Vec3::new(2.5, 1.0, 4.0),
            dark_green, 2.0, sp,
        );

        // TINY ARMS — classic T-Rex
        self.fill_ellipsoid(
            base + glam::Vec3::new(4.5, 6.0, 5.0),
            glam::Vec3::new(1.0, 2.5, 1.0),
            green, 1.5, sp,
        );
        self.fill_ellipsoid(
            base + glam::Vec3::new(-4.5, 6.0, 5.0),
            glam::Vec3::new(1.0, 2.5, 1.0),
            green, 1.5, sp,
        );

        // LIGHT SOURCE — warm sun above and behind observer
        self.entities.push(Entity::new(
            glam::Vec3::new(center + 20.0, center + 20.0, center - 20.0),
            glam::Vec3::ZERO,
            10.0,
            [1.0, 0.9, 0.5],
        ));

        // FLOOR
        for x in 0..30 {
            for z in 0..30 {
                self.entities.push(Entity::new(
                    glam::Vec3::new(
                        center + x as f32 * 2.0 - 30.0,
                        center - 13.0,
                        center + z as f32 * 2.0 - 20.0,
                    ),
                    glam::Vec3::ZERO,
                    0.8,
                    if (x + z) % 2 == 0 {
                        [0.45, 0.35, 0.2]  // dirt
                    } else {
                        [0.3, 0.5, 0.15]   // grass
                    },
                ));
            }
        }

        log::info!("Demo scene: {} entities (T-Rex + floor + light)", self.entities.len());
    }

    /// Run one simulation tick
    pub fn tick(&mut self) {
        // Phase 0: decay the grid — old deposits fade
        for cell in self.cells.iter_mut() {
            cell.density *= 0.92;
            cell.color_r *= 0.92;
            cell.color_g *= 0.92;
            cell.color_b *= 0.92;
        }

        // Phase 1: each entity deposits to grid (for rendering)
        // Deposit = own color + whatever arrived via connections
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

            let ix = entity.position.x as i32;
            let iy = entity.position.y as i32;
            let iz = entity.position.z as i32;

            if Self::in_bounds(ix, iy, iz) {
                let idx = Self::index(ix as u32, iy as u32, iz as u32);
                let mag = entity.deposit_magnitude;

                // Own deposit + incoming from graph
                let total_r = entity.color[0] * mag + entity.incoming_r;
                let total_g = entity.color[1] * mag + entity.incoming_g;
                let total_b = entity.color[2] * mag + entity.incoming_b;
                let total_d = mag + entity.incoming_density;

                let cell = &mut self.cells[idx];
                cell.density = (cell.density + total_d).min(50.0);
                cell.color_r = (cell.color_r + total_r).min(50.0);
                cell.color_g = (cell.color_g + total_g).min(50.0);
                cell.color_b = (cell.color_b + total_b).min(50.0);
            }
        }

        // Phase 2: propagate deposits along connections
        // Each entity sends a fraction of its total deposit to each connected neighbor.
        // This IS light propagation. One hop per tick = c.
        let propagation_strength: f32 = 0.3; // fraction sent to EACH neighbor

        // Collect what each entity wants to send
        let sends: Vec<(f32, f32, f32, f32, Vec<usize>)> = self.entities.iter().map(|e| {
            let n = e.connections.len() as f32;
            if n == 0.0 {
                return (0.0, 0.0, 0.0, 0.0, vec![]);
            }
            let frac = propagation_strength / n; // divide among connections
            let mag = e.deposit_magnitude;
            // Send own color + incoming (re-emit what you received)
            let send_r = (e.color[0] * mag + e.incoming_r) * frac;
            let send_g = (e.color[1] * mag + e.incoming_g) * frac;
            let send_b = (e.color[2] * mag + e.incoming_b) * frac;
            let send_d = (mag + e.incoming_density) * frac;
            (send_r, send_g, send_b, send_d, e.connections.clone())
        }).collect();

        // Clear incoming for next tick
        for entity in self.entities.iter_mut() {
            entity.incoming_r = 0.0;
            entity.incoming_g = 0.0;
            entity.incoming_b = 0.0;
            entity.incoming_density = 0.0;
        }

        // Deliver deposits along edges
        for (send_r, send_g, send_b, send_d, connections) in &sends {
            for &target in connections {
                self.entities[target].incoming_r += send_r;
                self.entities[target].incoming_g += send_g;
                self.entities[target].incoming_b += send_b;
                self.entities[target].incoming_density += send_d;
            }
        }

        self.tick += 1;
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.cells)
    }
}
