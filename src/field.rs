// Diff Field — the persistent 3D texture that IS the universe.
//
// Entities deposit into it. The field spreads at c (one cell per tick).
// The observer reads it. Nothing else happens.
//
// The field doesn't move. It updates. Photons are not particles traveling
// through space — they are the update wavefront propagating through this texture.
// The observer moves through the field, colliding with deposits.

use glam::IVec3;

/// Field resolution — 128³ for v0.1 (2M cells, ~8MB at f32)
pub const FIELD_SIZE: u32 = 128;
pub const FIELD_CELLS: usize = (FIELD_SIZE * FIELD_SIZE * FIELD_SIZE) as usize;

/// A single deposit in the field
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FieldCell {
    /// Accumulated deposit density (0 = vacuum, >0 = something deposited here)
    pub density: f32,
    /// Color channels — what kind of deposit (R, G, B)
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

/// An entity that deposits into the field
pub struct Entity {
    pub position: glam::Vec3,
    pub velocity: glam::Vec3,
    pub deposit_magnitude: f32,
    pub color: [f32; 3],
}

/// The diff field — CPU-side representation
pub struct DiffField {
    /// The field data — flat array indexed by z * SIZE² + y * SIZE + x
    pub cells: Vec<FieldCell>,
    /// Active entities that deposit each tick
    pub entities: Vec<Entity>,
    /// Tick counter
    pub tick: u64,
}

impl DiffField {
    pub fn new() -> Self {
        let mut field = Self {
            cells: vec![FieldCell::default(); FIELD_CELLS],
            entities: Vec::new(),
            tick: 0,
        };

        // Seed with some initial entities
        field.spawn_demo_scene();

        field
    }

    /// Flat index from 3D coordinates
    fn index(x: u32, y: u32, z: u32) -> usize {
        (z * FIELD_SIZE * FIELD_SIZE + y * FIELD_SIZE + x) as usize
    }

    /// Check bounds
    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && x < FIELD_SIZE as i32
            && y < FIELD_SIZE as i32
            && z < FIELD_SIZE as i32
    }

    /// Spawn demo entities — a grid of depositing objects
    fn spawn_demo_scene(&mut self) {
        let center = FIELD_SIZE as f32 / 2.0;

        // A cluster of entities forming a "wall"
        for x in 0..20 {
            for y in 0..20 {
                self.entities.push(Entity {
                    position: glam::Vec3::new(
                        center + x as f32 * 2.0 - 20.0,
                        center + y as f32 * 2.0 - 20.0,
                        center + 30.0, // wall 30 cells ahead
                    ),
                    velocity: glam::Vec3::ZERO,
                    deposit_magnitude: 2.0,
                    color: [0.8, 0.6, 0.3], // warm wall color
                });
            }
        }

        // A sphere of entities
        let sphere_center = glam::Vec3::new(center - 15.0, center, center + 20.0);
        let sphere_radius = 8.0;
        for i in 0..500 {
            // Fibonacci sphere distribution
            let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
            let theta = 2.0 * std::f32::consts::PI * i as f32 / golden;
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / 500.0).acos();

            let pos = sphere_center
                + glam::Vec3::new(
                    sphere_radius * phi.sin() * theta.cos(),
                    sphere_radius * phi.sin() * theta.sin(),
                    sphere_radius * phi.cos(),
                );

            self.entities.push(Entity {
                position: pos,
                velocity: glam::Vec3::ZERO,
                deposit_magnitude: 1.5,
                color: [0.3, 0.5, 0.9], // cool blue sphere
            });
        }

        // A moving entity — orbiting point
        self.entities.push(Entity {
            position: glam::Vec3::new(center, center + 10.0, center + 15.0),
            velocity: glam::Vec3::new(0.3, 0.0, 0.1),
            deposit_magnitude: 5.0,
            color: [1.0, 0.2, 0.2], // bright red — hot moving object
        });

        // Floor plane
        for x in 0..40 {
            for z in 0..40 {
                self.entities.push(Entity {
                    position: glam::Vec3::new(
                        center + x as f32 * 2.0 - 40.0,
                        center - 20.0, // below observer
                        center + z as f32 * 2.0 - 20.0,
                    ),
                    velocity: glam::Vec3::ZERO,
                    deposit_magnitude: 1.0,
                    // Checkerboard pattern
                    color: if (x + z) % 2 == 0 {
                        [0.6, 0.6, 0.6]
                    } else {
                        [0.3, 0.3, 0.3]
                    },
                });
            }
        }

        log::info!(
            "Demo scene: {} entities ({} wall + 500 sphere + 1 orbiter + 1600 floor)",
            self.entities.len(),
            20 * 20
        );
    }

    /// Run one simulation tick
    pub fn tick(&mut self) {
        // Phase 1: entities deposit into field
        for entity in &mut self.entities {
            // Move entity
            entity.position += entity.velocity;

            // Bounce off field boundaries
            for i in 0..3 {
                if entity.position[i] < 1.0 || entity.position[i] >= (FIELD_SIZE - 1) as f32 {
                    entity.velocity[i] *= -1.0;
                    entity.position[i] = entity.position[i].clamp(1.0, (FIELD_SIZE - 2) as f32);
                }
            }

            // Deposit at current position
            let ix = entity.position.x as i32;
            let iy = entity.position.y as i32;
            let iz = entity.position.z as i32;

            if Self::in_bounds(ix, iy, iz) {
                let idx = Self::index(ix as u32, iy as u32, iz as u32);
                let cell = &mut self.cells[idx];
                cell.density = (cell.density + entity.deposit_magnitude).min(50.0);
                cell.color_r = (cell.color_r + entity.color[0] * entity.deposit_magnitude).min(50.0);
                cell.color_g = (cell.color_g + entity.color[1] * entity.deposit_magnitude).min(50.0);
                cell.color_b = (cell.color_b + entity.color[2] * entity.deposit_magnitude).min(50.0);
            }
        }

        // Phase 2: field spreading — each cell averages with its 6 neighbors
        // This IS light propagation. One cell per tick. c = 1.
        // We use a simple diffusion kernel: cell = (cell + sum(neighbors)) / 7
        //
        // For v0.1 we do this on CPU. v0.2 moves this to a compute shader.
        // The spreading factor controls how fast deposits dilute.
        let spread_factor: f32 = 0.005; // very gentle spreading

        // We need a copy to read from while writing
        let old = self.cells.clone();

        for z in 1..(FIELD_SIZE - 1) {
            for y in 1..(FIELD_SIZE - 1) {
                for x in 1..(FIELD_SIZE - 1) {
                    let idx = Self::index(x, y, z);
                    let current = old[idx];

                    if current.density < 0.001 {
                        // Check if any neighbor has something to spread
                        let neighbors = [
                            old[Self::index(x - 1, y, z)],
                            old[Self::index(x + 1, y, z)],
                            old[Self::index(x, y - 1, z)],
                            old[Self::index(x, y + 1, z)],
                            old[Self::index(x, y, z - 1)],
                            old[Self::index(x, y, z + 1)],
                        ];

                        let neighbor_density: f32 =
                            neighbors.iter().map(|n| n.density).sum::<f32>() / 6.0;

                        if neighbor_density > 0.001 {
                            let neighbor_r: f32 =
                                neighbors.iter().map(|n| n.color_r).sum::<f32>() / 6.0;
                            let neighbor_g: f32 =
                                neighbors.iter().map(|n| n.color_g).sum::<f32>() / 6.0;
                            let neighbor_b: f32 =
                                neighbors.iter().map(|n| n.color_b).sum::<f32>() / 6.0;

                            let cell = &mut self.cells[idx];
                            cell.density += neighbor_density * spread_factor;
                            cell.color_r += neighbor_r * spread_factor;
                            cell.color_g += neighbor_g * spread_factor;
                            cell.color_b += neighbor_b * spread_factor;
                        }
                    } else {
                        // Cell has density — spread some to neighbors (handled by their reads)
                        // Just apply mild decay to prevent infinite accumulation
                        let cell = &mut self.cells[idx];
                        cell.density *= 0.99;
                        cell.color_r *= 0.99;
                        cell.color_g *= 0.99;
                        cell.color_b *= 0.99;
                    }
                }
            }
        }

        self.tick += 1;
    }

    /// Get raw cell data as bytes for GPU upload
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.cells)
    }
}
