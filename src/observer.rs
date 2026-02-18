// Observer — the thing that swims through the diff field.
//
// The observer has mass (it exists). Mass means energy. Energy means
// it can fight the substrate stream. Its velocity is always < c because
// it has memory (state, history, accumulated imprints).
//
// A photon has no mass, no energy, no choice. It IS the stream.
// The observer is not the stream. It moves through the stream.

use glam::{Mat4, Quat, Vec3};
use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// Maximum observer speed as fraction of c.
/// At v=1.0 you're a photon — no rendering possible.
/// At v=0 you're stationary — maximum visual depth.
const MAX_SPEED: f32 = 0.5; // half c — generous but not blinding

/// Movement acceleration in cells per second²
const ACCEL: f32 = 20.0;
/// Friction / drag — how quickly you stop without input
const DRAG: f32 = 5.0;
/// Mouse sensitivity
const MOUSE_SENSITIVITY: f32 = 0.002;

pub struct Observer {
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,   // radians, rotation around Y axis
    pub pitch: f32,  // radians, rotation around X axis

    /// Field of view in radians — varies with speed
    pub base_fov: f32,
}

impl Observer {
    pub fn new() -> Self {
        Self {
            // Start in front of the dino's face, looking back at it
            position: Vec3::new(64.0, 64.0, 95.0),
            velocity: Vec3::ZERO,
            yaw: -std::f32::consts::FRAC_PI_2, // face -Z toward the dino
            pitch: 0.0,
            base_fov: std::f32::consts::FRAC_PI_2, // 90 degrees
        }
    }

    /// Current speed as fraction of c (c = 1 cell per tick = 30 cells/sec at 30 tps)
    pub fn speed(&self) -> f32 {
        // c = 1 cell/tick. At 30 ticks/sec, c = 30 cells/sec in world units.
        // Speed as fraction of c:
        self.velocity.length() / 30.0
    }

    /// Effective field of view — narrows with speed (relativistic aberration)
    pub fn effective_fov(&self) -> f32 {
        let v = self.speed().min(0.99);
        // Relativistic aberration: FOV narrows as v → c
        // half_angle = acos((cos(base_half) + v) / (1 + v * cos(base_half)))
        // Simplified: just scale linearly for v0.1
        self.base_fov * (1.0 - v * 0.8)
    }

    /// Forward direction vector
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Right direction vector
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    /// Up direction vector
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    /// View matrix for the GPU
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }

    /// Projection matrix
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.effective_fov(), aspect, 0.1, 500.0)
    }

    /// Update from keyboard input
    pub fn update(&mut self, keys: &HashSet<KeyCode>, dt: f64) {
        let dt = dt as f32;
        let forward = self.forward();
        let right = self.right();

        // Acceleration from input
        let mut accel = Vec3::ZERO;
        if keys.contains(&KeyCode::KeyW) {
            accel += forward;
        }
        if keys.contains(&KeyCode::KeyS) {
            accel -= forward;
        }
        if keys.contains(&KeyCode::KeyD) {
            accel += right;
        }
        if keys.contains(&KeyCode::KeyA) {
            accel -= right;
        }
        if keys.contains(&KeyCode::Space) {
            accel += Vec3::Y;
        }
        if keys.contains(&KeyCode::ShiftLeft) {
            accel -= Vec3::Y;
        }

        if accel.length() > 0.0 {
            accel = accel.normalize() * ACCEL;
        }

        // Apply acceleration
        self.velocity += accel * dt;

        // Apply drag (deceleration when no input)
        self.velocity *= (-DRAG * dt).exp();

        // Clamp to max speed (can't reach c)
        let max_world_speed = MAX_SPEED * 30.0; // convert from c-fraction to cells/sec
        if self.velocity.length() > max_world_speed {
            self.velocity = self.velocity.normalize() * max_world_speed;
        }

        // Update position
        self.position += self.velocity * dt;
    }

    /// Mouse look
    pub fn mouse_look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * MOUSE_SENSITIVITY;
        self.pitch -= dy * MOUSE_SENSITIVITY;

        // Clamp pitch to avoid flipping
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }
}
