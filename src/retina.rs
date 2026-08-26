// Retina — the observer as a receptor array.
//
// The image is not a sample of a cached volume. It is the persistent state of
// W×H receptors on the image plane, each the sum of what has arrived along
// entity→receptor pipes. Pipes carry deltas only. Occlusion is transmittance
// along the pipe — nothing else.

use glam::{Mat4, Vec3, Vec4};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::field::cutoff_window;

pub const RETINA_W: u32 = 320;
pub const RETINA_H: u32 = 180;
pub const RETINA_ISO: f32 = 0.3;
pub const DELTA_EPS: f32 = 1e-4;
pub const RELINK_SHIFT: f32 = 0.5;
pub const ATTEN_END_MARGIN: f32 = 1.5;
pub const ATTEN_SAMPLES_PER_CELL: f32 = 1.0;
pub const ATTEN_THRESHOLD: f32 = 0.6;
pub const ATTEN_K_DEFAULT: f32 = 0.5;
pub const MIN_RETINA_DIM: u32 = 20;
pub const MAX_RETINA_DIM: u32 = 2560;

/// One entity's contribution as the retina sees it. Index-aligned with
/// `DiffField::entities` so pipes and edges can name entities by index.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub position: Vec3,
    /// Deposit radii (axis-aligned gaussian). ZERO → point source, unit radii.
    pub radii: Vec3,
    pub normal: Vec3,
    /// Static density used for transmittance (magnitude × boost, untuned).
    pub opacity: f32,
    /// Boosted density delivered this tick (τ applied by the retina).
    pub density: f32,
    /// Boosted color delivered this tick.
    pub color: [f32; 3],
    /// Gets pipes: solid, in frustum, not culled by trie depth.
    pub drawable: bool,
    /// Attenuates segments: solid, regardless of frustum.
    pub occluder: bool,
}

impl Source {
    pub fn kernel_radii(&self) -> Vec3 {
        if self.radii == Vec3::ZERO { Vec3::ONE } else { self.radii }
    }

    /// Feathered gaussian, 1.0 at the center, exactly 0.0 at exponent 4.
    /// Same kernel the grid deposit used, evaluated analytically.
    pub fn kernel(&self, p: Vec3) -> f32 {
        let d = (p - self.position) / self.kernel_radii();
        let e = d.dot(d);
        if e >= 4.0 { 0.0 } else { (-e).exp() * cutoff_window(e) }
    }
}

/// Uniform grid over occluder sources; cell = the widest kernel extent so a
/// 27-cell neighbourhood always covers every kernel that can reach a point.
pub struct SpatialHash {
    cell: f32,
    map: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHash {
    pub fn build(sources: &[Source]) -> Self {
        let max_r = sources.iter()
            .filter(|s| s.occluder)
            .map(|s| s.kernel_radii().max_element())
            .fold(1.0f32, f32::max);
        let cell = max_r * 2.0;
        let mut map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, s) in sources.iter().enumerate() {
            if !s.occluder { continue; }
            map.entry(Self::key(s.position, cell)).or_default().push(i);
        }
        Self { cell, map }
    }

    fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
        ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32)
    }

    /// Σ opacity·kernel over occluders near `p`, excluding indices in `skip`.
    pub fn density_at(&self, sources: &[Source], p: Vec3, skip: &[usize]) -> f32 {
        let (cx, cy, cz) = Self::key(p, self.cell);
        let mut rho = 0.0f32;
        for dz in -1..=1_i32 {
            for dy in -1..=1_i32 {
                for dx in -1..=1_i32 {
                    if let Some(bucket) = self.map.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in bucket {
                            if skip.contains(&j) { continue; }
                            rho += sources[j].opacity * sources[j].kernel(p);
                        }
                    }
                }
            }
        }
        rho
    }
}

/// Fraction of light surviving the segment from → to:
/// exp(−k · ∫ max(0, ρ − ATTEN_THRESHOLD) ds), skipping ATTEN_END_MARGIN at
/// both ends so an entity's own kernel (and its partner's) never self-shadows.
pub fn segment_transmittance(
    sources: &[Source],
    hash: &SpatialHash,
    from: Vec3,
    to: Vec3,
    skip: &[usize],
    k: f32,
) -> f32 {
    let ab = to - from;
    let len = ab.length();
    if len - 2.0 * ATTEN_END_MARGIN <= 0.0 { return 1.0; }
    let dir = ab / len;
    let step = 1.0 / ATTEN_SAMPLES_PER_CELL;
    let mut integral = 0.0f32;
    let mut t = ATTEN_END_MARGIN + 0.5 * step;
    while t < len - ATTEN_END_MARGIN {
        let rho = hash.density_at(sources, from + dir * t, skip);
        integral += (rho - ATTEN_THRESHOLD).max(0.0) * step;
        t += step;
    }
    (-k * integral).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(pos: Vec3, radii: Vec3, opacity: f32) -> Source {
        Source {
            position: pos,
            radii,
            normal: Vec3::Y,
            opacity,
            density: opacity,
            color: [1.0, 1.0, 1.0],
            drawable: true,
            occluder: true,
        }
    }

    #[test]
    fn kernel_is_one_at_center_and_zero_past_cutoff() {
        let s = src(Vec3::ZERO, Vec3::new(2.0, 1.0, 1.0), 1.0);
        assert!((s.kernel(Vec3::ZERO) - 1.0).abs() < 1e-6);
        assert_eq!(s.kernel(Vec3::new(4.0, 0.0, 0.0)), 0.0); // e = 4 → cut
        assert!(s.kernel(Vec3::new(2.0, 0.0, 0.0)) > 0.3);     // e = 1 → exp(-1)
        // point source falls back to unit radii
        let p = src(Vec3::ZERO, Vec3::ZERO, 1.0);
        assert_eq!(p.kernel_radii(), Vec3::ONE);
    }

    #[test]
    fn transmittance_is_one_with_no_occluder_on_the_segment() {
        let sources = vec![src(Vec3::new(10.0, 0.0, -5.0), Vec3::ONE, 10.0)];
        let hash = SpatialHash::build(&sources);
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, &[], 0.5);
        assert!((t - 1.0).abs() < 1e-6, "off-segment occluder attenuated: {}", t);
    }

    #[test]
    fn transmittance_matches_fine_quadrature_for_centered_occluder() {
        let occ = src(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 10.0);
        let sources = vec![occ];
        let hash = SpatialHash::build(&sources);
        let k = 0.5;
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, &[], k);
        // Independent reference: fine quadrature of the same integrand
        let mut integral = 0.0f32;
        let dz = 0.01;
        let mut z = -10.0 + ATTEN_END_MARGIN;
        while z < -ATTEN_END_MARGIN {
            let rho = occ.opacity * occ.kernel(Vec3::new(0.0, 0.0, z));
            integral += (rho - ATTEN_THRESHOLD).max(0.0) * dz;
            z += dz;
        }
        let expected = (-k * integral).exp();
        assert!(t < 0.5, "centered opaque occluder barely attenuates: {}", t);
        assert!((t - expected).abs() < 0.15 * expected.max(0.05), "τ={} expected≈{}", t, expected);
    }

    #[test]
    fn transmittance_skips_listed_sources_and_end_margins() {
        let a = src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 50.0); // at `from`
        let b = src(Vec3::ZERO, Vec3::ONE, 50.0);                  // at `to`
        let mid = src(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 50.0);
        let sources = vec![a, b, mid];
        let hash = SpatialHash::build(&sources);
        // Endpoint sources sit inside the 1.5-cell margins → never sampled
        let t_ends = segment_transmittance(&sources[..2], &SpatialHash::build(&sources[..2]),
            a.position, b.position, &[], 0.5);
        assert!((t_ends - 1.0).abs() < 1e-6, "endpoint sources leaked into the integral: {}", t_ends);
        // Skipping the middle occluder restores full transmittance
        let t_skip = segment_transmittance(&sources, &hash, a.position, b.position, &[2], 0.5);
        assert!((t_skip - 1.0).abs() < 1e-6);
        let t_block = segment_transmittance(&sources, &hash, a.position, b.position, &[], 0.5);
        assert!(t_block < 0.1);
    }

    #[test]
    fn spatial_hash_ignores_non_occluders() {
        let mut ghost = src(Vec3::ZERO, Vec3::ONE, 100.0);
        ghost.occluder = false;
        let sources = vec![ghost];
        let hash = SpatialHash::build(&sources);
        assert_eq!(hash.density_at(&sources, Vec3::ZERO, &[]), 0.0);
    }
}
