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

/// Persistent state of one image-plane cell. Sums, never averages: the
/// shader divides by density. Never decays; only deltas ever touch it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Receptor {
    pub density: f32,
    pub color: [f32; 3],
    pub normal: Vec3,
    pub depth: f32,
}

impl Receptor {
    fn add(&mut self, p: &PipeState) {
        self.density += p.density;
        self.color[0] += p.color[0];
        self.color[1] += p.color[1];
        self.color[2] += p.color[2];
        self.normal += p.normal;
        self.depth += p.depth;
    }
    fn sub(&mut self, p: &PipeState) {
        self.density -= p.density;
        self.color[0] -= p.color[0];
        self.color[1] -= p.color[1];
        self.color[2] -= p.color[2];
        self.normal -= p.normal;
        self.depth -= p.depth;
    }
    fn add_receptor(&mut self, o: &Receptor) {
        self.density += o.density;
        self.color[0] += o.color[0];
        self.color[1] += o.color[1];
        self.color[2] += o.color[2];
        self.normal += o.normal;
        self.depth += o.depth;
    }
}

/// What a pipe last delivered — the delta baseline. Same shape as a receptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct PipeState {
    pub density: f32,
    pub color: [f32; 3],
    pub normal: Vec3,
    pub depth: f32,
}

impl PipeState {
    fn scaled(&self, w: f32) -> PipeState {
        PipeState {
            density: self.density * w,
            color: [self.color[0] * w, self.color[1] * w, self.color[2] * w],
            normal: self.normal * w,
            depth: self.depth * w,
        }
    }
    fn minus(&self, o: &PipeState) -> PipeState {
        PipeState {
            density: self.density - o.density,
            color: [self.color[0] - o.color[0], self.color[1] - o.color[1], self.color[2] - o.color[2]],
            normal: self.normal - o.normal,
            depth: self.depth - o.depth,
        }
    }
    fn max_abs(&self) -> f32 {
        self.density.abs()
            .max(self.color[0].abs()).max(self.color[1].abs()).max(self.color[2].abs())
            .max(self.normal.abs().max_element())
            .max(self.depth.abs())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RetinaStats {
    pub pipes_total: usize,
    pub pipes_sent: usize,
    pub relinks: u64,
    pub mean_trans: f32,
    pub relink_ms: f32,
}

/// Image-space footprint of a projected axis-aligned gaussian: center,
/// eye distance, inverse 2×2 covariance (e = a·du² + 2b·du·dv + c·dv²),
/// and half-extents of the e < 4 bounding box.
struct Footprint {
    u: f32,
    v: f32,
    depth: f32,
    a: f32,
    b: f32,
    c: f32,
    hu: f32,
    hv: f32,
}

/// World point → (u right, v down, in receptor units; eye distance). None if
/// behind the eye. Matches the shader's UV convention (v=0 at the top).
fn project(vp: &Mat4, w: u32, h: u32, p: Vec3) -> Option<(f32, f32, f32)> {
    let c = *vp * Vec4::new(p.x, p.y, p.z, 1.0);
    if c.w <= 1e-4 { return None; }
    let ndc = c.truncate() / c.w;
    Some((
        (ndc.x * 0.5 + 0.5) * w as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * h as f32,
        c.w,
    ))
}

fn footprint(vp: &Mat4, w: u32, h: u32, s: &Source) -> Option<Footprint> {
    let (u, v, depth) = project(vp, w, h, s.position)?;
    let r = s.kernel_radii();
    let (mut suu, mut suv, mut svv) = (0.0f32, 0.0f32, 0.0f32);
    for axis in [Vec3::X * r.x, Vec3::Y * r.y, Vec3::Z * r.z] {
        if let Some((pu, pv, _)) = project(vp, w, h, s.position + axis) {
            let du = pu - u;
            let dv = pv - v;
            suu += du * du;
            suv += du * dv;
            svv += dv * dv;
        }
    }
    // Floor the variance at half a receptor so distant entities keep a pipe.
    let min_var = 0.25;
    suu = suu.max(min_var);
    svv = svv.max(min_var);
    let det = (suu * svv - suv * suv).max(1e-6);
    Some(Footprint {
        u, v, depth,
        a: svv / det,
        b: -suv / det,
        c: suu / det,
        hu: 2.0 * suu.sqrt(),
        hv: 2.0 * svv.sqrt(),
    })
}

/// Observer position from the view-projection: the camera center maps to
/// clip (0,0,c,0) (w = 0), so pull that direction back and divide.
pub fn eye_from_view_proj(view_proj: Mat4) -> Vec3 {
    let h = view_proj.inverse() * Vec4::new(0.0, 0.0, 1.0, 0.0);
    h.truncate() / h.w
}

pub struct Retina {
    pub width: u32,
    pub height: u32,
    pub receptors: Vec<Receptor>,
    // Pipes entity → receptor, SoA, per-entity contiguous slices.
    pipe_start: Vec<u32>,
    pipe_count: Vec<u32>,
    pipe_receptor: Vec<u32>,
    pipe_weight: Vec<f32>,
    pipe_last: Vec<PipeState>,
    /// τ toward the eye, per entity (1.0 for entities without pipes).
    entity_trans: Vec<f32>,
    /// Eye distance per entity at the last relink.
    entity_depth: Vec<f32>,
    last_view_proj: Option<Mat4>,
    pub stats: RetinaStats,
    /// Set when any delta arrived since the renderer last uploaded.
    pub dirty: bool,
}

impl Retina {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            receptors: vec![Receptor::default(); (width * height) as usize],
            pipe_start: Vec::new(),
            pipe_count: Vec::new(),
            pipe_receptor: Vec::new(),
            pipe_weight: Vec::new(),
            pipe_last: Vec::new(),
            entity_trans: Vec::new(),
            entity_depth: Vec::new(),
            last_view_proj: None,
            stats: RetinaStats::default(),
            dirty: true,
        }
    }

    /// Change resolution. Everything is rebuilt at the next tick's relink.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.clamp(MIN_RETINA_DIM, MAX_RETINA_DIM);
        let height = height.clamp(MIN_RETINA_DIM, MAX_RETINA_DIM);
        *self = Retina::new(width, height);
    }

    pub fn pipes_of(&self, i: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let (start, count) = if i < self.pipe_start.len() {
            (self.pipe_start[i] as usize, self.pipe_count[i] as usize)
        } else { (0, 0) };
        (start..start + count).map(move |k| (self.pipe_receptor[k], self.pipe_weight[k]))
    }

    pub fn transmittance(&self, i: usize) -> f32 {
        self.entity_trans.get(i).copied().unwrap_or(1.0)
    }

    pub fn depth_of(&self, i: usize) -> f32 {
        self.entity_depth.get(i).copied().unwrap_or(0.0)
    }

    /// True when the scene AABB's corners have shifted ≥ RELINK_SHIFT
    /// receptors between the last linking view and this one.
    pub fn needs_relink(&self, view_proj: Mat4, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        let Some(last) = self.last_view_proj else { return true; };
        if last == view_proj { return false; }
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { aabb_min.x } else { aabb_max.x },
                if i & 2 == 0 { aabb_min.y } else { aabb_max.y },
                if i & 4 == 0 { aabb_min.z } else { aabb_max.z },
            );
            match (project(&last, self.width, self.height, corner),
                   project(&view_proj, self.width, self.height, corner)) {
                (Some(a), Some(b)) => {
                    if (a.0 - b.0).abs().max((a.1 - b.1).abs()) >= RELINK_SHIFT { return true; }
                }
                _ => return true,
            }
        }
        false
    }

    /// Full rebuild: drop every pipe (subtracting what it last sent — the
    /// receptors are then exactly zero), re-project every drawable source,
    /// recompute τ toward the eye.
    pub fn relink(&mut self, sources: &[Source], hash: &SpatialHash, view_proj: Mat4, eye: Vec3, atten_k: f32) {
        let t0 = std::time::Instant::now();
        let (w, h) = (self.width, self.height);

        // 1. Drop. Receptors are the exact sum of pipe_last, so this lands on
        //    zero; the explicit reset below only removes float drift.
        for (k, &rc) in self.pipe_receptor.iter().enumerate() {
            self.receptors[rc as usize].sub(&self.pipe_last[k]);
        }
        debug_assert!(self.receptors.iter().all(|r| r.density.abs() < 1e-2),
            "receptors not zero after dropping all pipes");
        for r in &mut self.receptors { *r = Receptor::default(); }

        // 2–3. Project footprints.
        let fps: Vec<Option<Footprint>> = sources.par_iter()
            .map(|s| if s.drawable { footprint(&view_proj, w, h, s) } else { None })
            .collect();
        let n = sources.len();
        self.entity_depth = fps.iter().map(|f| f.as_ref().map(|f| f.depth).unwrap_or(0.0)).collect();

        // 4. Pipes with feathered gaussian weights.
        self.pipe_start = vec![0; n];
        self.pipe_count = vec![0; n];
        self.pipe_receptor.clear();
        self.pipe_weight.clear();
        for (i, fp) in fps.iter().enumerate() {
            self.pipe_start[i] = self.pipe_receptor.len() as u32;
            let Some(fp) = fp else { continue; };
            let u0 = (fp.u - fp.hu).floor().max(0.0) as i64;
            let u1 = (fp.u + fp.hu).ceil().min(w as f32 - 1.0) as i64;
            let v0 = (fp.v - fp.hv).floor().max(0.0) as i64;
            let v1 = (fp.v + fp.hv).ceil().min(h as f32 - 1.0) as i64;
            if u0 > u1 || v0 > v1 { continue; }
            for rv in v0..=v1 {
                for ru in u0..=u1 {
                    let du = ru as f32 + 0.5 - fp.u;
                    let dv = rv as f32 + 0.5 - fp.v;
                    let e = fp.a * du * du + 2.0 * fp.b * du * dv + fp.c * dv * dv;
                    if e >= 4.0 { continue; }
                    self.pipe_receptor.push(rv as u32 * w + ru as u32);
                    self.pipe_weight.push((-e).exp() * cutoff_window(e));
                }
            }
            self.pipe_count[i] = self.pipe_receptor.len() as u32 - self.pipe_start[i];
        }
        self.pipe_last = vec![PipeState::default(); self.pipe_receptor.len()];

        // 5. τ toward the eye — only for entities that actually got pipes
        // (parallel — this is the cost).
        let pipe_count = &self.pipe_count;
        self.entity_trans = (0..n).into_par_iter().map(|i| {
            if pipe_count[i] > 0 {
                segment_transmittance(sources, hash, sources[i].position, eye, &[i], atten_k)
            } else { 1.0 }
        }).collect();

        // 6. Bookkeeping.
        self.last_view_proj = Some(view_proj);
        let linked: Vec<f32> = (0..n).filter(|&i| self.pipe_count[i] > 0).map(|i| self.entity_trans[i]).collect();
        self.stats.relinks += 1;
        self.stats.pipes_total = self.pipe_receptor.len();
        self.stats.mean_trans = if linked.is_empty() { 1.0 } else { linked.iter().sum::<f32>() / linked.len() as f32 };
        self.stats.relink_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.dirty = true;
    }
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

    /// Eye at origin looking down −Z, 90° fov, aspect w/h. With odd w/h a
    /// point on the axis projects exactly onto the center receptor's center.
    pub(super) fn test_view_proj(w: u32, h: u32) -> Mat4 {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, w as f32 / h as f32, 0.1, 500.0);
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        proj * view
    }

    #[test]
    fn eye_is_recovered_from_view_proj() {
        let eye = eye_from_view_proj(test_view_proj(63, 35));
        assert!(eye.length() < 1e-3, "eye={:?}", eye);
    }

    #[test]
    fn point_source_on_axis_links_center_receptor_with_unit_weight() {
        let vp = test_view_proj(63, 35);
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, 1.0)];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        let center: u32 = 17 * 63 + 31;
        let w = r.pipes_of(0).find(|&(rc, _)| rc == center).map(|(_, w)| w);
        assert!(w.is_some(), "no pipe to the center receptor");
        assert!((w.unwrap() - 1.0).abs() < 1e-5, "center weight {}", w.unwrap());
        assert!((r.depth_of(0) - 10.0).abs() < 1e-3);
        assert!(r.pipes_of(0).count() >= 1);
    }

    #[test]
    fn gaussian_footprint_is_symmetric_and_sized_by_projection() {
        // 90° fov at distance 10: 1 cell = 35/2/10 = 1.75 receptors.
        // radii 2 → σ = 3.5 receptors, extent 2σ = 7 → ~π·7² ≈ 150 receptors.
        let vp = test_view_proj(63, 35);
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(2.0), 1.0)];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        let pipes: HashMap<u32, f32> = r.pipes_of(0).collect();
        assert!(pipes.len() > 100 && pipes.len() < 200, "footprint {} pipes", pipes.len());
        for k in 1..=5u32 {
            let l = pipes[&(17 * 63 + 31 - k)];
            let rt = pipes[&(17 * 63 + 31 + k)];
            let up = pipes[&((17 - k) * 63 + 31)];
            assert!((l - rt).abs() < 1e-5 && (l - up).abs() < 1e-5, "asymmetric at k={}", k);
        }
    }

    #[test]
    fn off_screen_and_non_drawable_sources_get_no_pipes() {
        let vp = test_view_proj(63, 35);
        let mut hidden = src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 1.0);
        hidden.drawable = false;
        let sources = vec![
            src(Vec3::new(0.0, 0.0, 10.0), Vec3::ONE, 1.0),   // behind the eye
            src(Vec3::new(100.0, 0.0, -10.0), Vec3::ONE, 1.0), // far off to the right
            hidden,
        ];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        for i in 0..3 {
            assert_eq!(r.pipes_of(i).count(), 0, "source {} got pipes", i);
            assert_eq!(r.transmittance(i), 1.0, "source {} τ not defaulted to 1.0", i);
        }
        assert_eq!(r.stats.pipes_total, 0);
    }

    #[test]
    fn occluded_source_has_low_transmittance() {
        let vp = test_view_proj(63, 35);
        let sources = vec![
            src(Vec3::new(0.0, 0.0, -20.0), Vec3::ONE, 40.0), // behind
            src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 40.0), // in front
        ];
        let hash = SpatialHash::build(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        assert!(r.transmittance(0) < 0.2, "back τ={}", r.transmittance(0));
        assert!((r.transmittance(1) - 1.0).abs() < 1e-3, "front τ={}", r.transmittance(1));
    }

    #[test]
    fn relink_trigger_follows_projected_shift() {
        let vp = test_view_proj(63, 35);
        let mut r = Retina::new(63, 35);
        let (lo, hi) = (Vec3::new(-5.0, -5.0, -15.0), Vec3::new(5.0, 5.0, -5.0));
        assert!(r.needs_relink(vp, lo, hi), "fresh retina must relink");
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 1.0)];
        r.relink(&sources, &SpatialHash::build(&sources), vp, Vec3::ZERO, ATTEN_K_DEFAULT);
        assert!(!r.needs_relink(vp, lo, hi), "same view must not relink");
        // Tiny nudge: 0.001 cells at distance 10 ≈ 0.002 receptors → no relink
        let nudge = Mat4::from_translation(Vec3::new(0.001, 0.0, 0.0));
        assert!(!r.needs_relink(vp * nudge, lo, hi));
        // 1 cell sideways ≈ 1.75 receptors → relink
        let shove = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        assert!(r.needs_relink(vp * shove, lo, hi));
    }
}
