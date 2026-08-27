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
/// Projected motion, in receptors, that forces a relink — of the scene box's
/// corners or of any linked source's center. A tenth of a receptor: below it
/// the pipes are still where the picture wants them.
pub const RELINK_SHIFT: f32 = 0.1;
pub const ATTEN_END_MARGIN: f32 = 1.5;
pub const ATTEN_SAMPLES_PER_CELL: f32 = 1.0;
pub const ATTEN_THRESHOLD: f32 = 0.6;
pub const ATTEN_K_DEFAULT: f32 = 0.5;
/// Stop integrating a segment once k·∫ passes this: τ < 2e-9, well under an
/// f16 denormal, so the rest of the walk cannot change the picture.
pub const ATTEN_TAU_CUTOFF: f32 = 20.0;
pub const MIN_RETINA_DIM: u32 = 20;
pub const MAX_RETINA_DIM: u32 = 1280;
/// Ceiling on the transient scratch images `arrive` may hold at once. Each
/// worker chunk owns a full `n_rec × size_of::<Receptor>()` image, so without
/// a bound the peak grows with the core count *and* the resolution.
pub const ARRIVE_SCRATCH_BUDGET_BYTES: usize = 64 << 20;

/// How many entity chunks `arrive` splits into: one per worker thread, minus
/// however many the scratch budget cannot pay for, and never more than there
/// are entities. Pure so the bound is testable without a retina.
pub fn arrive_chunks(n_entities: usize, n_rec: usize, threads: usize) -> usize {
    let per_scratch = n_rec * std::mem::size_of::<Receptor>();
    let affordable = if per_scratch == 0 { threads } else { ARRIVE_SCRATCH_BUDGET_BYTES / per_scratch };
    threads.max(1).min(affordable.max(1)).min(n_entities.max(1))
}

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

/// Cell edge of the occluder grid, in world cells. Small and fixed: buckets
/// hold only the occluders that can actually reach the cell.
pub const HASH_CELL: f32 = 2.0;

/// Uniform grid over occluder sources. Each occluder is inserted into every
/// cell its kernel extent box overlaps, so a single-cell lookup is exact:
/// `kernel()` is zero outside `position ± 2·kernel_radii()`, so an occluder
/// that can reach a point is in that point's cell.
pub struct SpatialHash {
    map: HashMap<(i32, i32, i32), Vec<usize>>,
    /// Widest kernel reach of any occluder, per axis. A clip box grown by this
    /// is conservative: nothing outside it can reach a sample inside it.
    max_extent: Vec3,
}

impl SpatialHash {
    pub fn build(sources: &[Source]) -> Self {
        let mut map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        let mut max_extent = Vec3::ZERO;
        for (i, s) in sources.iter().enumerate() {
            if !s.occluder { continue; }
            // kernel() > 0 requires |(p − position)/radii|² < 4 — i.e. p inside
            // the box position ± 2·radii.
            let extent = s.kernel_radii() * 2.0;
            max_extent = max_extent.max(extent);
            let lo = Self::key(s.position - extent);
            let hi = Self::key(s.position + extent);
            for cz in lo.2..=hi.2 {
                for cy in lo.1..=hi.1 {
                    for cx in lo.0..=hi.0 {
                        map.entry((cx, cy, cz)).or_default().push(i);
                    }
                }
            }
        }
        Self { map, max_extent }
    }

    fn key(p: Vec3) -> (i32, i32, i32) {
        (
            (p.x / HASH_CELL).floor() as i32,
            (p.y / HASH_CELL).floor() as i32,
            (p.z / HASH_CELL).floor() as i32,
        )
    }

    /// Σ opacity·kernel over occluders that can reach `p`, excluding indices
    /// in `skip`. One cell: coverage is exact by how `build` inserts.
    pub fn density_at(&self, sources: &[Source], p: Vec3, skip: &[usize]) -> f32 {
        let Some(bucket) = self.map.get(&Self::key(p)) else { return 0.0; };
        let mut rho = 0.0f32;
        for &j in bucket {
            if skip.contains(&j) { continue; }
            rho += sources[j].opacity * sources[j].kernel(p);
        }
        rho
    }
}

/// Parametric entry/exit of `from + dir·t`, t ∈ [0, len], through an
/// axis-aligned box. `None` when the segment never enters it.
fn clip_to_box(from: Vec3, dir: Vec3, len: f32, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let (mut t0, mut t1) = (0.0f32, len);
    for a in 0..3 {
        if dir[a].abs() < 1e-12 {
            // Parallel to this slab: either always inside it or never.
            if from[a] < bmin[a] || from[a] > bmax[a] { return None; }
            continue;
        }
        let mut ta = (bmin[a] - from[a]) / dir[a];
        let mut tb = (bmax[a] - from[a]) / dir[a];
        if ta > tb { std::mem::swap(&mut ta, &mut tb); }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 { return None; }
    }
    Some((t0, t1))
}

/// Fraction of light surviving the segment from → to:
/// exp(−k · ∫ max(0, ρ − threshold) ds), skipping ATTEN_END_MARGIN at
/// both ends so an entity's own kernel (and its partner's) never self-shadows.
///
/// `threshold` is the density that counts as "not occluding", floored at
/// ATTEN_THRESHOLD. A caller that walks from an entity buried inside other
/// occluders (the dino's receptor shell sits inside its own skeleton) raises it
/// to the density already surrounding that entity — the end margin cannot help
/// there, because the body it is buried in extends far past 1.5 cells.
///
/// Two bounds keep the cost off the segment's length. `aabb` (the scene's
/// geometry box) clips the walk to the stretch that can hold occluders at all —
/// samples outside it are zero by construction, so this changes nothing but the
/// work; the box is grown by the widest kernel reach in `hash` to stay
/// conservative. And once `k·∫` passes ATTEN_TAU_CUTOFF the segment is opaque
/// past anything the renderer can show, so the walk stops at a hard zero.
pub fn segment_transmittance(
    sources: &[Source],
    hash: &SpatialHash,
    from: Vec3,
    to: Vec3,
    skip: &[usize],
    k: f32,
    threshold: f32,
    aabb: Option<(Vec3, Vec3)>,
) -> f32 {
    let threshold = threshold.max(ATTEN_THRESHOLD);
    let ab = to - from;
    let len = ab.length();
    if len - 2.0 * ATTEN_END_MARGIN <= 0.0 { return 1.0; }
    let dir = ab / len;
    let step = 1.0 / ATTEN_SAMPLES_PER_CELL;
    let (mut lo, mut hi) = (ATTEN_END_MARGIN, len - ATTEN_END_MARGIN);
    if let Some((bmin, bmax)) = aabb {
        let Some((t0, t1)) = clip_to_box(from, dir, len, bmin - hash.max_extent, bmax + hash.max_extent)
            else { return 1.0; };
        lo = lo.max(t0);
        hi = hi.min(t1);
        if lo >= hi { return 1.0; }
    }
    // The sample lattice stays anchored at `from` — t_m = margin + (m+½)·step —
    // so clipping only drops samples, never moves the surviving ones.
    let m0 = ((lo - ATTEN_END_MARGIN) / step - 0.5).ceil().max(0.0);
    let mut integral = 0.0f32;
    let mut t = ATTEN_END_MARGIN + (m0 + 0.5) * step;
    while t < hi {
        let rho = hash.density_at(sources, from + dir * t, skip);
        integral += (rho - threshold).max(0.0) * step;
        if k * integral > ATTEN_TAU_CUTOFF { return 0.0; }
        t += step;
    }
    (-k * integral).exp()
}

/// Persistent state of one image-plane cell. Sums, never averages: the
/// renderer divides by density at upload. Never decays; only deltas touch it.
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
    /// Withdraw what a pipe last delivered. Only `relink`'s debug-build check
    /// that the receptors really are the sum of the pipes uses it.
    #[cfg(debug_assertions)]
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
    /// Projected center (u, v) per entity at the last relink, NaN for the
    /// entities that had no footprint. The baseline `needs_relink` compares
    /// against to catch a source that moves under a motionless camera.
    last_centers: Vec<(f32, f32)>,
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
            last_centers: Vec::new(),
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

    /// Test/diagnostic API: the retina itself never reads its pipes back, but
    /// `log_stats`' debug self-check and the exactness tests do.
    #[allow(dead_code)]
    pub fn pipes_of(&self, i: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let (start, count) = if i < self.pipe_start.len() {
            (self.pipe_start[i] as usize, self.pipe_count[i] as usize)
        } else { (0, 0) };
        (start..start + count).map(move |k| (self.pipe_receptor[k], self.pipe_weight[k]))
    }

    /// Test/diagnostic API.
    #[allow(dead_code)]
    pub fn transmittance(&self, i: usize) -> f32 {
        self.entity_trans.get(i).copied().unwrap_or(1.0)
    }

    /// Test/diagnostic API.
    #[allow(dead_code)]
    pub fn depth_of(&self, i: usize) -> f32 {
        self.entity_depth.get(i).copied().unwrap_or(0.0)
    }

    /// True when the picture's geometry has moved on the image plane since the
    /// last relink — because the *camera* moved (the scene AABB's corners
    /// project ≥ RELINK_SHIFT receptors away from where they did) or because a
    /// *source* moved (its center projects ≥ RELINK_SHIFT receptors from where
    /// it sat when it was linked).
    ///
    /// The second half is not redundant: pipes are frozen between relinks, so
    /// an animating scene under a motionless camera is a still image until the
    /// trigger notices the sources themselves.
    pub fn needs_relink(&self, sources: &[Source], view_proj: Mat4, aabb_min: Vec3, aabb_max: Vec3) -> bool {
        let Some(last) = self.last_view_proj else { return true; };
        if last != view_proj {
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
        }
        // The entity set itself changed shape — nothing to compare against.
        if self.last_centers.len() != sources.len() { return true; }
        for (i, s) in sources.iter().enumerate() {
            let (lu, lv) = self.last_centers[i];
            // NaN: this source had no footprint at the last relink, so it has
            // no pipes to be stale. Whether it draws is `contribution`'s call.
            if !s.drawable || lu.is_nan() { continue; }
            let Some((u, v, _)) = project(&view_proj, self.width, self.height, s.position) else { return true; };
            if (u - lu).abs().max((v - lv).abs()) >= RELINK_SHIFT { return true; }
        }
        false
    }

    /// Full rebuild: drop every pipe (subtracting what it last sent — the
    /// receptors are then exactly zero), re-project every drawable source,
    /// recompute τ toward the eye. `aabb_min/max` bound the scene's geometry,
    /// which is the only stretch of an entity→eye segment that can attenuate.
    pub fn relink(&mut self, sources: &[Source], hash: &SpatialHash, view_proj: Mat4, eye: Vec3,
                  aabb_min: Vec3, aabb_max: Vec3, atten_k: f32) {
        let t0 = std::time::Instant::now();
        let (w, h) = (self.width, self.height);

        // 1. Drop every pipe. Receptors are the exact sum of pipe_last, so
        //    withdrawing all of it lands on zero — which is what the reset
        //    below writes regardless. So only debug builds pay for the scatter;
        //    they pay it to check that invariant, which is the whole point.
        #[cfg(debug_assertions)]
        {
            for (k, &rc) in self.pipe_receptor.iter().enumerate() {
                self.receptors[rc as usize].sub(&self.pipe_last[k]);
            }
            debug_assert!(self.receptors.iter().all(|r| r.density.abs() < 1e-2),
                "receptors not zero after dropping all pipes");
        }
        for r in &mut self.receptors { *r = Receptor::default(); }

        // 2–3. Project footprints.
        let fps: Vec<Option<Footprint>> = sources.par_iter()
            .map(|s| if s.drawable { footprint(&view_proj, w, h, s) } else { None })
            .collect();
        let n = sources.len();
        self.entity_depth = fps.iter().map(|f| f.as_ref().map(|f| f.depth).unwrap_or(0.0)).collect();
        self.last_centers = fps.iter()
            .map(|f| f.as_ref().map(|f| (f.u, f.v)).unwrap_or((f32::NAN, f32::NAN)))
            .collect();

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
        let aabb = Some((aabb_min, aabb_max));
        self.entity_trans = (0..n).into_par_iter().map(|i| {
            if pipe_count[i] > 0 {
                // Relative threshold: an entity is not occluded by what it is
                // already buried in. ρ_self is the density of *other* occluders
                // at the entity's own position — for a receptor on the dino's
                // shell that is its own skeleton, 4–8 deep. Only density above
                // it (the body behind, the floor under) can dim the entity.
                let rho_self = hash.density_at(sources, sources[i].position, &[i]);
                let threshold = rho_self.max(ATTEN_THRESHOLD);
                segment_transmittance(sources, hash, sources[i].position, eye, &[i], atten_k, threshold, aabb)
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

    /// What entity `i` offers its pipes this tick (before the footprint
    /// weight). A source that stopped drawing offers zero, so its pipes
    /// withdraw what they last sent.
    fn contribution(&self, i: usize, s: &Source) -> PipeState {
        if !s.drawable { return PipeState::default(); }
        let d = s.density * self.entity_trans[i];
        PipeState {
            density: d,
            color: [s.color[0] * self.entity_trans[i], s.color[1] * self.entity_trans[i], s.color[2] * self.entity_trans[i]],
            normal: s.normal * d,
            depth: self.entity_depth[i] * d,
        }
    }

    /// Phase 3′: every pipe sends `new − last` if it exceeds DELTA_EPS.
    /// Parallel over entities; receptors are shared by many entities, so a
    /// chunk of entities accumulates into a scratch image of its own. The
    /// chunking is by worker thread, not by rayon's adaptive splitting: a
    /// scratch image is the size of the retina, so the number of them (and of
    /// full-image adds at the end) must not grow with the entity count — and
    /// `arrive_chunks` caps it again so the peak does not grow with the
    /// resolution × core count product either.
    pub fn arrive(&mut self, sources: &[Source]) -> usize {
        let n_rec = self.receptors.len();
        let n = sources.len().min(self.pipe_count.len());

        // Contributions first (needs &self), then the per-entity mutable views
        // of pipe_last (contiguous, ascending) — direct field borrows, no unsafe.
        let contribs: Vec<PipeState> = (0..n).map(|i| self.contribution(i, &sources[i])).collect();
        let mut slices: Vec<&mut [PipeState]> = Vec::with_capacity(n);
        let mut rest: &mut [PipeState] = &mut self.pipe_last;
        for i in 0..n {
            let (head, tail) = rest.split_at_mut(self.pipe_count[i] as usize);
            slices.push(head);
            rest = tail;
        }

        let pipe_start = &self.pipe_start;
        let pipe_receptor = &self.pipe_receptor;
        let pipe_weight = &self.pipe_weight;

        // One contiguous entity range per chunk — one chunk per worker thread,
        // fewer when a full scratch image each would blow ARRIVE_SCRATCH_BUDGET_BYTES.
        // The scratch image is allocated on the range's first delta, so a
        // settled scene allocates nothing and adds nothing.
        let chunk_len = n.div_ceil(arrive_chunks(n, n_rec, rayon::current_num_threads())).max(1);
        let parts: Vec<(Option<Vec<Receptor>>, usize)> = slices.par_chunks_mut(chunk_len)
            .enumerate()
            .map(|(c, chunk)| {
                let mut scratch: Option<Vec<Receptor>> = None;
                let mut sent = 0usize;
                for (off_i, last) in chunk.iter_mut().enumerate() {
                    let i = c * chunk_len + off_i;
                    let start = pipe_start[i] as usize;
                    for (off, l) in last.iter_mut().enumerate() {
                        let k = start + off;
                        let new = contribs[i].scaled(pipe_weight[k]);
                        let delta = new.minus(l);
                        if delta.max_abs() > DELTA_EPS {
                            let acc = scratch.get_or_insert_with(|| vec![Receptor::default(); n_rec]);
                            acc[pipe_receptor[k] as usize].add(&delta);
                            *l = new;
                            sent += 1;
                        }
                    }
                }
                (scratch, sent)
            })
            .collect();

        // Merge the ≤k scratches into the image, split by receptor range so
        // every thread owns disjoint receptors — the adds are the same, and in
        // the same order per receptor, as adding the scratches one by one.
        let sent: usize = parts.iter().map(|(_, s)| *s).sum();
        let scratches: Vec<&[Receptor]> = parts.iter().filter_map(|(s, _)| s.as_deref()).collect();
        if !scratches.is_empty() {
            let rec_chunk = n_rec.div_ceil(rayon::current_num_threads().max(1)).max(1);
            self.receptors.par_chunks_mut(rec_chunk).enumerate().for_each(|(c, dst)| {
                let (base, len) = (c * rec_chunk, dst.len());
                for s in &scratches {
                    for (r, x) in dst.iter_mut().zip(&s[base..base + len]) { r.add_receptor(x); }
                }
            });
        }
        if sent > 0 { self.dirty = true; }
        self.stats.pipes_sent = sent;
        sent
    }

    /// One retina step: relink if the view or the links moved, then arrive.
    pub fn tick(&mut self, sources: &[Source], view_proj: Mat4, aabb_min: Vec3, aabb_max: Vec3, force_relink: bool, atten_k: f32) {
        if force_relink || self.needs_relink(sources, view_proj, aabb_min, aabb_max) {
            let hash = SpatialHash::build(sources);
            let eye = eye_from_view_proj(view_proj);
            self.relink(sources, &hash, view_proj, eye, aabb_min, aabb_max, atten_k);
        }
        self.arrive(sources);
    }

    /// Reference image: Σ over pipes of contribution·weight, from scratch.
    /// The incremental receptors must equal this (up to DELTA_EPS per pipe).
    #[allow(dead_code)] // test/diagnostic API — release builds skip the self-check
    pub fn direct_sum(&self, sources: &[Source]) -> Vec<Receptor> {
        let mut out = vec![Receptor::default(); self.receptors.len()];
        for i in 0..sources.len().min(self.pipe_count.len()) {
            let c = self.contribution(i, &sources[i]);
            for (rc, w) in self.pipes_of(i) {
                out[rc as usize].add(&c.scaled(w));
            }
        }
        out
    }

    pub fn log_stats(&self, sources: &[Source]) {
        let above = self.receptors.iter().filter(|r| r.density >= RETINA_ISO).count();
        let max_d = self.receptors.iter().map(|r| r.density).fold(0.0f32, f32::max);
        log::info!(
            "Retina {}x{}: {} receptors ≥ iso ({:.1}%), max density {:.2}; pipes {} total / {} sent last tick; mean τ {:.3}; {} relinks (last {:.2} ms)",
            self.width, self.height, above,
            100.0 * above as f64 / self.receptors.len().max(1) as f64, max_d,
            self.stats.pipes_total, self.stats.pipes_sent, self.stats.mean_trans,
            self.stats.relinks, self.stats.relink_ms,
        );

        // The incremental image is only ever as good as the claim that it
        // equals the sum of the pipes. Debug builds check it against the
        // from-scratch reference whenever the diagnostics key is pressed;
        // anything past a pipe's worth of DELTA_EPS means deltas have drifted.
        #[cfg(debug_assertions)]
        {
            let want = self.direct_sum(sources);
            let dev = self.receptors.iter().zip(&want)
                .map(|(got, want)| {
                    (got.density - want.density).abs()
                        .max((0..3).map(|c| (got.color[c] - want.color[c]).abs()).fold(0.0, f32::max))
                        .max((got.normal - want.normal).abs().max_element())
                        .max((got.depth - want.depth).abs())
                })
                .fold(0.0f32, f32::max);
            log::info!("Retina self-check: max |receptor − direct_sum| = {:.2e}", dev);
        }
        #[cfg(not(debug_assertions))]
        let _ = sources;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scene AABB the way `advance_entities` builds it: every source's
    /// position grown by its full kernel reach.
    fn box_of(sources: &[Source]) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for s in sources {
            lo = lo.min(s.position - s.kernel_radii() * 2.0);
            hi = hi.max(s.position + s.kernel_radii() * 2.0);
        }
        (lo, hi)
    }

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
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, &[], 0.5, ATTEN_THRESHOLD, None);
        assert!((t - 1.0).abs() < 1e-6, "off-segment occluder attenuated: {}", t);
    }

    #[test]
    fn transmittance_matches_fine_quadrature_for_centered_occluder() {
        let occ = src(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 10.0);
        let sources = vec![occ];
        let hash = SpatialHash::build(&sources);
        let k = 0.5;
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, &[], k, ATTEN_THRESHOLD, None);
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
            a.position, b.position, &[], 0.5, ATTEN_THRESHOLD, None);
        assert!((t_ends - 1.0).abs() < 1e-6, "endpoint sources leaked into the integral: {}", t_ends);
        // Skipping the middle occluder restores full transmittance
        let t_skip = segment_transmittance(&sources, &hash, a.position, b.position, &[2], 0.5, ATTEN_THRESHOLD, None);
        assert!((t_skip - 1.0).abs() < 1e-6);
        let t_block = segment_transmittance(&sources, &hash, a.position, b.position, &[], 0.5, ATTEN_THRESHOLD, None);
        assert!(t_block < 0.1);
    }

    /// The clip must be free of consequence: samples outside the geometry AABB
    /// are zero anyway, so an eye 500 cells out must read exactly what an eye
    /// 20 cells out reads. (Both distances differ by a whole number of steps,
    /// so the sample lattice lands on the same world points.)
    #[test]
    fn aabb_clip_does_not_change_tau_for_a_distant_eye() {
        let entity = Vec3::ZERO;
        let occ = src(Vec3::new(0.0, 0.0, 6.0), Vec3::ONE, 10.0);
        let sources = vec![occ];
        let hash = SpatialHash::build(&sources);
        // Scene AABB holds every solid — entity and occluder both.
        let aabb = Some((Vec3::new(-8.0, -8.0, -8.0), Vec3::new(8.0, 8.0, 12.0)));
        let near = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, 20.0), entity, &[], 0.5, ATTEN_THRESHOLD, aabb);
        let far = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, 500.0), entity, &[], 0.5, ATTEN_THRESHOLD, aabb);
        assert!(near < 0.9, "the occluder did not attenuate at all: {}", near);
        assert!((near - far).abs() < 1e-6, "clipped far τ={} != near τ={}", far, near);
        // And the clip agrees with integrating the whole 500-cell segment.
        let unclipped = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, 500.0), entity, &[], 0.5, ATTEN_THRESHOLD, None);
        assert!((far - unclipped).abs() < 1e-6, "clip changed τ: {} vs {}", far, unclipped);
    }

    #[test]
    fn aabb_clip_skips_a_segment_that_misses_the_box() {
        let sources = vec![src(Vec3::new(0.0, 0.0, 6.0), Vec3::ONE, 10.0)];
        let hash = SpatialHash::build(&sources);
        // A box far off to the side: the segment never enters it, so nothing
        // on it can attenuate and τ is 1.0 without sampling at all.
        let aabb = Some((Vec3::new(400.0, 400.0, 400.0), Vec3::new(420.0, 420.0, 420.0)));
        let t = segment_transmittance(&sources, &hash, Vec3::new(0.0, 0.0, 20.0), Vec3::ZERO, &[], 0.5, ATTEN_THRESHOLD, aabb);
        assert_eq!(t, 1.0);
    }

    /// Past k·∫ = 20 the survivor fraction is < 2e-9 — under f16 denormals and
    /// under anything the renderer can show. The walk stops and returns a hard
    /// zero, and that must be a real early-out, not `exp` underflowing.
    #[test]
    fn opaque_segment_early_outs_to_exactly_zero() {
        let sources = vec![src(Vec3::new(0.0, 0.0, 0.0), Vec3::splat(4.0), 12.0)];
        let hash = SpatialHash::build(&sources);
        let (from, to) = (Vec3::new(0.0, 0.0, 20.0), Vec3::new(0.0, 0.0, -20.0));
        // Measure the integral itself with a k too small to trip the early-out.
        let probe = segment_transmittance(&sources, &hash, from, to, &[], 1e-3, ATTEN_THRESHOLD, None);
        let integral = -probe.ln() / 1e-3;
        assert!(integral > 25.0, "test occluder too thin to reach the cutoff: ∫={}", integral);
        // Pick k so k·∫ = 25: past the cutoff, but exp(−25) is a perfectly
        // representable 1.4e-11 — so a plain exp would NOT give zero here.
        let k = 25.0 / integral;
        assert!((-25.0f32).exp() > 0.0, "exp(−25) underflowed; the test proves nothing");
        let t = segment_transmittance(&sources, &hash, from, to, &[], k, ATTEN_THRESHOLD, None);
        assert_eq!(t, 0.0, "early-out did not fire at k·∫≈25 (τ={})", t);
        // Just under the cutoff the normal path still returns a positive τ.
        let under = segment_transmittance(&sources, &hash, from, to, &[], 19.0 / integral, ATTEN_THRESHOLD, None);
        assert!(under > 0.0 && under < 1e-8, "τ just under the cutoff: {}", under);
    }

    /// The threshold `relink` gives an entity's own segment: only density
    /// *above* what already surrounds it can occlude it.
    fn rel_threshold(sources: &[Source], hash: &SpatialHash, i: usize) -> f32 {
        hash.density_at(sources, sources[i].position, &[i]).max(ATTEN_THRESHOLD)
    }

    /// The dino's receptor shell sits inside its own skeleton metaballs, so
    /// ρ_others at a shell entity is already 4–8. A shell facing the eye must
    /// still arrive: nothing between it and the eye is denser than the body it
    /// is already buried in.
    #[test]
    fn relative_threshold_lets_a_front_shell_see_the_eye() {
        let body = src(Vec3::new(0.0, 0.0, -14.0), Vec3::splat(4.0), 8.0);
        let shell = src(Vec3::new(0.0, 0.0, -11.0), Vec3::ONE, 1.0); // e = (3/4)²
        let sources = vec![body, shell];
        let hash = SpatialHash::build(&sources);
        let thr = rel_threshold(&sources, &hash, 1);
        assert!(thr > 4.0, "test shell is not inside the body: ρ_self={}", thr);
        let t = segment_transmittance(&sources, &hash, sources[1].position, Vec3::ZERO, &[1],
            ATTEN_K_DEFAULT, thr, None);
        assert!(t >= 0.9, "front shell occluded by its own body: τ={}", t);
    }

    /// The relative threshold must not make the body transparent: a shell on
    /// the far side still crosses the core, where ρ ≫ ρ_self.
    #[test]
    fn relative_threshold_still_occludes_a_back_shell() {
        let body = src(Vec3::new(0.0, 0.0, -14.0), Vec3::splat(4.0), 8.0);
        let shell = src(Vec3::new(0.0, 0.0, -17.0), Vec3::ONE, 1.0); // symmetric, behind
        let sources = vec![body, shell];
        let hash = SpatialHash::build(&sources);
        let thr = rel_threshold(&sources, &hash, 1);
        let t = segment_transmittance(&sources, &hash, sources[1].position, Vec3::ZERO, &[1],
            ATTEN_K_DEFAULT, thr, None);
        assert!(t <= 0.05, "back shell shines through the body: τ={}", t);
    }

    /// Floor far below the dino has no body around it, so its threshold is the
    /// absolute floor — and the body straight above it still blocks the eye.
    #[test]
    fn relative_threshold_still_occludes_the_floor_under_a_body() {
        let body = src(Vec3::new(0.0, 0.0, -14.0), Vec3::splat(4.0), 8.0);
        let floor = src(Vec3::new(0.0, -12.0, -14.0), Vec3::ZERO, 1.0);
        let sources = vec![body, floor];
        let hash = SpatialHash::build(&sources);
        let thr = rel_threshold(&sources, &hash, 1);
        assert_eq!(thr, ATTEN_THRESHOLD, "floor is not clear of the body: ρ_self={}", thr);
        let eye = Vec3::new(0.0, 12.0, -14.0); // straight above, through the body
        let t = segment_transmittance(&sources, &hash, sources[1].position, eye, &[1],
            ATTEN_K_DEFAULT, thr, None);
        assert!(t <= 0.05, "floor under the body is not shadowed: τ={}", t);
    }

    #[test]
    fn spatial_hash_covers_the_whole_kernel_extent_of_a_big_source() {
        // Radii 8 → the kernel is non-zero out to |d| = 2·8 = 16 cells.
        let sources = vec![src(Vec3::ZERO, Vec3::splat(8.0), 3.0)];
        let hash = SpatialHash::build(&sources);
        // e = (15/8)² ≈ 3.52 < 4 → inside the extent, must be found.
        let inside = hash.density_at(&sources, Vec3::new(15.0, 0.0, 0.0), &[]);
        assert!(inside > 0.0, "big-radius source missed inside its extent: {}", inside);
        assert!((inside - 3.0 * sources[0].kernel(Vec3::new(15.0, 0.0, 0.0))).abs() < 1e-6);
        // e = (17/8)² ≈ 4.52 ≥ 4 → past the cutoff, exactly zero.
        assert_eq!(hash.density_at(&sources, Vec3::new(17.0, 0.0, 0.0), &[]), 0.0);
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
    fn eye_is_recovered_from_a_non_origin_view_proj() {
        // The real scene's camera never sits at the origin: recover an eye
        // 310 cells out along +Z, looking back at the field center.
        let want = Vec3::new(256.0, 256.0, 310.0);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 2000.0);
        let view = Mat4::look_at_rh(want, Vec3::new(256.0, 256.0, 256.0), Vec3::Y);
        let eye = eye_from_view_proj(proj * view);
        assert!((eye - want).length() < 1e-2, "eye={:?} want={:?}", eye, want);
    }

    #[test]
    fn point_source_on_axis_links_center_receptor_with_unit_weight() {
        let vp = test_view_proj(63, 35);
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, 1.0)];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
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
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
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
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
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
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(63, 35);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        assert!(r.transmittance(0) < 0.2, "back τ={}", r.transmittance(0));
        assert!((r.transmittance(1) - 1.0).abs() < 1e-3, "front τ={}", r.transmittance(1));
    }

    #[test]
    fn relink_trigger_follows_projected_shift() {
        let vp = test_view_proj(63, 35);
        let mut r = Retina::new(63, 35);
        let (lo, hi) = (Vec3::new(-5.0, -5.0, -15.0), Vec3::new(5.0, 5.0, -5.0));
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 1.0)];
        assert!(r.needs_relink(&sources, vp, lo, hi), "fresh retina must relink");
        r.relink(&sources, &SpatialHash::build(&sources), vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        assert!(!r.needs_relink(&sources, vp, lo, hi), "same view must not relink");
        // Tiny nudge: 0.001 cells at distance 10 ≈ 0.002 receptors → no relink
        let nudge = Mat4::from_translation(Vec3::new(0.001, 0.0, 0.0));
        assert!(!r.needs_relink(&sources, vp * nudge, lo, hi));
        // 1 cell sideways ≈ 1.75 receptors → relink
        let shove = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        assert!(r.needs_relink(&sources, vp * shove, lo, hi));
    }

    /// Pipes are fixed at the last relink, so a source that moves under a
    /// motionless camera is frozen on screen until the next one. The trigger
    /// has to watch the sources, not just the view.
    #[test]
    fn relink_trigger_follows_a_source_that_moves_under_a_still_camera() {
        let vp = test_view_proj(63, 35);
        let mut r = Retina::new(63, 35);
        let mut sources = vec![
            src(Vec3::new(0.0, 0.0, -10.0), Vec3::ONE, 1.0),
            src(Vec3::new(2.0, 1.0, -12.0), Vec3::ONE, 1.0),
        ];
        let (lo, hi) = box_of(&sources);
        r.relink(&sources, &SpatialHash::build(&sources), vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        assert!(!r.needs_relink(&sources, vp, lo, hi), "nothing moved but a relink was asked for");
        // One walker step: 0.1 cells at distance 10 ≈ 0.17 receptors ≥ RELINK_SHIFT.
        sources[1].position.x += 0.1;
        assert!(r.needs_relink(&sources, vp, lo, hi), "a moving source did not trigger a relink");
        // Relinking re-baselines it, and the new position is then quiet again.
        r.relink(&sources, &SpatialHash::build(&sources), vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        assert!(!r.needs_relink(&sources, vp, lo, hi));
        // Sub-threshold drift still costs nothing.
        sources[1].position.x += 0.01;
        assert!(!r.needs_relink(&sources, vp, lo, hi), "0.01 cells ≈ 0.017 receptors must not relink");
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX >> 1) as f32
    }

    fn scene() -> Vec<Source> {
        vec![
            src(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(2.0), 5.0),
            src(Vec3::new(3.0, 1.0, -12.0), Vec3::new(1.0, 2.0, 1.0), 3.0),
            src(Vec3::new(-4.0, -2.0, -9.0), Vec3::ONE, 8.0),
            src(Vec3::new(1.0, -1.0, -20.0), Vec3::splat(3.0), 2.0),
            src(Vec3::new(0.0, 4.0, -15.0), Vec3::ZERO, 1.0),
        ]
    }

    fn assert_receptors_match(r: &Retina, sources: &[Source], tol: f32) {
        let want = r.direct_sum(sources);
        for (i, (got, want)) in r.receptors.iter().zip(&want).enumerate() {
            assert!((got.density - want.density).abs() < tol, "receptor {} density {} vs {}", i, got.density, want.density);
            for c in 0..3 {
                assert!((got.color[c] - want.color[c]).abs() < tol, "receptor {} color[{}]", i, c);
            }
            assert!((got.normal - want.normal).length() < tol, "receptor {} normal", i);
            assert!((got.depth - want.depth).abs() < tol * 30.0, "receptor {} depth", i);
        }
    }

    #[test]
    fn receptors_equal_direct_sum_after_random_updates() {
        let vp = test_view_proj(63, 35);
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let mut sources = scene();
        let mut r = Retina::new(63, 35);
        let mut seed = 7u64;
        for _ in 0..20 {
            for s in &mut sources {
                s.density = 1.0 + 40.0 * lcg(&mut seed);
                s.color = [lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)];
            }
            r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
            assert_receptors_match(&r, &sources, 1e-2);
        }
        assert_eq!(r.stats.relinks, 1, "static view must link exactly once");
        // Smoke the diagnostics path, including the debug-only self-check that
        // recomputes direct_sum — it must not panic or index out of range.
        r.log_stats(&sources);
    }

    #[test]
    fn settled_scene_sends_nothing() {
        let vp = test_view_proj(63, 35);
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let sources = scene();
        let mut r = Retina::new(63, 35);
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert!(r.stats.pipes_sent > 0);
        r.dirty = false;
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert_eq!(r.stats.pipes_sent, 0, "settled scene still sends deltas");
        assert!(!r.dirty);
    }

    #[test]
    fn relink_keeps_receptors_exact() {
        let vp_a = test_view_proj(63, 35);
        let vp_b = vp_a * Mat4::from_translation(Vec3::new(2.0, 0.5, 0.0));
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let sources = scene();
        let mut r = Retina::new(63, 35);
        r.tick(&sources, vp_a, lo, hi, false, ATTEN_K_DEFAULT);
        let before: Vec<Receptor> = r.receptors.clone();
        r.tick(&sources, vp_b, lo, hi, false, ATTEN_K_DEFAULT);
        assert_eq!(r.stats.relinks, 2);
        assert_ne!(before, r.receptors, "view moved but the image did not");
        assert_receptors_match(&r, &sources, 1e-2);
    }

    #[test]
    fn source_that_stops_drawing_is_withdrawn() {
        let vp = test_view_proj(63, 35);
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let mut sources = scene();
        let mut r = Retina::new(63, 35);
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        sources[0].drawable = false; // culled between relinks (e.g. trie depth)
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert_receptors_match(&r, &sources, 1e-2);
    }

    #[test]
    fn arrive_scratch_stays_inside_the_memory_budget() {
        let bytes = std::mem::size_of::<Receptor>();
        assert_eq!(bytes, 32, "Receptor grew — the scratch budget math assumes 32 B");
        // A 1280×720 retina (the new MAX_RETINA_DIM aspect) is 921_600
        // receptors ≈ 29.5 MB of scratch each; 64 MB buys two.
        let n_rec = 1280 * 720;
        let affordable = ARRIVE_SCRATCH_BUDGET_BYTES / (n_rec * bytes);
        assert_eq!(affordable, 2);
        for threads in [1, 2, 4, 22, 128] {
            let c = arrive_chunks(50_000, n_rec, threads);
            assert!(c <= affordable, "{} chunks at {} threads exceeds {}", c, threads, affordable);
            assert!(c >= 1);
        }
        // A retina small enough that every worker can afford a scratch gets
        // one chunk per thread — the bound must not cost parallelism.
        assert_eq!(arrive_chunks(50_000, (RETINA_W * RETINA_H) as usize, 22), 22);
        // Never more chunks than entities, and never zero.
        assert_eq!(arrive_chunks(3, 64, 22), 3);
        assert_eq!(arrive_chunks(0, 64, 22), 1);
        // Even a retina too big for a single scratch still gets one chunk.
        assert_eq!(arrive_chunks(50_000, ARRIVE_SCRATCH_BUDGET_BYTES, 22), 1);
        // The clamp that keeps it reachable at all.
        assert_eq!(MAX_RETINA_DIM, 1280);
    }

    #[test]
    fn resolution_change_preserves_mean_density() {
        let (lo, hi) = (Vec3::new(-10.0, -10.0, -25.0), Vec3::new(10.0, 10.0, -5.0));
        let sources = vec![src(Vec3::new(0.0, 0.0, -10.0), Vec3::splat(3.0), 10.0)];
        let mean = |w: u32, h: u32| {
            let mut r = Retina::new(w, h);
            r.tick(&sources, test_view_proj(w, h), lo, hi, false, ATTEN_K_DEFAULT);
            r.receptors.iter().map(|x| x.density).sum::<f32>() / (w * h) as f32
        };
        let (coarse, fine) = (mean(63, 35), mean(126, 70));
        assert!((coarse - fine).abs() < 0.2 * fine, "coarse {} vs fine {}", coarse, fine);
        let mut r = Retina::new(63, 35);
        r.resize(126, 70);
        assert_eq!(r.receptors.len(), 126 * 70);
        r.resize(1, 1);
        assert_eq!((r.width, r.height), (MIN_RETINA_DIM, MIN_RETINA_DIM));
    }
}
