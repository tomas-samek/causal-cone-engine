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
    /// This source's `position`, `radii` and `opacity` are the same every tick,
    /// so the spatial hash may keep its cell insertions across relinks. Only
    /// the sources that actually move — the walker — pay to be re-hashed.
    pub is_static: bool,
    /// Creature, not scenery: the shader gives it reptile scales. Carried as a
    /// real flag because the colour heuristic it replaced ("green enough")
    /// started matching the lit floor.
    pub skin: bool,
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

type CellMap = HashMap<(i32, i32, i32), Vec<usize>>;

/// Uniform grid over occluder sources. Each occluder is inserted into every
/// cell its kernel extent box overlaps, so a single-cell lookup is exact:
/// `kernel()` is zero outside `position ± 2·kernel_radii()`, so an occluder
/// that can reach a point is in that point's cell.
///
/// Split in two by `Source::is_static`. The scene is overwhelmingly furniture —
/// floor, rock, sun — that occupies the same cells forever, and only the walker
/// moves. Hashing the furniture once and re-hashing only the walker each relink
/// is the whole point of the split; a lookup reads both cells and sums them, so
/// it is exactly the density one combined map would have given.
#[derive(Default)]
pub struct SpatialHash {
    static_map: CellMap,
    dynamic_map: CellMap,
    static_extent: Vec3,
    dynamic_extent: Vec3,
    /// Widest kernel reach of any occluder in *either* map, per axis. A clip box
    /// grown by this is conservative: nothing outside it can reach a sample
    /// inside it.
    max_extent: Vec3,
    /// The static set the static map was built from: (source count, static
    /// occluder count). A change means the indices in `static_map` no longer
    /// name what they used to, so the map is rebuilt.
    static_key: Option<(usize, usize)>,
    /// How many times the static map has been (re)built. Diagnostic only —
    /// the incremental-hash test watches it to prove the reuse is real.
    static_builds: u64,
}

impl SpatialHash {
    /// One-shot hash of a whole source set. Same split as `update`, so a fresh
    /// build and an updated hash read identically.
    pub fn build(sources: &[Source]) -> Self {
        let mut hash = Self::default();
        hash.update(sources);
        hash
    }

    /// Re-hash for this tick's source positions. The static map survives
    /// untouched unless the source set itself changed shape; the dynamic map is
    /// rebuilt every time.
    ///
    /// The contract `is_static` carries is that such a source never moves and
    /// never changes radii or opacity. A source that breaks it goes stale in the
    /// static map, silently.
    pub fn update(&mut self, sources: &[Source]) {
        let key = (sources.len(), sources.iter().filter(|s| s.occluder && s.is_static).count());
        if self.static_key != Some(key) {
            self.static_map.clear();
            self.static_extent = Self::insert(&mut self.static_map, sources, true);
            self.static_key = Some(key);
            self.static_builds += 1;
        }
        self.dynamic_map.clear();
        self.dynamic_extent = Self::insert(&mut self.dynamic_map, sources, false);
        self.max_extent = self.static_extent.max(self.dynamic_extent);
    }

    /// Insert every occluder whose `is_static` matches `want_static` into every
    /// cell its kernel extent box overlaps. Returns the widest extent inserted.
    fn insert(map: &mut CellMap, sources: &[Source], want_static: bool) -> Vec3 {
        let mut max_extent = Vec3::ZERO;
        for (i, s) in sources.iter().enumerate() {
            if !s.occluder || s.is_static != want_static { continue; }
            // kernel() > 0 requires |(p − position)/radii|² < 4 — i.e. p inside
            // the box position ± 2·radii.
            let extent = s.kernel_radii() * KERNEL_EXTENT;
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
        max_extent
    }

    fn key(p: Vec3) -> (i32, i32, i32) {
        (
            (p.x / HASH_CELL).floor() as i32,
            (p.y / HASH_CELL).floor() as i32,
            (p.z / HASH_CELL).floor() as i32,
        )
    }

    /// Widest kernel reach of any occluder in the hash, per axis.
    #[allow(dead_code)] // test/diagnostic API — `segment_transmittance` reads the field
    pub fn max_extent(&self) -> Vec3 { self.max_extent }

    /// How many times the static map has been built. Test/diagnostic API.
    #[allow(dead_code)]
    pub fn static_builds(&self) -> u64 { self.static_builds }

    /// Σ opacity·kernel over occluders that can reach `p`, excluding indices
    /// in `skip`. One cell per map: coverage is exact by how `insert` inserts,
    /// and the two maps partition the occluders, so the sum is the total.
    pub fn density_at(&self, sources: &[Source], p: Vec3, skip: &[usize]) -> f32 {
        let cell = Self::key(p);
        let mut rho = 0.0f32;
        for map in [&self.static_map, &self.dynamic_map] {
            let Some(bucket) = map.get(&cell) else { continue; };
            for &j in bucket {
                if skip.contains(&j) { continue; }
                rho += sources[j].opacity * sources[j].kernel(p);
            }
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
    /// Σ density·skin over what arrived, so `skin / density` is the
    /// density-weighted fraction of this receptor that is creature.
    pub skin: f32,
}

impl Receptor {
    fn add(&mut self, p: &PipeState) {
        self.density += p.density;
        self.color[0] += p.color[0];
        self.color[1] += p.color[1];
        self.color[2] += p.color[2];
        self.normal += p.normal;
        self.depth += p.depth;
        self.skin += p.skin;
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
        self.skin -= p.skin;
    }
    fn add_receptor(&mut self, o: &Receptor) {
        self.density += o.density;
        self.color[0] += o.color[0];
        self.color[1] += o.color[1];
        self.color[2] += o.color[2];
        self.normal += o.normal;
        self.depth += o.depth;
        self.skin += o.skin;
    }
}

/// What a pipe last delivered — the delta baseline. Same shape as a receptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct PipeState {
    pub density: f32,
    pub color: [f32; 3],
    pub normal: Vec3,
    pub depth: f32,
    pub skin: f32,
}

impl PipeState {
    fn scaled(&self, w: f32) -> PipeState {
        PipeState {
            density: self.density * w,
            color: [self.color[0] * w, self.color[1] * w, self.color[2] * w],
            normal: self.normal * w,
            depth: self.depth * w,
            skin: self.skin * w,
        }
    }
    fn minus(&self, o: &PipeState) -> PipeState {
        PipeState {
            density: self.density - o.density,
            color: [self.color[0] - o.color[0], self.color[1] - o.color[1], self.color[2] - o.color[2]],
            normal: self.normal - o.normal,
            depth: self.depth - o.depth,
            skin: self.skin - o.skin,
        }
    }
    fn max_abs(&self) -> f32 {
        self.density.abs()
            .max(self.color[0].abs()).max(self.color[1].abs()).max(self.color[2].abs())
            .max(self.normal.abs().max_element())
            .max(self.depth.abs())
            .max(self.skin.abs())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RetinaStats {
    pub pipes_total: usize,
    pub pipes_sent: usize,
    pub relinks: u64,
    pub mean_trans: f32,
    pub relink_ms: f32,
    /// Where `relink_ms` went. `hash_build_ms` is `SpatialHash::update`, which
    /// `tick` pays just before the relink, so it is *not* part of `relink_ms`.
    pub relink_footprint_ms: f32,
    pub relink_tau_ms: f32,
    pub relink_pipes_ms: f32,
    pub hash_build_ms: f32,
}

/// Image-space footprint of a projected axis-aligned gaussian: center,
/// eye distance, inverse 2×2 covariance (e = a·du² + 2b·du·dv + c·dv²),
/// and the receptor rectangle the pipe loop walks. The rectangle is the
/// intersection of two boxes — the 2σ box the covariance needs and the
/// kernel's true projected extent (see `footprint`) — so it is neither
/// centred on `(u, v)` nor symmetric.
struct Footprint {
    u: f32,
    v: f32,
    depth: f32,
    a: f32,
    b: f32,
    c: f32,
    u0: f32,
    u1: f32,
    v0: f32,
    v1: f32,
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

/// Where the kernel stops: `kernel()` is exactly zero past `2·radii`, so that
/// surface is the kernel's true extent and the outer edge of its footprint.
const KERNEL_EXTENT: f32 = 2.0;

/// Exact range of the projective coordinate `(num·p̃) / (den·p̃)` over the
/// axis-aligned ellipsoid `centre ± semi` — the ellipsoid's true perspective
/// silhouette along one clip axis. `None` if the ellipsoid is not strictly in
/// front of the plane `den·p̃ = 0`, i.e. it straddles the eye and has no
/// screen-space hull at all.
///
/// Sampling the six axis endpoints instead would *under*estimate this: the
/// silhouette's extreme is a tangency somewhere on the surface, generally not
/// on an axis. Writing `p = c + diag(semi)·z` with `|z| ≤ 1`, the value is
/// `(A + α·z)/(B + β·z)`, and at an extremum `t` the supporting hyperplane
/// gives `|α − tβ| = |tB − A|`. Squaring that is a quadratic in `t` whose two
/// roots are exactly the min and max:
///
/// ```text
/// P t² + 2Q t + R = 0,  P = B² − |β|², Q = α·β − AB, R = A² − |α|²
/// ```
///
/// `P > 0` is precisely "the ellipsoid is in front of the eye", and it is also
/// what makes the discriminant `|Bα − Aβ|² − |α × β|²` non-negative — not
/// Cauchy–Schwarz, which only says the second term is ≥ 0. Split `α` into the
/// parts parallel and perpendicular to `β`: the perpendicular part of
/// `Bα − Aβ` is `Bα_⊥`, so `|Bα − Aβ| ≥ B|α_⊥| ≥ |β||α_⊥| = |α × β|` exactly
/// when `B ≥ |β|`. That form is also what is evaluated here — `Q² − PR` in f32
/// cancels two large products against each other, and `|α|²|β|² − (α·β)²`
/// cancels two more.
fn projective_range(num: Vec4, den: Vec4, centre: Vec3, semi: Vec3) -> Option<(f32, f32)> {
    let c = centre.extend(1.0);
    let (a, b) = (num.dot(c), den.dot(c));
    let (alpha, beta) = (num.truncate() * semi, den.truncate() * semi);
    let p = b * b - beta.length_squared();
    if b <= 0.0 || p <= 1e-6 { return None; }
    let q = alpha.dot(beta) - a * b;
    let disc = ((b * alpha - a * beta).length_squared() - alpha.cross(beta).length_squared()).max(0.0);
    let s = disc.sqrt();
    Some(((-q - s) / p, (-q + s) / p))
}

/// Image-space footprint of `s` under `vp`, or `None` if it has none.
///
/// The gaussian is modelled in image space by a 2×2 covariance, which is a
/// *linearisation* of the projection around the centre. Two things keep that
/// model from running away near the eye, where the perspective divide is at
/// its most nonlinear:
///
/// 1. **Central differences.** Each axis is projected on both sides,
///    `position ± axis·r`, and the image-space half-span `(p_plus - p_minus)/2`
///    is that axis' contribution. A one-sided difference took the near
///    endpoint's offset raw, and as that endpoint approached the near plane
///    its offset grew without bound — the needle that striped the image. The
///    two sides move oppositely under the divide, so most of that cancels.
/// 2. **A bounding box that is the intersection of two boxes**, each
///    correcting the other's failure mode:
///    - the **extent box**: the exact screen-space bounding box of the
///      kernel's extent ellipsoid, `position ± KERNEL_EXTENT·r`, which is
///      precisely the surface where `kernel()` stops being zero. It is the
///      true perspective silhouette — a tangency on the ellipsoid, computed
///      in closed form by `projective_range`, *not* the hull of the six axis
///      endpoints, which is a strict underestimate of it. The gaussian model
///      is only trusted where the kernel actually lands, so a kernel whose
///      extent projects entirely off-screen contributes nothing however wide
///      its covariance came out. That is what removes the rock needles beside
///      the camera, with no depth heuristic: they are simply not on screen.
///    - the **2σ box**, `centre ± 2·sqrt(s_uu)` by `± 2·sqrt(s_vv)`, which
///      circumscribes the `e = 4` ellipse the pipe loop actually evaluates.
///      The ellipse's support in a direction `n` is `2·sqrt(Σ(v·n)²)`, and for
///      an oblique kernel that is strictly *wider* than the hull of the
///      projected axis endpoints, `2·max|v·n|`. Bounding by that hull stopped
///      the pipes while the weights were still ~0.06 — the axis-aligned
///      staircase cut out of the dino's silhouette. Wherever the linearisation
///      holds (everything not right against the eye) the extent box is the
///      wider of the two, the 2σ box binds, and the weights fade to zero
///      before the edge by construction.
///    Both boxes contain the projected centre, so the intersection is never
///    empty and a source too small to span a receptor still keeps the one
///    receptor under it. The screen clip is the pipe loop's.
///
/// Two independent gates give no footprint at all, and both are kept. If the
/// eye is *inside* the kernel (`depth < max radius`) there is no outside view
/// of it to project. And if the extent ellipsoid straddles the eye plane it has
/// no screen-space hull at all: `projective_range` returns `None` and so does
/// the whole footprint, rather than reporting a flat blob from one good axis.
/// Neither subsumes the other. The ellipsoid gate is usually the stricter one
/// (it reaches out to `KERNEL_EXTENT·r`, not `r`), but for anisotropic radii
/// whose long axis runs across the view — say `r = (8, 8, 1)` seen along −Z at
/// depth 4 — the ellipsoid's depth half-extent is only 2, so `|β| < B` passes
/// while `depth < r.max_element()` still fires. That is the case the cheap
/// early-out is there for.
fn footprint(vp: &Mat4, w: u32, h: u32, s: &Source) -> Option<Footprint> {
    let (u, v, depth) = project(vp, w, h, s.position)?;
    let r = s.kernel_radii();
    if depth < r.max_element() { return None; }

    // Extent box: exact silhouette of the ellipsoid position ± 2r. `den` is
    // the clip-w row; `num` is clip-x for u and clip-y for v. v runs down the
    // screen, so its range is the y range mirrored.
    let (den, reach) = (vp.row(3), r * KERNEL_EXTENT);
    let (gx0, gx1) = projective_range(vp.row(0), den, s.position, reach)?;
    let (gy0, gy1) = projective_range(vp.row(1), den, s.position, reach)?;
    let (eu0, eu1) = ((gx0 * 0.5 + 0.5) * w as f32, (gx1 * 0.5 + 0.5) * w as f32);
    let (ev0, ev1) = ((0.5 - gy1 * 0.5) * h as f32, (0.5 - gy0 * 0.5) * h as f32);

    let (mut suu, mut suv, mut svv) = (0.0f32, 0.0f32, 0.0f32);
    for axis in [Vec3::X * r.x, Vec3::Y * r.y, Vec3::Z * r.z] {
        let (pu, pv, _) = project(vp, w, h, s.position + axis)?;
        let (mu, mv, _) = project(vp, w, h, s.position - axis)?;
        let du = (pu - mu) * 0.5;
        let dv = (pv - mv) * 0.5;
        suu += du * du;
        suv += du * dv;
        svv += dv * dv;
    }
    // Floor the variance at half a receptor so distant entities keep a pipe.
    let min_var = 0.25;
    suu = suu.max(min_var);
    svv = svv.max(min_var);
    let det = (suu * svv - suv * suv).max(1e-6);
    // 2σ box ∩ extent box.
    let (hu, hv) = (2.0 * suu.sqrt(), 2.0 * svv.sqrt());
    Some(Footprint {
        u, v, depth,
        a: svv / det,
        b: -suv / det,
        c: suu / det,
        u0: (u - hu).max(eu0),
        u1: (u + hu).min(eu1),
        v0: (v - hv).max(ev0),
        v1: (v + hv).min(ev1),
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
    /// Occluder grid, kept across ticks so its static half is hashed once.
    hash: SpatialHash,
    /// Per-chunk pipe scratch, kept across relinks for its capacity:
    /// (receptors, weights, per-entity counts). One entry per worker chunk, not
    /// per entity — 17k relink-time allocations is a cost of its own.
    pipe_scratch: Vec<(Vec<u32>, Vec<f32>, Vec<u32>)>,
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
            hash: SpatialHash::default(),
            pipe_scratch: Vec::new(),
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
        let t_reset = t0.elapsed();

        // 2–3. Project footprints.
        let fps: Vec<Option<Footprint>> = sources.par_iter()
            .map(|s| if s.drawable { footprint(&view_proj, w, h, s) } else { None })
            .collect();
        let n = sources.len();
        self.entity_depth = fps.iter().map(|f| f.as_ref().map(|f| f.depth).unwrap_or(0.0)).collect();
        self.last_centers = fps.iter()
            .map(|f| f.as_ref().map(|f| (f.u, f.v)).unwrap_or((f32::NAN, f32::NAN)))
            .collect();
        let t_fps = t0.elapsed();

        // 4. Pipes with feathered gaussian weights. Embarrassingly parallel:
        //    each footprint writes only its own receptor/weight run. Chunks of
        //    entities fill scratch buffers in parallel, then the chunks are
        //    concatenated in order — so the SoA arrays come out byte-for-byte
        //    what the serial loop produced (entity order, row-major within a
        //    footprint), which is what the exactness tests pin down.
        let chunks = (rayon::current_num_threads() * 8).max(1);
        let chunk_len = n.div_ceil(chunks).max(1);
        let n_chunks = n.div_ceil(chunk_len);
        // Scratch is kept across relinks for its capacity — per chunk, never
        // per entity.
        if self.pipe_scratch.len() < n_chunks {
            self.pipe_scratch.resize_with(n_chunks, Default::default);
        }
        self.pipe_scratch[..n_chunks].par_iter_mut().zip(fps.par_chunks(chunk_len))
            .for_each(|((recs, weights, counts), fps_chunk)| {
                recs.clear();
                weights.clear();
                counts.clear();
                for fp in fps_chunk {
                    let before = recs.len() as u32;
                    let Some(fp) = fp else { counts.push(0); continue; };
                    let u0 = fp.u0.floor().max(0.0) as i64;
                    let u1 = fp.u1.ceil().min(w as f32 - 1.0) as i64;
                    let v0 = fp.v0.floor().max(0.0) as i64;
                    let v1 = fp.v1.ceil().min(h as f32 - 1.0) as i64;
                    if u0 > u1 || v0 > v1 { counts.push(0); continue; }
                    for rv in v0..=v1 {
                        for ru in u0..=u1 {
                            let du = ru as f32 + 0.5 - fp.u;
                            let dv = rv as f32 + 0.5 - fp.v;
                            let e = fp.a * du * du + 2.0 * fp.b * du * dv + fp.c * dv * dv;
                            if e >= 4.0 { continue; }
                            recs.push(rv as u32 * w + ru as u32);
                            weights.push((-e).exp() * cutoff_window(e));
                        }
                    }
                    counts.push(recs.len() as u32 - before);
                }
            });

        self.pipe_start.clear();
        self.pipe_start.resize(n, 0);
        self.pipe_count.clear();
        self.pipe_count.resize(n, 0);
        let total: usize = self.pipe_scratch[..n_chunks].iter().map(|(r, _, _)| r.len()).sum();
        self.pipe_receptor.clear();
        self.pipe_receptor.reserve(total);
        self.pipe_weight.clear();
        self.pipe_weight.reserve(total);
        let mut i = 0usize;
        let mut base = 0u32;
        for (recs, weights, counts) in &self.pipe_scratch[..n_chunks] {
            for &c in counts.iter() {
                self.pipe_start[i] = base;
                self.pipe_count[i] = c;
                base += c;
                i += 1;
            }
            self.pipe_receptor.extend_from_slice(recs);
            self.pipe_weight.extend_from_slice(weights);
        }
        debug_assert_eq!(i, n, "chunked pipe counts did not cover every entity");
        debug_assert_eq!(base as usize, total);
        // Reuse the allocation: at ~577k pipes this buffer is ~18 MB, and
        // handing it back to the allocator every relink is not free.
        self.pipe_last.clear();
        self.pipe_last.resize(total, PipeState::default());
        let t_pipes = t0.elapsed();

        // 5. τ toward the eye — only for entities that actually got pipes
        // (parallel). Measured at ~8.5 ms of a ~36 ms relink on the demo
        // scene: a quarter of it. The serial pipe loop above is the cost.
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
        let t_tau = t0.elapsed();

        // 6. Bookkeeping.
        self.last_view_proj = Some(view_proj);
        let linked: Vec<f32> = (0..n).filter(|&i| self.pipe_count[i] > 0).map(|i| self.entity_trans[i]).collect();
        self.stats.relinks += 1;
        self.stats.pipes_total = self.pipe_receptor.len();
        self.stats.mean_trans = if linked.is_empty() { 1.0 } else { linked.iter().sum::<f32>() / linked.len() as f32 };
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        self.stats.relink_footprint_ms = ms(t_fps - t_reset);
        self.stats.relink_pipes_ms = ms(t_pipes - t_fps);
        self.stats.relink_tau_ms = ms(t_tau - t_pipes);
        // `tick` builds the hash; a direct `relink` was handed one, so the
        // number it last recorded says nothing about this call.
        self.stats.hash_build_ms = 0.0;
        self.stats.relink_ms = ms(t0.elapsed());
        log::debug!(
            "relink {:.2} ms = reset {:.2} + footprints {:.2} + pipes {:.2} + τ {:.2} + bookkeeping {:.2} ({} entities, {} pipes; hash build is `tick`'s, not counted here)",
            self.stats.relink_ms, ms(t_reset), self.stats.relink_footprint_ms,
            self.stats.relink_pipes_ms, self.stats.relink_tau_ms,
            self.stats.relink_ms - ms(t_tau), n, self.stats.pipes_total,
        );
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
            // Density-weighted, exactly like `depth`: the receptor sums it and
            // the renderer divides by density to get a fraction back.
            skin: if s.skin { d } else { 0.0 },
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
            // The hash lives in the retina so its static half survives the
            // tick; take it out for the duration so `relink` can borrow `self`
            // mutably, and hand it straight back.
            let mut hash = std::mem::take(&mut self.hash);
            let t0 = std::time::Instant::now();
            hash.update(sources);
            let build_ms = t0.elapsed().as_secs_f32() * 1000.0;
            let eye = eye_from_view_proj(view_proj);
            self.relink(sources, &hash, view_proj, eye, aabb_min, aabb_max, atten_k);
            self.stats.hash_build_ms = build_ms;
            self.hash = hash;
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
        log::info!(
            "Retina relink breakdown: hash build {:.2} ms + relink {:.2} ms (footprints {:.2}, pipes {:.2}, τ {:.2})",
            self.stats.hash_build_ms, self.stats.relink_ms,
            self.stats.relink_footprint_ms, self.stats.relink_pipes_ms, self.stats.relink_tau_ms,
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
                        .max((got.skin - want.skin).abs())
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
            is_static: false,
            skin: false,
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

    /// A few points that straddle both occluders, their overlap, and empty air.
    fn probe_points() -> Vec<Vec3> {
        vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.5, -0.5),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(-1.5, 1.0, 1.0),
            Vec3::new(40.0, 40.0, 40.0), // nothing here
        ]
    }

    /// Splitting the occluders across a static and a dynamic map must be
    /// invisible: `density_at` sums both cells, so it reads exactly what one
    /// combined build reads.
    #[test]
    fn split_hash_reads_the_same_density_as_a_combined_build() {
        let mut stat = src(Vec3::ZERO, Vec3::splat(1.5), 4.0);
        stat.is_static = true;
        let mut dynamic = src(Vec3::new(3.0, 0.0, 0.0), Vec3::ONE, 7.0);
        dynamic.is_static = false;
        let sources = vec![stat, dynamic];

        let split = SpatialHash::build(&sources);
        // Reference: the same occluders, every one of them declared static, so
        // they all land in one map.
        let all_static: Vec<Source> = sources.iter().map(|s| Source { is_static: true, ..*s }).collect();
        let combined = SpatialHash::build(&all_static);

        let mut nonzero = 0;
        for p in probe_points() {
            let a = split.density_at(&sources, p, &[]);
            let b = combined.density_at(&all_static, p, &[]);
            assert_eq!(a, b, "split vs combined at {:?}: {} vs {}", p, a, b);
            if a > 0.0 { nonzero += 1; }
        }
        assert!(nonzero >= 4, "probe points barely touched the occluders ({} hits)", nonzero);
        assert_eq!(split.max_extent(), combined.max_extent(), "max_extent must cover both maps");
        // `skip` still reaches into both maps.
        let p = Vec3::new(2.0, 0.0, 0.0);
        assert_eq!(split.density_at(&sources, p, &[0]), 7.0 * sources[1].kernel(p));
        assert_eq!(split.density_at(&sources, p, &[1]), 4.0 * sources[0].kernel(p));
    }

    /// Moving the dynamic occluder and re-running `update` must rebuild only
    /// the dynamic map yet read exactly like a hash built from scratch.
    #[test]
    fn updating_only_the_dynamic_map_matches_a_fresh_build() {
        let mut stat = src(Vec3::ZERO, Vec3::splat(1.5), 4.0);
        stat.is_static = true;
        let mut dynamic = src(Vec3::new(3.0, 0.0, 0.0), Vec3::ONE, 7.0);
        dynamic.is_static = false;
        let mut sources = vec![stat, dynamic];

        let mut hash = SpatialHash::build(&sources);
        let static_builds = hash.static_builds();

        for step in 0..4 {
            sources[1].position += Vec3::new(-1.25, 0.75, 0.5);
            hash.update(&sources);
            assert_eq!(hash.static_builds(), static_builds,
                "step {}: the static map was rebuilt though the static set never changed", step);
            let fresh = SpatialHash::build(&sources);
            for p in probe_points() {
                let a = hash.density_at(&sources, p, &[]);
                let b = fresh.density_at(&sources, p, &[]);
                assert_eq!(a, b, "step {} at {:?}: incremental {} vs fresh {}", step, p, a, b);
            }
            assert_eq!(hash.max_extent(), fresh.max_extent());
        }

        // A changed entity set is the one thing `update` cannot carry over, so
        // it must notice and rebuild the static map.
        let mut extra = src(Vec3::new(-6.0, 0.0, 0.0), Vec3::ONE, 3.0);
        extra.is_static = true;
        sources.push(extra);
        hash.update(&sources);
        assert_eq!(hash.static_builds(), static_builds + 1, "static set grew but the map did not");
        let fresh = SpatialHash::build(&sources);
        for p in probe_points().into_iter().chain([Vec3::new(-6.0, 0.0, 0.0)]) {
            assert_eq!(hash.density_at(&sources, p, &[]), fresh.density_at(&sources, p, &[]), "at {:?}", p);
        }
    }

    /// `segment_transmittance` walks the split hash and must not care.
    #[test]
    fn transmittance_is_unchanged_by_the_static_dynamic_split() {
        let mut floor = src(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE, 10.0);
        floor.is_static = true;
        let mut walker = src(Vec3::new(0.0, 0.0, -7.0), Vec3::ONE, 10.0);
        walker.is_static = false;
        let sources = vec![floor, walker];
        let split = SpatialHash::build(&sources);
        let all_static: Vec<Source> = sources.iter().map(|s| Source { is_static: true, ..*s }).collect();
        let combined = SpatialHash::build(&all_static);
        let (from, to) = (Vec3::new(0.0, 0.0, -12.0), Vec3::ZERO);
        let a = segment_transmittance(&sources, &split, from, to, &[], 0.5, ATTEN_THRESHOLD, None);
        let b = segment_transmittance(&all_static, &combined, from, to, &[], 0.5, ATTEN_THRESHOLD, None);
        assert!(a < 0.5, "test occluders do not attenuate: {}", a);
        assert_eq!(a, b, "split τ={} combined τ={}", a, b);
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

    /// A kernel whose extent reaches the near plane has no image-plane
    /// footprint. `footprint()` linearises the projection by projecting the
    /// three axis endpoints; once `w < 2·r` the depth endpoint lands at
    /// `w' ≈ 0⁺` and its projected offset is unbounded, so the covariance
    /// degenerates into a needle painting a band across the whole image.
    /// Both entities in the user's screenshot were exactly this: unit kernels
    /// at view depth 0.30 and 0.45, spanning 299×18 and 181×26 receptors.
    #[test]
    fn kernels_that_reach_the_near_plane_get_no_pipes() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        let sources = vec![
            // Beside the camera, well outside the frustum, but at view depth
            // 0.304 — the 299-column band from the screenshot.
            src(Vec3::new(15.7, 0.5, -0.304), Vec3::ONE, 1.0),
            // Straight ahead but inside the kernel extent: the eye is engulfed.
            src(Vec3::new(0.0, 0.5, -0.304), Vec3::ONE, 1.0),
        ];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        for i in 0..sources.len() {
            assert_eq!(r.pipes_of(i).count(), 0,
                "near-plane source {} got {} pipes", i, r.pipes_of(i).count());
        }
        assert_eq!(r.stats.pipes_total, 0);
    }

    /// Entity 7540 in the demo scene — a unit kernel (GROUP_ROCK,
    /// `deposit_radii == ZERO`) 23.5 cells off the view axis at centre depth
    /// 2.43 — striped all 320 columns of the retina. It is nowhere near the
    /// screen: its projected centre sits hundreds of receptors off the right
    /// edge, and only the runaway covariance of a one-sided difference dragged
    /// a bounding box back across the image. Bounded by its own projected
    /// extent it contributes nothing, with no appeal to its depth.
    #[test]
    fn an_off_axis_kernel_beside_the_camera_gets_no_pipes() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        // Eye at the origin looking −Z, so view depth is just −z.
        let s = src(Vec3::new(23.5, 0.0, -2.43), Vec3::ONE, 1.0);
        // Not cut for being close: the eye is well outside this kernel.
        let (cu, _, depth) = project(&vp, w, h, s.position).expect("centre is in front of the eye");
        assert!(depth > s.kernel_radii().max_element(),
            "the eye must be outside the kernel for this to test the extent bound, depth {}", depth);
        assert!(cu > w as f32, "centre must project off the right edge, u = {}", cu);

        let sources = vec![s];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        assert_eq!(r.pipes_of(0).count(), 0,
            "off-axis source beside the camera got {} pipes", r.pipes_of(0).count());
        assert_eq!(r.stats.pipes_total, 0);
    }

    /// The counterweight to the needle cuts: a floor tile just ahead and just
    /// below the eye, the geometry you fly over low. Its centre is 2.5 cells
    /// out and its near endpoint 1.5, so any gate that judged an endpoint's
    /// depth against the kernel extent would swallow the floor in front of the
    /// camera. It has to be drawn, and drawn as a tile rather than a band.
    #[test]
    fn a_floor_tile_just_ahead_and_below_the_eye_is_still_drawn() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        let sources = vec![src(Vec3::new(0.0, -0.5, -2.5), Vec3::ONE, 1.0)];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        let pipes: Vec<u32> = r.pipes_of(0).map(|(rc, _)| rc).collect();
        assert!(!pipes.is_empty(), "the floor ahead of the eye was cut");
        let u0 = pipes.iter().map(|rc| rc % w).min().unwrap();
        let u1 = pipes.iter().map(|rc| rc % w).max().unwrap();
        assert!(u1 - u0 + 1 < 100,
            "floor tile spans {} of {} columns — a band, not a tile", u1 - u0 + 1, w);
    }

    /// The cut is exactly at the kernel extent: one radius further out and the
    /// source is drawn normally, as a compact blob rather than a band.
    #[test]
    fn a_source_just_past_the_kernel_extent_still_gets_a_compact_footprint() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        let sources = vec![src(Vec3::new(0.0, 0.5, -3.0), Vec3::ONE, 1.0)];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        let pipes: Vec<u32> = r.pipes_of(0).map(|(rc, _)| rc).collect();
        assert!(!pipes.is_empty(), "w = 3 > 2·r, the source must still be drawn");
        let u0 = pipes.iter().map(|rc| rc % w).min().unwrap();
        let u1 = pipes.iter().map(|rc| rc % w).max().unwrap();
        assert!(u1 - u0 + 1 < 100,
            "footprint spans {} of {} columns — still a band", u1 - u0 + 1, w);
    }

    /// The closed-form silhouette must be exactly the min/max of the projected
    /// ellipsoid — tight enough that nothing on the surface escapes it, and no
    /// looser than that. Brute-forced against a dense sampling of the surface,
    /// for an off-axis ellipsoid whose extremes are nowhere near an axis
    /// endpoint (which is the whole reason the six-endpoint hull was wrong).
    #[test]
    fn projective_range_is_the_exact_silhouette_of_the_ellipsoid() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        let (c, semi) = (Vec3::new(6.0, 2.0, -18.0), Vec3::new(10.0, 12.0, 16.0));
        let (g0, g1) = projective_range(vp.row(0), vp.row(3), c, semi)
            .expect("ellipsoid is in front of the eye");

        let g_of = |z: Vec3| { let clip = vp * (c + z * semi).extend(1.0); clip.x / clip.w };
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in 0..=600 {
            for j in 0..=600 {
                let theta = i as f32 / 600.0 * std::f32::consts::PI;
                let phi = j as f32 / 600.0 * std::f32::consts::TAU;
                let z = Vec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
                let g = g_of(z);
                lo = lo.min(g);
                hi = hi.max(g);
            }
        }
        assert!(g0 <= lo + 1e-3 && g1 >= hi - 1e-3, "silhouette misses the surface: [{}, {}] vs [{}, {}]", g0, g1, lo, hi);
        assert!((g0 - lo).abs() < 2e-3 && (g1 - hi).abs() < 2e-3, "silhouette is loose: [{}, {}] vs [{}, {}]", g0, g1, lo, hi);

        // The extremes are tangencies, not axis endpoints — which is exactly
        // why the hull of the six endpoints is a strict underestimate.
        let (mut hull_lo, mut hull_hi) = (f32::MAX, f32::MIN);
        for z in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            hull_lo = hull_lo.min(g_of(z));
            hull_hi = hull_hi.max(g_of(z));
        }
        assert!(hull_lo > g0 + 0.01 && hull_hi < g1 - 0.01,
            "six-endpoint hull [{}, {}] is not strictly inside the silhouette [{}, {}]",
            hull_lo, hull_hi, g0, g1);

        // Straddling the eye plane: |β| = 2 exceeds B = 1.5, no hull at all.
        assert!(projective_range(vp.row(0), vp.row(3), Vec3::new(0.0, 0.0, -1.5), Vec3::ONE * 2.0).is_none());
    }

    /// A big oblique metaball must fade out inside its own bounding box, not
    /// stop at the box edge with the weights still large — that hard stop is
    /// the axis-aligned staircase cut through the dino's silhouette.
    ///
    /// The bound has to be the *ellipse's* support, `2·sqrt(Σ(v·n)²)`, which
    /// for an oblique kernel is strictly wider than the hull of the projected
    /// axis endpoints, `2·max|v·n|`. The three axes here project to spans of
    /// ~11, ~13 and ~(7, −2) receptors, so the u-hull stops ~4.5 receptors
    /// short of the ellipse and cuts the blob where its weight is still ~0.06.
    #[test]
    fn a_big_oblique_metaball_fades_out_before_its_footprint_edge() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        // Off-axis in x and y, so no projected axis lines up with a screen axis.
        let sources = vec![src(Vec3::new(6.0, 2.0, -18.0), Vec3::new(5.0, 6.0, 8.0), 1.0)];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        let pipes: Vec<(u32, u32, f32)> = r.pipes_of(0)
            .map(|(rc, weight)| (rc % w, rc / w, weight)).collect();
        assert!(!pipes.is_empty(), "the metaball got no pipes at all");

        // The centre is on screen, so somewhere in there a pipe carries ~all
        // of the source. Without this the "no hard cut" check is vacuous.
        let max_w = pipes.iter().map(|&(_, _, x)| x).fold(0.0f32, f32::max);
        assert!(max_w >= 0.9, "centre is not on screen: max weight {}", max_w);

        let (cu0, cu1) = (pipes.iter().map(|p| p.0).min().unwrap(), pipes.iter().map(|p| p.0).max().unwrap());
        let (rv0, rv1) = (pipes.iter().map(|p| p.1).min().unwrap(), pipes.iter().map(|p| p.1).max().unwrap());
        // The footprint must be well inside the screen, or "the border of the
        // pipe set" would be the screen edge and prove nothing about the box.
        assert!(cu0 > 0 && cu1 < w - 1 && rv0 > 0 && rv1 < h - 1,
            "footprint touches the screen edge: u {}..{}, v {}..{}", cu0, cu1, rv0, rv1);
        for &(cu, rv, weight) in &pipes {
            if cu == cu0 || cu == cu1 || rv == rv0 || rv == rv1 {
                assert!(weight < 0.05,
                    "hard cut at the footprint border: weight {} at (u {}, v {}) of u {}..{}, v {}..{}",
                    weight, cu, rv, cu0, cu1, rv0, rv1);
            }
        }
    }

    /// The eye inside the kernel's *extent* — not just its core — has no
    /// outside view of it either: the −Z endpoint at 2r sits behind the eye,
    /// so the projected hull is not a screen-space box at all.
    #[test]
    fn a_kernel_whose_extent_straddles_the_eye_gets_no_pipes() {
        let (w, h) = (141u32, 79u32);
        let vp = test_view_proj(w, h);
        // depth 1.5 > r = 1, so the cheap eye-inside-core early-out misses it;
        // the +Z extent endpoint is at z = +0.5, behind the eye.
        let sources = vec![src(Vec3::new(0.0, 0.0, -1.5), Vec3::ONE, 1.0)];
        let hash = SpatialHash::build(&sources);
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &hash, vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);
        assert_eq!(r.pipes_of(0).count(), 0,
            "kernel straddling the eye got {} pipes", r.pipes_of(0).count());
        assert_eq!(r.stats.pipes_total, 0);
    }

    /// The pipe arrays are built in parallel over chunks of entities and then
    /// concatenated. The layout that comes out has to be the one the serial
    /// loop produced: entity order, row-major within a footprint, `pipe_start`
    /// the prefix sum of `pipe_count`. Enough sources here to span many chunks.
    #[test]
    fn chunked_pipe_build_matches_the_serial_layout() {
        let (w, h) = (63u32, 35u32);
        let vp = test_view_proj(w, h);
        let mut seed = 11u64;
        let n = 4 * rayon::current_num_threads() * 8 + 37; // several entities per chunk
        let sources: Vec<Source> = (0..n).map(|_| {
            src(
                Vec3::new(lcg(&mut seed) * 16.0 - 8.0, lcg(&mut seed) * 12.0 - 6.0, -6.0 - lcg(&mut seed) * 20.0),
                Vec3::splat(0.4 + lcg(&mut seed) * 2.5),
                1.0,
            )
        }).collect();
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.relink(&sources, &SpatialHash::build(&sources), vp, Vec3::ZERO, lo, hi, ATTEN_K_DEFAULT);

        // Reference: the same footprints, walked serially exactly as `relink`
        // walked them before it was parallelised.
        let mut want_rec: Vec<u32> = Vec::new();
        let mut want_w: Vec<f32> = Vec::new();
        let mut want_start: Vec<u32> = Vec::new();
        for s in &sources {
            want_start.push(want_rec.len() as u32);
            let Some(fp) = footprint(&vp, w, h, s) else { continue; };
            let u0 = fp.u0.floor().max(0.0) as i64;
            let u1 = fp.u1.ceil().min(w as f32 - 1.0) as i64;
            let v0 = fp.v0.floor().max(0.0) as i64;
            let v1 = fp.v1.ceil().min(h as f32 - 1.0) as i64;
            if u0 > u1 || v0 > v1 { continue; }
            for rv in v0..=v1 {
                for ru in u0..=u1 {
                    let du = ru as f32 + 0.5 - fp.u;
                    let dv = rv as f32 + 0.5 - fp.v;
                    let e = fp.a * du * du + 2.0 * fp.b * du * dv + fp.c * dv * dv;
                    if e >= 4.0 { continue; }
                    want_rec.push(rv as u32 * w + ru as u32);
                    want_w.push((-e).exp() * cutoff_window(e));
                }
            }
        }
        assert!(want_rec.len() > 10_000, "test scene is too thin to prove anything: {} pipes", want_rec.len());
        assert_eq!(r.stats.pipes_total, want_rec.len(), "pipe count differs from the serial build");

        let mut k = 0usize;
        for i in 0..n {
            let got: Vec<(u32, f32)> = r.pipes_of(i).collect();
            assert_eq!(r.pipe_start[i], want_start[i], "entity {} start", i);
            assert_eq!(r.pipe_start[i] as usize, k, "entity {} start is not the running prefix sum", i);
            for (off, &(rc, weight)) in got.iter().enumerate() {
                assert_eq!(rc, want_rec[k + off], "entity {} pipe {} receptor", i, off);
                assert_eq!(weight, want_w[k + off], "entity {} pipe {} weight", i, off);
            }
            // Row-major within the footprint: receptor indices strictly ascend.
            assert!(got.windows(2).all(|p| p[0].0 < p[1].0), "entity {} pipes are out of order", i);
            k += got.len();
        }
        assert_eq!(k, want_rec.len(), "pipes_of did not cover the whole array");
    }

    /// `skin` is a density-weighted sum like every other receptor channel, so
    /// `skin / density` is the fraction of what arrived at this receptor that
    /// came from a creature. Where a dino overlaps the floor the renderer needs
    /// that fraction — not a colour guess — to decide who gets scales.
    #[test]
    fn skin_fraction_is_the_density_weighted_share_of_creature_sources() {
        let (w, h) = (63u32, 35u32);
        let vp = test_view_proj(w, h);
        // Opacity 0 so nothing attenuates anything: τ = 1 for both, and the
        // fraction is decided purely by density × footprint weight.
        let mut creature = src(Vec3::new(-1.0, 0.0, -10.0), Vec3::splat(2.0), 0.0);
        creature.density = 3.0;
        creature.skin = true;
        let mut scenery = src(Vec3::new(1.5, 0.8, -10.0), Vec3::splat(2.0), 0.0);
        scenery.density = 1.0;
        scenery.skin = false;
        let sources = vec![creature, scenery];
        let (lo, hi) = box_of(&sources);
        let mut r = Retina::new(w, h);
        r.tick(&sources, vp, lo, hi, false, ATTEN_K_DEFAULT);
        assert!((r.transmittance(0) - 1.0).abs() < 1e-6 && (r.transmittance(1) - 1.0).abs() < 1e-6);

        // Centre receptor: both reach it, with different weights.
        let centre = 17 * w + 31;
        let weight = |i: usize| r.pipes_of(i).find(|&(rc, _)| rc == centre).map(|(_, x)| x);
        let (wc, ws) = (weight(0).expect("creature misses the centre"), weight(1).expect("scenery misses the centre"));
        assert!(wc > 0.1 && ws > 0.1 && (wc - ws).abs() > 1e-3,
            "the two sources must overlap with unequal weight: {} {}", wc, ws);
        let rec = r.receptors[centre as usize];
        let want = 3.0 * wc / (3.0 * wc + ws);
        assert!((rec.skin / rec.density - want).abs() < 1e-5,
            "skin fraction {} want {}", rec.skin / rec.density, want);

        // A receptor only the creature reaches is all skin, and one only the
        // scenery reaches is none of it.
        let only = |i: usize, j: usize| {
            let other: Vec<u32> = r.pipes_of(j).map(|(rc, _)| rc).collect();
            r.pipes_of(i).map(|(rc, _)| rc).find(|rc| !other.contains(rc))
                .expect("no receptor exclusive to this source")
        };
        let pure = r.receptors[only(0, 1) as usize];
        assert!(pure.density > 1e-3 && (pure.skin / pure.density - 1.0).abs() < 1e-5,
            "creature-only receptor is not all skin: {}/{}", pure.skin, pure.density);
        let bare = r.receptors[only(1, 0) as usize];
        assert!(bare.density > 1e-3 && bare.skin.abs() < 1e-6,
            "scenery-only receptor picked up skin: {}", bare.skin);
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
            assert!((got.skin - want.skin).abs() < tol, "receptor {} skin {} vs {}", i, got.skin, want.skin);
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
        assert_eq!(bytes, 36, "Receptor changed size — the scratch budget math below assumes it");
        // A 1280×720 retina (the new MAX_RETINA_DIM aspect) is 921_600
        // receptors ≈ 33 MB of scratch each; 64 MB buys two.
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
