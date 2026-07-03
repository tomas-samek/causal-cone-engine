// Walk controller — moves the walker group (the dino) as one rigid body.
//
// Speeds are stored honestly in cells per world-tick (fractions of c).
// At strongly subluminal speeds (1e-6 c) motion is imperceptible in real
// time — ~9 hours per cell — so each sim step advances `time_lapse`
// world-ticks of world time: displacement = speed × time_lapse.
// The light field is quasi-static at this ratio (adiabatic regime), so
// light transport keeps running once per sim step unchanged.

use glam::Vec3;

/// Dino speed in cells per world-tick (fraction of c). Strongly subluminal.
pub const DINO_SPEED_C: f32 = 1e-6;
/// World-ticks that elapse per sim step (default ×100,000 time-lapse).
pub const TIME_LAPSE_DEFAULT: u64 = 100_000;
/// Upper clamp for time-lapse adjustment (2^20).
pub const TIME_LAPSE_MAX: u64 = 1_048_576;
/// Max cells of drift from spawn along the walk axis before turning around.
/// Keeps the feet clear of the rock (needs Z-offset > +8) and the tail
/// roughly over the floor.
pub const WALK_SPAN: f32 = 6.0;

pub struct WalkController {
    /// Speed in cells per world-tick (≤ 1e-6 c).
    pub speed_c: f32,
    /// Current walk direction (unit vector, ±Z in v1).
    pub direction: Vec3,
    /// World-ticks per sim step.
    pub time_lapse: u64,
    /// Cumulative displacement from spawn. Per-step f32 rounding drift is
    /// ~1e-5 cells and bounded (offset stays within ±span; internal edge
    /// geometry is re-derived from actual positions at each refresh).
    pub offset: Vec3,
    /// Max |offset.z| before the direction flips.
    pub span: f32,
}

impl WalkController {
    pub fn new() -> Self {
        Self {
            speed_c: DINO_SPEED_C,
            direction: Vec3::Z,
            time_lapse: TIME_LAPSE_DEFAULT,
            offset: Vec3::ZERO,
            span: WALK_SPAN,
        }
    }

    /// Advance one sim step. Returns the displacement to apply to every
    /// walker entity this step; flips direction once the span is reached.
    pub fn step(&mut self) -> Vec3 {
        let delta = self.direction * (self.speed_c * self.time_lapse as f32);
        self.offset += delta;
        // Assign (don't toggle) so a runtime lapse change at the boundary
        // can't flip direction every step and trap the walker outside the span.
        if self.offset.z >= self.span {
            self.direction = -Vec3::Z;
        } else if self.offset.z <= -self.span {
            self.direction = Vec3::Z;
        }
        delta
    }

    pub fn halve_time_lapse(&mut self) {
        self.time_lapse = (self.time_lapse / 2).max(1);
    }

    pub fn double_time_lapse(&mut self) {
        self.time_lapse = (self.time_lapse * 2).min(TIME_LAPSE_MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn step_displacement_is_speed_times_lapse() {
        let mut w = WalkController::new();
        let d = w.step();
        // 1e-6 cells/world-tick × 100_000 world-ticks/step = 0.1 cells/step, along +Z
        assert!((d.z - 0.1).abs() < 1e-5, "dz = {}", d.z);
        assert_eq!(d.x, 0.0);
        assert_eq!(d.y, 0.0);
        assert!((w.offset.z - 0.1).abs() < 1e-5);
    }

    #[test]
    fn lapse_of_one_gives_literal_subluminal_step() {
        let mut w = WalkController::new();
        w.time_lapse = 1;
        let d = w.step();
        assert!((d.z - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn offset_is_exact_sum_of_returned_deltas() {
        let mut w = WalkController::new();
        let mut sum = Vec3::ZERO;
        for _ in 0..500 {
            sum += w.step();
        }
        assert_eq!(sum, w.offset);
    }

    #[test]
    fn turns_around_at_span_and_stays_bounded() {
        let mut w = WalkController::new();
        // 0.1 cells/step, span 6.0 → direction must flip around step 60
        let mut flipped = false;
        for _ in 0..500 {
            w.step();
            if w.direction.z < 0.0 {
                flipped = true;
            }
            // never drifts beyond span + one step of slack
            assert!(w.offset.z.abs() <= WALK_SPAN + 0.11, "offset = {}", w.offset.z);
        }
        assert!(flipped, "walker never turned around");
    }

    #[test]
    fn lapse_change_at_span_does_not_trap_walker() {
        // Reproduces the toggle-trap: with the old `direction = -direction`
        // toggle, shrinking the step while the offset sits past the span made
        // the direction flip every tick, pinning the walker at the boundary.
        let mut w = WalkController::new();

        // Walk out to the +Z boundary at the default lapse (0.1 cells/step).
        let mut guard = 0;
        while w.offset.z < WALK_SPAN {
            w.step();
            guard += 1;
            assert!(guard < 1000, "never reached the span");
        }
        // The crossing step assigned the inward direction.
        assert_eq!(w.direction, -Vec3::Z, "should head back inward after crossing");

        // Shrink the step 16× (delta ~0.00625 cells) — the danger zone where a
        // per-step toggle would vibrate at the boundary forever.
        for _ in 0..4 {
            w.halve_time_lapse();
        }

        let mut escaped = false;
        for _ in 0..2000 {
            w.step();
            // (i) While still at/beyond the span, direction never re-flips to +Z.
            if w.offset.z >= WALK_SPAN {
                assert_eq!(w.direction, -Vec3::Z,
                    "re-flipped at the boundary (trap): offset.z = {}", w.offset.z);
            }
            // (ii) The walker escapes inward.
            if w.offset.z < WALK_SPAN - 1.0 {
                escaped = true;
                break;
            }
        }
        assert!(escaped, "walker never escaped the boundary — trapped");

        // Restore the large step and confirm the walk stays bounded with it too.
        for _ in 0..4 {
            w.double_time_lapse();
        }
        for _ in 0..500 {
            w.step();
            assert!(w.offset.z.abs() <= WALK_SPAN + 1.2,
                "unbounded with large step: offset.z = {}", w.offset.z);
        }
    }

    #[test]
    fn time_lapse_clamps() {
        let mut w = WalkController::new();
        w.time_lapse = 1;
        w.halve_time_lapse();
        assert_eq!(w.time_lapse, 1);
        w.time_lapse = TIME_LAPSE_MAX;
        w.double_time_lapse();
        assert_eq!(w.time_lapse, TIME_LAPSE_MAX);
        w.time_lapse = TIME_LAPSE_DEFAULT;
        w.double_time_lapse();
        assert_eq!(w.time_lapse, 200_000);
        w.halve_time_lapse();
        assert_eq!(w.time_lapse, 100_000);
    }
}
