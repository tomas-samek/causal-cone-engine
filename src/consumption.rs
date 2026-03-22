use std::collections::HashMap;

// ── Constants ────────────────────────────────────────────────────────────────

pub const TARGET_COVERAGE: f32 = 0.50;
pub const SEED_THRESHOLD: usize = 60;
pub const MAX_TRIE_DEPTH: u16 = 20;

// ── DepositToken ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepositToken {
    pub density: u8,     // 0..=15
    pub rg: u8,          // RRRR_GGGG (4 bits each)
    pub b_reserved: u8,  // BBBB_0000
}

impl DepositToken {
    /// Quantize continuous deposit values into 4-bit levels.
    /// Each input channel is in [0, 1]; density is scaled by `density_scale` first.
    pub fn from_deposit(density: f32, r: f32, g: f32, b: f32, density_scale: f32) -> Self {
        let quantize = |v: f32| -> u8 { (v * 15.0).round().clamp(0.0, 15.0) as u8 };
        let d = quantize((density * density_scale).clamp(0.0, 1.0));
        let rq = quantize(r.clamp(0.0, 1.0));
        let gq = quantize(g.clamp(0.0, 1.0));
        let bq = quantize(b.clamp(0.0, 1.0));
        Self {
            density: d,
            rg: (rq << 4) | gq,
            b_reserved: bq << 4,
        }
    }
}

// ── Spectrum ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Spectrum {
    tokens: HashMap<DepositToken, ()>,
}

impl Spectrum {
    /// Build a spectrum from observation counts, keeping the most-frequent
    /// tokens that together cover at least `TARGET_COVERAGE` of the total.
    pub fn crystallize_from(obs_counts: &HashMap<DepositToken, u64>) -> Self {
        let total: u64 = obs_counts.values().sum();
        if total == 0 {
            return Self {
                tokens: HashMap::new(),
            };
        }

        // Sort tokens by count descending.
        let mut sorted: Vec<(DepositToken, u64)> =
            obs_counts.iter().map(|(&t, &c)| (t, c)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let mut kept = HashMap::new();
        let mut accumulated: u64 = 0;
        let threshold = (total as f32 * TARGET_COVERAGE) as u64;
        for (token, count) in sorted {
            kept.insert(token, ());
            accumulated += count;
            if accumulated >= threshold {
                break;
            }
        }

        Self { tokens: kept }
    }

    pub fn contains(&self, token: &DepositToken) -> bool {
        self.tokens.contains_key(token)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ── Seed ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Seed {
    pub birth_tick: u64,
    pub entries: Vec<(u64, DepositToken)>,
}

impl Seed {
    pub fn new(tick: u64) -> Self {
        Self {
            birth_tick: tick,
            entries: Vec::new(),
        }
    }

    pub fn feed(&mut self, tick: u64, token: DepositToken) {
        self.entries.push((tick, token));
    }

    pub fn should_promote(&self) -> bool {
        self.entries.len() >= SEED_THRESHOLD
    }
}

// ── ConsumptionState ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ConsumptionState {
    pub depth: u16,
    pub birth_tick: u64,
    pub learning_window: u64,
    pub learning: bool,
    pub obs_counts: HashMap<DepositToken, u64>,
    pub learning_ticks: u64,
    pub spectrum: Spectrum,
    pub consumed: u64,
    pub rejected: u64,
    pub trie_child: Option<usize>,
    pub trie_parent: Option<usize>,
    pub seed: Option<Seed>,
    pub consumption_log: Option<Vec<(u64, DepositToken)>>,
    pub learning_log: Option<Vec<(u64, DepositToken)>>,
}

impl ConsumptionState {
    pub fn new(depth: u16, birth_tick: u64, enable_logs: bool) -> Self {
        let learning_window = birth_tick.max(1);
        Self {
            depth,
            birth_tick,
            learning_window,
            learning: true,
            obs_counts: HashMap::new(),
            learning_ticks: 0,
            spectrum: Spectrum {
                tokens: HashMap::new(),
            },
            consumed: 0,
            rejected: 0,
            trie_child: None,
            trie_parent: None,
            seed: None,
            consumption_log: if enable_logs { Some(Vec::new()) } else { None },
            learning_log: if enable_logs { Some(Vec::new()) } else { None },
        }
    }

    /// Record an observation during the learning phase.
    pub fn observe(&mut self, tick: u64, token: DepositToken) {
        *self.obs_counts.entry(token).or_insert(0) += 1;
        self.learning_ticks += 1;
        if let Some(ref mut log) = self.learning_log {
            log.push((tick, token));
        }
        // Auto-crystallize when we have observed for the full learning window.
        if self.learning_ticks >= self.learning_window {
            self.crystallize();
        }
    }

    /// Crystallize the spectrum from accumulated observations and leave learning mode.
    pub(crate) fn crystallize(&mut self) {
        self.spectrum = Spectrum::crystallize_from(&self.obs_counts);
        self.learning = false;
    }

    /// Record a consumption event.
    pub fn consume(&mut self, tick: u64, token: DepositToken) {
        self.consumed += 1;
        if let Some(ref mut log) = self.consumption_log {
            log.push((tick, token));
        }
    }
}

// ── cascade_process ──────────────────────────────────────────────────────────

/// Iterative trie-routing for a deposit token.
/// Returns `Some(new_idx)` if a new child state was spawned.
pub fn cascade_process(
    states: &mut Vec<ConsumptionState>,
    start_idx: usize,
    tick: u64,
    token: DepositToken,
) -> Option<usize> {
    let mut current_idx = start_idx;

    loop {
        // Extract fields we need before mutating.
        let learning = states[current_idx].learning;
        let contains = states[current_idx].spectrum.contains(&token);
        let child = states[current_idx].trie_child;
        let depth = states[current_idx].depth;

        if learning {
            states[current_idx].observe(tick, token);
            return None;
        }

        if contains {
            states[current_idx].consume(tick, token);
            return None;
        }

        // Token rejected at this level.
        states[current_idx].rejected += 1;

        // Try cascading to child.
        if let Some(child_idx) = child {
            current_idx = child_idx;
            continue;
        }

        // No child — check depth limit.
        if depth >= MAX_TRIE_DEPTH {
            return None;
        }

        // Feed seed; create if needed.
        if states[current_idx].seed.is_none() {
            states[current_idx].seed = Some(Seed::new(tick));
        }
        states[current_idx].seed.as_mut().unwrap().feed(tick, token);

        if states[current_idx].seed.as_ref().unwrap().should_promote() {
            // Promote: create a new child.
            let seed = states[current_idx].seed.take().unwrap();
            let new_depth = depth + 1;
            let new_idx = states.len();

            let mut child_state = ConsumptionState::new(new_depth, tick, false);
            child_state.trie_parent = Some(current_idx);

            // Pre-load seed tokens into the child's observations.
            for &(t, tok) in &seed.entries {
                child_state.observe(t, tok);
            }

            states.push(child_state);
            states[current_idx].trie_child = Some(new_idx);

            return Some(new_idx);
        }

        return None;
    }
}

// ── depth_color ──────────────────────────────────────────────────────────────

/// Diagnostic rainbow: root = red, deeper = blue.
pub fn depth_color(depth: u16) -> [f32; 3] {
    let t = (depth as f32 / 6.0).clamp(0.0, 1.0);
    // red → green → blue
    let r = (1.0 - t).max(0.0);
    let g = if t < 0.5 { t * 2.0 } else { (1.0 - t) * 2.0 };
    let b = t;
    [r, g, b]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deposit_token_quantization() {
        let full = DepositToken::from_deposit(1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(full.density, 15);
        assert_eq!(full.rg, 0xFF);
        assert_eq!(full.b_reserved, 0xF0);

        let zero = DepositToken::from_deposit(0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq!(zero.density, 0);
        assert_eq!(zero.rg, 0);
        assert_eq!(zero.b_reserved, 0);
    }

    #[test]
    fn spectrum_crystallize_coverage() {
        let mut obs = HashMap::new();
        let a = DepositToken { density: 1, rg: 0x10, b_reserved: 0 };
        let b = DepositToken { density: 2, rg: 0x20, b_reserved: 0 };
        obs.insert(a, 60);
        obs.insert(b, 40);

        let spectrum = Spectrum::crystallize_from(&obs);
        // A alone is 60% which exceeds 50% target — only A should be kept.
        assert!(spectrum.contains(&a));
        assert!(!spectrum.contains(&b));
        assert_eq!(spectrum.len(), 1);
    }

    #[test]
    fn consumption_state_observe_and_crystallize() {
        // birth_tick=0 → window=max(1,0)=1, so one observe crystallizes.
        let mut state = ConsumptionState::new(0, 0, false);
        assert!(state.learning);
        assert_eq!(state.learning_window, 1);

        let tok = DepositToken { density: 5, rg: 0x33, b_reserved: 0x40 };
        state.observe(0, tok);
        assert!(!state.learning);
        assert!(state.spectrum.contains(&tok));
    }

    #[test]
    fn causal_window_scales_with_birth_tick() {
        let state = ConsumptionState::new(0, 500, false);
        assert_eq!(state.learning_window, 500);
    }

    #[test]
    fn seed_promotion() {
        let mut seed = Seed::new(0);
        let tok = DepositToken { density: 1, rg: 0, b_reserved: 0 };
        for i in 0..SEED_THRESHOLD {
            seed.feed(i as u64, tok);
        }
        assert!(seed.should_promote());
    }

    #[test]
    fn cascade_consume_and_reject() {
        let tok_a = DepositToken { density: 1, rg: 0x10, b_reserved: 0 };
        let tok_b = DepositToken { density: 2, rg: 0x20, b_reserved: 0 };

        // Root recognizes A.
        let mut root = ConsumptionState::new(0, 0, false);
        root.learning = false;
        let mut root_tokens = HashMap::new();
        root_tokens.insert(tok_a, ());
        root.spectrum = Spectrum { tokens: root_tokens };

        // Child recognizes B.
        let mut child = ConsumptionState::new(1, 0, false);
        child.learning = false;
        let mut child_tokens = HashMap::new();
        child_tokens.insert(tok_b, ());
        child.spectrum = Spectrum { tokens: child_tokens };
        child.trie_parent = Some(0);

        root.trie_child = Some(1);

        let mut states = vec![root, child];

        // A should be consumed by root.
        cascade_process(&mut states, 0, 10, tok_a);
        assert_eq!(states[0].consumed, 1);
        assert_eq!(states[1].consumed, 0);

        // B should cascade to child.
        cascade_process(&mut states, 0, 11, tok_b);
        assert_eq!(states[0].rejected, 1);
        assert_eq!(states[1].consumed, 1);
    }

    #[test]
    fn cascade_spawns_child_from_seed() {
        let tok_a = DepositToken { density: 1, rg: 0x10, b_reserved: 0 };
        let tok_b = DepositToken { density: 2, rg: 0x20, b_reserved: 0 };

        // Root recognizes only A.
        let mut root = ConsumptionState::new(0, 0, false);
        root.learning = false;
        let mut root_tokens = HashMap::new();
        root_tokens.insert(tok_a, ());
        root.spectrum = Spectrum { tokens: root_tokens };

        let mut states = vec![root];

        // Feed SEED_THRESHOLD B tokens — should eventually spawn a child.
        let mut spawned = None;
        for i in 0..SEED_THRESHOLD as u64 {
            if let Some(idx) = cascade_process(&mut states, 0, i, tok_b) {
                spawned = Some(idx);
            }
        }

        assert!(spawned.is_some());
        let child_idx = spawned.unwrap();
        assert_eq!(states.len(), 2);
        assert_eq!(states[child_idx].depth, 1);
        assert_eq!(states[child_idx].trie_parent, Some(0));
        assert_eq!(states[0].trie_child, Some(child_idx));
    }
}
