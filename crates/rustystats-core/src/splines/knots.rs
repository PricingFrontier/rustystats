//! Exact order statistics for spline knots (`specs/out-of-core-fit.md` §6.5).
//!
//! Interior knot i of m is `sorted[knot_rank(i, m, n)]` over the column sorted
//! by `f64::total_cmp`, duplicates kept. [`OrderStatistics`] finds those values
//! without ever holding the column: an MSD radix select over
//! [`total_order_key`], 16 key bits per pass, four passes over the data.
//! [`exact_order_statistics`] runs the same four passes over an in-memory
//! slice, so the in-memory and out-of-core paths place identical knots.

use crate::error::{Result, RustyStatsError};

/// Passes over the data a selection takes: one per 16-bit digit of the key.
pub const ORDER_STATISTIC_PASSES: usize = 4;

const DIGIT_BITS: u32 = 16;
const BUCKETS: usize = 1 << DIGIT_BITS;
const SIGN_BIT: u64 = 1 << 63;

/// 0-based rank of interior knot `i` (1-based, `1..=m`) of `m` among `n`
/// sorted values: `floor(i / (m + 1) · (n − 1))`, evaluated in `f64` exactly as
/// the knot placement always has.
pub fn knot_rank(i: usize, m: usize, n: usize) -> usize {
    let p = (i as f64) / ((m + 1) as f64);
    (p * n.saturating_sub(1) as f64) as usize
}

/// Map an `f64` to a `u64` whose unsigned order is `f64::total_cmp` order
/// (−0.0 below +0.0, every bit pattern distinct): `!bits` when the sign bit is
/// set, else `bits ^ (1 << 63)`.
pub fn total_order_key(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & SIGN_BIT != 0 {
        !bits
    } else {
        bits ^ SIGN_BIT
    }
}

/// Inverse of [`total_order_key`]; recovers the exact bit pattern.
pub fn from_total_order_key(key: u64) -> f64 {
    let bits = if key & SIGN_BIT != 0 {
        key ^ SIGN_BIT
    } else {
        !key
    };
    f64::from_bits(bits)
}

/// Lowest key bit of pass `pass`'s digit: pass j (0-based) reads key bits
/// `[48 − 16j, 64 − 16j)`.
fn digit_shift(pass: usize) -> u32 {
    64 - DIGIT_BITS * (pass as u32 + 1)
}

/// Key bits resolved before pass `pass`: the top `16 · pass` bits.
fn prefix_mask(pass: usize) -> u64 {
    !(u64::MAX >> (DIGIT_BITS * pass as u32))
}

fn protocol_error(message: String) -> RustyStatsError {
    RustyStatsError::InvalidValue(message)
}

/// Exact order statistics (0-based ranks into the `total_cmp`-sorted values,
/// duplicates kept) over values that arrive in batches, by MSD radix select.
///
/// The caller streams the same `n` values through four passes:
/// `begin_pass(j)`, any number of `observe` batches, `end_pass()`, for
/// j = 0..[`Self::passes_needed`]. Pass j counts, for every rank, the keys
/// whose resolved high bits match the rank's prefix in a 65,536-bucket
/// histogram of key bits `[48 − 16j, 64 − 16j)`. Each rank then moves to the
/// bucket holding it and subtracts the counts of the buckets below. There is no
/// early exit; after the fourth pass every key is fully resolved and
/// [`Self::results`] decodes it.
///
/// Memory: one histogram of 65,536 `u64` counts (512 KiB) per distinct live
/// prefix, held only while a pass is open. Pass 0 has a single prefix; later
/// passes have at most one per rank. Ranks sharing a prefix share a histogram.
///
/// Every pass must see the same multiset of values. A pass that sees a
/// different count, or a different count under a resolved prefix, is refused at
/// `end_pass`; the selector is then left as it was before the pass began, so the
/// pass can be run again.
#[derive(Clone, Debug)]
pub struct OrderStatistics {
    n: u64,
    /// Per rank, in the caller's order: the key bits resolved by completed
    /// passes (zero below them).
    prefixes: Vec<u64>,
    /// Per rank: its rank among the values sharing its prefix.
    within: Vec<u64>,
    /// Per rank: how many values share its prefix.
    shared: Vec<u64>,
    completed: usize,
    active: Option<ActivePass>,
}

#[derive(Clone, Debug)]
struct ActivePass {
    index: usize,
    seen: u64,
    /// One per distinct prefix, sorted by prefix.
    histograms: Vec<PrefixHistogram>,
}

#[derive(Clone, Debug)]
struct PrefixHistogram {
    prefix: u64,
    /// Values the previous pass counted under this prefix.
    expected: u64,
    counts: Box<[u64]>,
}

impl OrderStatistics {
    /// A selector for `ranks` (0-based, in any order, duplicates allowed) among
    /// `n` values. Refuses `n == 0` and any rank not below `n`.
    pub fn new(ranks: Vec<u64>, n: u64) -> Result<Self> {
        if n == 0 {
            return Err(RustyStatsError::EmptyInput(
                "order statistics need at least one value (n = 0)".to_string(),
            ));
        }
        if let Some(&rank) = ranks.iter().find(|&&rank| rank >= n) {
            return Err(RustyStatsError::InvalidValue(format!(
                "order statistic rank {rank} is out of range for {n} values \
                 (ranks are 0-based and must be below n)"
            )));
        }
        let count = ranks.len();
        Ok(Self {
            n,
            prefixes: vec![0; count],
            within: ranks,
            shared: vec![n; count],
            completed: 0,
            active: None,
        })
    }

    /// Passes over the data the selection takes: 4, or 0 when no ranks were
    /// requested.
    pub fn passes_needed(&self) -> usize {
        if self.prefixes.is_empty() {
            0
        } else {
            ORDER_STATISTIC_PASSES
        }
    }

    /// Open pass `pass` (0-based). Passes run in order, one at a time.
    pub fn begin_pass(&mut self, pass: usize) -> Result<()> {
        if let Some(active) = &self.active {
            return Err(protocol_error(format!(
                "order statistics pass {} is still open; end it before beginning pass {pass}",
                active.index
            )));
        }
        let needed = self.passes_needed();
        if self.completed >= needed {
            return Err(protocol_error(format!(
                "order statistics take {needed} passes and all have ended; got pass {pass}"
            )));
        }
        if pass != self.completed {
            return Err(protocol_error(format!(
                "order statistics expected pass {} of {needed} next, got pass {pass}",
                self.completed
            )));
        }
        let mut prefixes: Vec<(u64, u64)> = self
            .prefixes
            .iter()
            .copied()
            .zip(self.shared.iter().copied())
            .collect();
        prefixes.sort_unstable();
        prefixes.dedup_by_key(|&mut (prefix, _)| prefix);
        let histograms = prefixes
            .into_iter()
            .map(|(prefix, expected)| PrefixHistogram {
                prefix,
                expected,
                counts: vec![0; BUCKETS].into_boxed_slice(),
            })
            .collect();
        self.active = Some(ActivePass {
            index: pass,
            seen: 0,
            histograms,
        });
        Ok(())
    }

    /// Count one batch of values into the open pass. Refuses NaN and ±inf
    /// before counting anything, so a refused batch leaves the pass unchanged.
    pub fn observe(&mut self, values: &[f64]) -> Result<()> {
        if let Some((position, value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(RustyStatsError::InvalidValue(format!(
                "order statistics need finite values, got {value} at position {position} \
                 of the batch"
            )));
        }
        self.count(values)
    }

    /// Count a batch without the finiteness check. Every bit pattern has a key,
    /// so NaN and ±inf are ordered by `total_cmp` like any other value.
    fn count(&mut self, values: &[f64]) -> Result<()> {
        let active = self.active.as_mut().ok_or_else(|| {
            protocol_error("order statistics received values outside a pass".to_string())
        })?;
        let shift = digit_shift(active.index);
        let mask = prefix_mask(active.index);
        let digit = |key: u64| ((key >> shift) as usize) & (BUCKETS - 1);
        match active.histograms.as_mut_slice() {
            [only] => {
                for &value in values {
                    let key = total_order_key(value);
                    if key & mask == only.prefix {
                        only.counts[digit(key)] += 1;
                    }
                }
            }
            histograms => {
                for &value in values {
                    let key = total_order_key(value);
                    if let Ok(h) = histograms.binary_search_by_key(&(key & mask), |h| h.prefix) {
                        histograms[h].counts[digit(key)] += 1;
                    }
                }
            }
        }
        active.seen += values.len() as u64;
        Ok(())
    }

    /// Close the open pass and move every rank into the bucket holding it.
    pub fn end_pass(&mut self) -> Result<()> {
        let active = self.active.take().ok_or_else(|| {
            protocol_error("order statistics has no open pass to end".to_string())
        })?;
        if active.seen != self.n {
            return Err(protocol_error(format!(
                "order statistics pass {} saw {} values, expected {}; every pass must see \
                 the same values",
                active.index, active.seen, self.n
            )));
        }
        for histogram in &active.histograms {
            let total: u64 = histogram.counts.iter().sum();
            if total != histogram.expected {
                return Err(protocol_error(format!(
                    "order statistics pass {} counted {total} values under a resolved prefix \
                     that held {} in the previous pass; the values changed between passes",
                    active.index, histogram.expected
                )));
            }
        }

        let shift = digit_shift(active.index);
        let mut prefixes = self.prefixes.clone();
        let mut within = self.within.clone();
        let mut shared = self.shared.clone();
        let mut order: Vec<usize> = (0..prefixes.len()).collect();
        order.sort_unstable_by_key(|&r| (self.prefixes[r], self.within[r]));
        let mut pending = order.into_iter().peekable();
        for histogram in &active.histograms {
            // Ranks of this prefix in ascending order: one forward scan.
            let mut below = 0u64;
            let mut bucket = 0usize;
            while let Some(r) = pending.next_if(|&r| self.prefixes[r] == histogram.prefix) {
                while bucket < BUCKETS && below + histogram.counts[bucket] <= self.within[r] {
                    below += histogram.counts[bucket];
                    bucket += 1;
                }
                let Some(&count) = histogram.counts.get(bucket) else {
                    return Err(protocol_error(format!(
                        "order statistics rank fell outside its prefix's {} values",
                        histogram.expected
                    )));
                };
                prefixes[r] |= (bucket as u64) << shift;
                within[r] -= below;
                shared[r] = count;
            }
        }
        self.prefixes = prefixes;
        self.within = within;
        self.shared = shared;
        self.completed += 1;
        Ok(())
    }

    /// The order statistics, in the order the ranks were given. Available once
    /// every pass has ended.
    pub fn results(&self) -> Result<Vec<f64>> {
        let needed = self.passes_needed();
        if self.active.is_some() || self.completed < needed {
            return Err(protocol_error(format!(
                "order statistics are not resolved yet: {} of {needed} passes ended",
                self.completed
            )));
        }
        Ok(self
            .prefixes
            .iter()
            .map(|&key| from_total_order_key(key))
            .collect())
    }
}

/// `sorted[rank]` for each of `ranks` over `values` sorted by `f64::total_cmp`
/// (duplicates kept), by the same four passes [`OrderStatistics`] runs over
/// batches. Unlike [`OrderStatistics::observe`] it accepts NaN and ±inf, ordered
/// by `total_cmp`, so in-memory knot placement keeps today's result for every
/// input; the out-of-core passes require finite spline inputs (§6.5).
pub fn exact_order_statistics(values: &[f64], ranks: &[u64]) -> Result<Vec<f64>> {
    let mut selector = OrderStatistics::new(ranks.to_vec(), values.len() as u64)?;
    for pass in 0..selector.passes_needed() {
        selector.begin_pass(pass)?;
        selector.count(values)?;
        selector.end_pass()?;
    }
    selector.results()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64: a deterministic test stream without a dependency.
    struct TestRng(u64);

    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }
    }

    fn sorted(values: &[f64]) -> Vec<f64> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted
    }

    fn all_ranks(n: usize) -> Vec<u64> {
        (0..n as u64).collect()
    }

    /// Assert bit-identical equality with `sorted[rank]`.
    fn assert_selects(values: &[f64], ranks: &[u64]) {
        let reference = sorted(values);
        let selected = exact_order_statistics(values, ranks).expect("selection should succeed");
        assert_eq!(selected.len(), ranks.len());
        for (&rank, value) in ranks.iter().zip(&selected) {
            let expected = reference[rank as usize];
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "rank {rank}: got {value}, expected {expected}"
            );
        }
    }

    /// Run the batched protocol with a fresh random batching on every pass.
    fn select_in_batches(
        values: &[f64],
        ranks: &[u64],
        rng: &mut TestRng,
        max_batch: usize,
    ) -> Vec<f64> {
        let mut selector = OrderStatistics::new(ranks.to_vec(), values.len() as u64)
            .expect("ranks should be in range");
        for pass in 0..selector.passes_needed() {
            selector.begin_pass(pass).expect("pass should open");
            let mut start = 0;
            while start < values.len() {
                let len = 1 + rng.below(max_batch);
                let end = (start + len).min(values.len());
                selector
                    .observe(&values[start..end])
                    .expect("finite batch should be accepted");
                start = end;
            }
            selector.end_pass().expect("pass should close");
        }
        selector.results().expect("selection should be resolved")
    }

    #[test]
    fn knot_rank_matches_the_quantile_index() {
        for n in 1..40usize {
            for m in 1..12usize {
                for i in 1..=m {
                    let p = (i as f64) / ((m + 1) as f64);
                    let expected = (p * (n - 1) as f64) as usize;
                    assert_eq!(knot_rank(i, m, n), expected);
                    assert!(knot_rank(i, m, n) < n);
                }
            }
        }
    }

    #[test]
    fn total_order_key_follows_total_cmp_and_round_trips() {
        let specials = [
            f64::NEG_INFINITY,
            f64::MIN,
            -1e300,
            -1.0,
            -f64::MIN_POSITIVE,
            -f64::from_bits(1),
            -0.0,
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1.0,
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
            -f64::NAN,
        ];
        for &a in &specials {
            assert_eq!(
                from_total_order_key(total_order_key(a)).to_bits(),
                a.to_bits()
            );
            for &b in &specials {
                assert_eq!(
                    total_order_key(a).cmp(&total_order_key(b)),
                    a.total_cmp(&b),
                    "{a} vs {b}"
                );
            }
        }
        let mut rng = TestRng(7);
        for _ in 0..10_000 {
            let key = rng.next_u64();
            assert_eq!(total_order_key(from_total_order_key(key)), key);
        }
    }

    #[test]
    fn every_rank_of_tiny_inputs() {
        let pool = [3.5, -2.0, 0.0, 3.5, 1e-310, -7.25];
        for n in 1..=5 {
            let values = &pool[..n];
            assert_selects(values, &all_ranks(n));
            let mut reversed = values.to_vec();
            reversed.reverse();
            assert_selects(&reversed, &all_ranks(n));
        }
    }

    #[test]
    fn all_values_equal() {
        let values = vec![42.0; 1000];
        assert_selects(&values, &all_ranks(1000));
    }

    #[test]
    fn two_distinct_values() {
        let values: Vec<f64> = (0..501)
            .map(|i| if i % 3 == 0 { 2.5 } else { -1.5 })
            .collect();
        assert_selects(&values, &all_ranks(values.len()));
    }

    #[test]
    fn heavy_duplicates_exactly_at_a_rank() {
        // Ranks 100..=399 are all 5.0; select the run's edges and neighbours.
        let mut values: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        values.extend(std::iter::repeat_n(5.0, 300));
        values.extend((0..100).map(|i| 10.0 + i as f64));
        let mut rng = TestRng(11);
        for i in (1..values.len()).rev() {
            values.swap(i, rng.below(i + 1));
        }
        assert_selects(&values, &[99, 100, 101, 250, 398, 399, 400]);
    }

    #[test]
    fn signed_zeros_order_negative_first() {
        let values = [0.0, -0.0, 0.0, -0.0, -0.0];
        let selected = exact_order_statistics(&values, &[0, 1, 2, 3, 4]).expect("selection");
        let signs: Vec<bool> = selected.iter().map(|v| v.is_sign_negative()).collect();
        assert_eq!(signs, [true, true, true, false, false]);
        assert_selects(&values, &all_ranks(values.len()));
    }

    #[test]
    fn subnormals_and_extremes() {
        let values = [
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::from_bits(0x000F_FFFF_FFFF_FFFF),
            -f64::from_bits(3),
            f64::MIN_POSITIVE,
            1e300,
            -1e300,
            0.0,
            -0.0,
            f64::MAX,
            f64::MIN,
        ];
        assert_selects(&values, &all_ranks(values.len()));
    }

    #[test]
    fn negative_only_data() {
        let mut rng = TestRng(3);
        let values: Vec<f64> = (0..2000)
            .map(|_| -((rng.next_u64() >> 11) as f64) * 1e-9 - 1e-300)
            .collect();
        assert_selects(&values, &all_ranks(values.len()));
    }

    #[test]
    fn ranks_in_any_order_with_duplicates() {
        let values: Vec<f64> = (0..97).map(|i| ((i * 37) % 97) as f64 - 48.0).collect();
        assert_selects(&values, &[96, 0, 50, 50, 3, 96, 0, 49, 51]);
    }

    #[test]
    fn random_batches_match_sorting_on_every_pass() {
        let mut rng = TestRng(2024);
        for case in 0..30 {
            let n = 1 + rng.below(3000);
            let values: Vec<f64> = (0..n)
                .map(|_| match rng.below(4) {
                    // Arbitrary finite bit patterns span every exponent.
                    0 => loop {
                        let v = f64::from_bits(rng.next_u64());
                        if v.is_finite() {
                            break v;
                        }
                    },
                    // Clustered values share long key prefixes.
                    1 => 1000.0 + (rng.below(50) as f64) * 1e-12,
                    2 => -(rng.below(7) as f64),
                    _ => (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64,
                })
                .collect();
            let reference = sorted(&values);
            let mut ranks: Vec<u64> = (0..1 + rng.below(12))
                .map(|_| rng.below(n) as u64)
                .collect();
            ranks.push(0);
            ranks.push(n as u64 - 1);
            let max_batch = if case % 3 == 0 { 1 } else { 1 + rng.below(n) };
            let selected = select_in_batches(&values, &ranks, &mut rng, max_batch);
            for (&rank, value) in ranks.iter().zip(&selected) {
                assert_eq!(value.to_bits(), reference[rank as usize].to_bits());
            }
            assert_selects(&values, &ranks);
        }
    }

    #[test]
    fn non_finite_values_are_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut selector = OrderStatistics::new(vec![0], 3).expect("valid ranks");
            selector.begin_pass(0).expect("pass should open");
            let err = selector
                .observe(&[1.0, bad, 2.0])
                .expect_err("non-finite value should be refused");
            let message = err.to_string();
            assert!(message.contains("finite"), "{message}");
            assert!(message.contains("position 1"), "{message}");
            // The refused batch counted nothing; the pass can continue.
            selector.observe(&[1.0, 2.0, 3.0]).expect("finite batch");
            selector.end_pass().expect("pass should close");
        }
    }

    #[test]
    fn invalid_construction_is_refused() {
        assert!(OrderStatistics::new(vec![0], 0).is_err());
        assert!(OrderStatistics::new(vec![3], 3).is_err());
        assert!(exact_order_statistics(&[], &[0]).is_err());
        let empty = OrderStatistics::new(Vec::new(), 4).expect("no ranks is valid");
        assert_eq!(empty.passes_needed(), 0);
        assert_eq!(
            empty.results().expect("nothing to resolve"),
            Vec::<f64>::new()
        );
    }

    #[test]
    fn pass_protocol_is_enforced() {
        let values = [3.0, 1.0, 2.0];
        let mut selector = OrderStatistics::new(vec![1], 3).expect("valid ranks");
        assert_eq!(selector.passes_needed(), 4);
        assert!(selector.observe(&values).is_err(), "observe outside a pass");
        assert!(selector.end_pass().is_err(), "end without a pass");
        assert!(selector.begin_pass(1).is_err(), "passes run in order");
        selector.begin_pass(0).expect("pass 0 opens");
        assert!(selector.begin_pass(0).is_err(), "one pass at a time");
        selector.observe(&values[..2]).expect("batch");
        assert!(selector.end_pass().is_err(), "a short pass is refused");
        assert!(selector.results().is_err(), "unresolved");

        // The refused pass left the selector as it was: run it again.
        selector.begin_pass(0).expect("pass 0 reopens");
        selector.observe(&values).expect("batch");
        selector.end_pass().expect("pass 0 closes");
        selector.begin_pass(1).expect("pass 1 opens");
        selector.observe(&[3.0, 1.0, 1e6]).expect("batch");
        assert!(
            selector.end_pass().is_err(),
            "changed values under a resolved prefix are refused"
        );
        for pass in 1..4 {
            selector.begin_pass(pass).expect("pass opens");
            selector.observe(&values).expect("batch");
            selector.end_pass().expect("pass closes");
        }
        assert!(selector.begin_pass(4).is_err(), "only four passes");
        assert_eq!(selector.results().expect("resolved"), vec![2.0]);
    }
}
