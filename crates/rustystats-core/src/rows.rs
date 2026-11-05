//! Deterministic reductions over rows (`specs/out-of-core-fit.md` §5.3).
//!
//! Every sum over rows in a fit is reduced over a fixed partition of global row
//! indices into blocks of [`GRAM_BLOCK_ROWS`] rows. A block's rows are folded in
//! row order into one partial; partials are added into the accumulator in block
//! order. The result therefore depends on neither the thread count nor how the
//! rows arrive: a fit that streams batches of any size (out of core) and a fit
//! that holds every row (in memory) add exactly the same numbers in exactly the
//! same order.
//!
//! [`BlockReducer`] is the one implementation both modes use. Rows are fed in
//! global order, in slices of any length; a block that a slice ends inside stays
//! open and the next slice's first rows continue it, so no row is copied.

use std::ops::Range;

use rayon::prelude::*;

/// Rows per reduction block.
///
/// Part of the numerical identity of every fit: changing it changes results in
/// their last bits.
pub const GRAM_BLOCK_ROWS: usize = 4096;

/// A reduction partial: rows fold into it, and it adds into an accumulator.
pub trait Partial: Send {
    /// Reset every element to zero, keeping capacity.
    fn reset(&mut self);
    /// `self += other`, element by element in index order.
    fn add_assign(&mut self, other: &Self);
}

impl Partial for f64 {
    fn reset(&mut self) {
        *self = 0.0;
    }
    fn add_assign(&mut self, other: &Self) {
        *self += *other;
    }
}

impl Partial for Vec<f64> {
    fn reset(&mut self) {
        self.iter_mut().for_each(|value| *value = 0.0);
    }
    fn add_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.len(), other.len());
        for (value, add) in self.iter_mut().zip(other.iter()) {
            *value += *add;
        }
    }
}

/// A packed weighted Gram and its right-hand side: X'WX (upper triangle,
/// row-major packed), X'Wz and z'Wz.
#[derive(Debug, Clone, PartialEq)]
pub struct GramPartial {
    pub xtx_packed: Vec<f64>,
    pub xtz: Vec<f64>,
    pub ztwz: f64,
}

/// Entries in a packed upper triangle of a p×p matrix.
pub const fn packed_len(p: usize) -> usize {
    p * (p + 1) / 2
}

impl GramPartial {
    pub fn zeros(p: usize) -> Self {
        Self {
            xtx_packed: vec![0.0; packed_len(p)],
            xtz: vec![0.0; p],
            ztwz: 0.0,
        }
    }

    pub fn p(&self) -> usize {
        self.xtz.len()
    }
}

impl Partial for GramPartial {
    fn reset(&mut self) {
        self.xtx_packed.reset();
        self.xtz.reset();
        self.ztwz = 0.0;
    }
    fn add_assign(&mut self, other: &Self) {
        self.xtx_packed.add_assign(&other.xtx_packed);
        self.xtz.add_assign(&other.xtz);
        self.ztwz += other.ztwz;
    }
}

/// Streams rows in global order into fixed blocks and reduces them
/// deterministically.
///
/// Memory: the accumulator, one open partial and one partial per worker thread,
/// so `threads + 2` partials whatever the row count.
#[derive(Clone)]
pub struct BlockReducer<P: Partial + Clone> {
    accumulator: P,
    open: Option<P>,
    pool: Vec<P>,
    next_row: usize,
}

impl<P: Partial + Clone> BlockReducer<P> {
    /// A reducer whose partials are clones of `zero` (which must be all zeros), with
    /// one worker partial per thread of the current Rayon pool.
    pub fn new(zero: P) -> Self {
        Self::with_threads(zero, rayon::current_num_threads())
    }

    /// A reducer with `threads` worker partials: the size of the pool its `feed`
    /// runs in, when that is not the current one. The result does not depend on it.
    pub fn with_threads(zero: P, threads: usize) -> Self {
        Self {
            pool: vec![zero.clone(); threads.max(1)],
            open: None,
            accumulator: zero,
            next_row: 0,
        }
    }

    /// Rows fed so far.
    pub fn rows(&self) -> usize {
        self.next_row
    }

    /// Feed the next `n_rows` rows (global rows `rows()..rows() + n_rows`).
    ///
    /// `fold(partial, local_rows)` must add the rows `local_rows` of the current
    /// slice into `partial`, one row at a time in increasing order; local row
    /// `i` is global row `rows() + i` at the time of the call.
    pub fn feed<F>(&mut self, n_rows: usize, fold: F)
    where
        F: Fn(&mut P, Range<usize>) + Sync,
    {
        let mut local = 0usize;

        if let Some(open) = self.open.as_mut() {
            let block_end = (self.next_row / GRAM_BLOCK_ROWS + 1) * GRAM_BLOCK_ROWS;
            let take = (block_end - self.next_row).min(n_rows);
            fold(open, 0..take);
            local = take;
            self.next_row += take;
            if self.next_row.is_multiple_of(GRAM_BLOCK_ROWS) {
                let closed = self.open.take().expect("open block is present");
                self.accumulator.add_assign(&closed);
                self.recycle(closed);
            }
        }

        let full_blocks = (n_rows - local) / GRAM_BLOCK_ROWS;
        let threads = self.pool.len();
        let mut block = 0usize;
        while block < full_blocks {
            let wave = threads.min(full_blocks - block);
            let wave_start = local + block * GRAM_BLOCK_ROWS;
            self.pool[..wave]
                .par_iter_mut()
                .enumerate()
                .for_each(|(offset, partial)| {
                    partial.reset();
                    let start = wave_start + offset * GRAM_BLOCK_ROWS;
                    fold(partial, start..start + GRAM_BLOCK_ROWS);
                });
            for partial in &self.pool[..wave] {
                self.accumulator.add_assign(partial);
            }
            block += wave;
        }
        local += full_blocks * GRAM_BLOCK_ROWS;
        self.next_row += full_blocks * GRAM_BLOCK_ROWS;

        if local < n_rows {
            let mut partial = self.pool.pop().unwrap_or_else(|| self.accumulator.clone());
            partial.reset();
            fold(&mut partial, local..n_rows);
            self.next_row += n_rows - local;
            self.open = Some(partial);
        }
    }

    fn recycle(&mut self, mut partial: P) {
        partial.reset();
        self.pool.push(partial);
    }

    /// Close the last (partial) block and return the reduced value.
    pub fn finish(mut self) -> P {
        if let Some(open) = self.open.take() {
            self.accumulator.add_assign(&open);
        }
        self.accumulator
    }
}

/// Reduce `n_rows` rows held in memory: the single-slice case of [`BlockReducer`].
pub fn reduce_rows<P, F>(zero: P, n_rows: usize, fold: F) -> P
where
    P: Partial + Clone,
    F: Fn(&mut P, Range<usize>) + Sync,
{
    let mut reducer = BlockReducer::new(zero);
    reducer.feed(n_rows, fold);
    reducer.finish()
}

/// Number of reduction blocks covering `n_rows` rows.
pub fn block_count(n_rows: usize) -> usize {
    n_rows.div_ceil(GRAM_BLOCK_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize) -> Vec<f64> {
        // Values spanning many magnitudes so that summation order shows in the bits.
        (0..n)
            .map(|i| {
                let x = (i as f64 * 0.618_033_988_749_895).fract();
                (x - 0.5) * 10f64.powi((i % 23) as i32 - 11)
            })
            .collect()
    }

    fn reduce_in_slices(data: &[f64], slices: &[usize]) -> f64 {
        let mut reducer = BlockReducer::new(0.0f64);
        let mut start = 0usize;
        for &len in slices {
            let slice = &data[start..start + len];
            reducer.feed(len, |acc, rows| {
                for row in rows {
                    *acc += slice[row];
                }
            });
            start += len;
        }
        assert_eq!(start, data.len());
        reducer.finish()
    }

    fn reduce_with_threads(data: &[f64], threads: usize) -> f64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| {
            reduce_rows(0.0f64, data.len(), |acc, rows| {
                for row in rows {
                    *acc += data[row];
                }
            })
        })
    }

    fn reference(data: &[f64]) -> f64 {
        let mut total = 0.0;
        for block in data.chunks(GRAM_BLOCK_ROWS) {
            let mut partial = 0.0;
            for value in block {
                partial += *value;
            }
            total += partial;
        }
        total
    }

    #[test]
    fn result_matches_block_ordered_reference_bitwise() {
        let data = values(3 * GRAM_BLOCK_ROWS + 17);
        assert_eq!(
            reduce_with_threads(&data, 4).to_bits(),
            reference(&data).to_bits()
        );
    }

    #[test]
    fn result_does_not_depend_on_thread_count() {
        let data = values(10 * GRAM_BLOCK_ROWS + 1234);
        let one = reduce_with_threads(&data, 1).to_bits();
        for threads in [2, 3, 4, 7, 8] {
            assert_eq!(reduce_with_threads(&data, threads).to_bits(), one);
        }
        // Worker partials sized for another pool (or none) give the same bits.
        for partials in [0, 1, 3, 16] {
            let mut reducer = BlockReducer::with_threads(0.0f64, partials);
            assert_eq!(reducer.pool.len(), partials.max(1));
            reducer.feed(data.len(), |acc, rows| {
                for row in rows {
                    *acc += data[row];
                }
            });
            assert_eq!(reducer.finish().to_bits(), one, "{partials} partials");
        }
    }

    #[test]
    fn result_does_not_depend_on_slicing() {
        let n = 6 * GRAM_BLOCK_ROWS + 999;
        let data = values(n);
        let whole = reduce_in_slices(&data, &[n]).to_bits();
        let slicings: Vec<Vec<usize>> = vec![
            vec![n],
            vec![GRAM_BLOCK_ROWS - 1, 2, n - GRAM_BLOCK_ROWS - 1],
            vec![GRAM_BLOCK_ROWS, GRAM_BLOCK_ROWS, n - 2 * GRAM_BLOCK_ROWS],
            {
                let mut sizes = Vec::new();
                let mut left = n;
                let mut step = 1usize;
                while left > 0 {
                    let take = step.min(left);
                    sizes.push(take);
                    left -= take;
                    step = step * 3 % 9973 + 1;
                }
                sizes
            },
        ];
        for slices in slicings {
            assert_eq!(reduce_in_slices(&data, &slices).to_bits(), whole);
        }
    }

    #[test]
    fn empty_and_tiny_inputs() {
        assert_eq!(reduce_in_slices(&[], &[]), 0.0);
        assert_eq!(reduce_in_slices(&[], &[0]), 0.0);
        assert_eq!(reduce_in_slices(&[2.5], &[1]), 2.5);
        assert_eq!(block_count(0), 0);
        assert_eq!(block_count(GRAM_BLOCK_ROWS), 1);
        assert_eq!(block_count(GRAM_BLOCK_ROWS + 1), 2);
    }

    #[test]
    fn gram_partial_adds_elementwise() {
        let mut a = GramPartial::zeros(2);
        let mut b = GramPartial::zeros(2);
        b.xtx_packed = vec![1.0, 2.0, 3.0];
        b.xtz = vec![4.0, 5.0];
        b.ztwz = 6.0;
        a.add_assign(&b);
        a.add_assign(&b);
        assert_eq!(a.xtx_packed, vec![2.0, 4.0, 6.0]);
        assert_eq!(a.xtz, vec![8.0, 10.0]);
        assert_eq!(a.ztwz, 12.0);
        a.reset();
        assert_eq!(a, GramPartial::zeros(2));
    }
}
