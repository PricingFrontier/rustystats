//! Target-encoding training values in DEC-1's keyed order (`specs/out-of-core-fit.md` §6.6).
//!
//! Permutation m of a target-encoded term processes its participating rows (every
//! row, or a fold's training rows) in ascending `(mix64(mix64(seed + m) ^ r), r)`,
//! r the global row. A row's value is `(S_< + prior·pw) / (C_< + pw)`: S_< and C_<
//! are its label's running totals of the target (or claims) and of the count (or
//! exposure) over the rows before it in that order, added one row at a time. Values
//! are averaged over permutations in permutation order; one permutation is not
//! averaged.
//!
//! [`TermEncoder`] is the one implementation both fit modes use:
//!
//! - **In memory** it holds every record, sorts them in RAM and walks them.
//! - **Spilled** it sorts every `run_records` records in RAM and writes them as a
//!   run under a scratch directory (te_sort). Runs are merged by key with fan-in
//!   [`MERGE_FAN_IN`] and 1 MiB buffers until at most that many remain (te_merge).
//!   The last merge streams into the walk, which writes `(row, value)` runs sorted
//!   by row (te_walk); those are merged by row into the permutation's row-aligned
//!   float64 column (te_merge), and the permutation columns are averaged into the
//!   term's column (te_average).
//!
//! The `mix64` key is a bijection of the row, so keys are unique: every route walks
//! the same order and adds the same totals, whatever the run count.
//!
//! The prior and the "auto" prior weight are block-ordered reductions
//! ([`PriorSums`]); the per-label totals prediction uses are sums in row order
//! ([`LevelTotals`]).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::slice::ParallelSliceMut;

use crate::error::{Result, RustyStatsError};
use crate::rows::BlockReducer;
use crate::solvers::passes::mix64;

/// Bytes of a sort record: key, row, code (with padding), target and exposure.
pub const RECORD_BYTES: usize = 40;
/// Bytes of a walk output record: row and value.
pub const ROW_VALUE_BYTES: usize = 16;
/// Runs merged at once.
pub const MERGE_FAN_IN: usize = 64;
/// Buffer bytes of each merge input and of the merge output.
pub const MERGE_BUFFER_BYTES: usize = 1 << 20;
/// Records a spill stage processes between progress reports.
pub const PROGRESS_RECORDS: u64 = 1 << 20;

/// The sort key of global row `row` in permutation `permutation` (DEC-1).
#[inline]
pub fn record_key(seed: u64, permutation: u64, row: u64) -> u64 {
    mix64(mix64(seed.wrapping_add(permutation)) ^ row)
}

/// A spill stage, reported to the progress callback with the records done and the
/// stage's total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpillStage {
    Merge,
    Walk,
    Average,
}

impl SpillStage {
    /// The pass kind `on_batch` reports for the stage (§4.4).
    pub fn kind(self) -> &'static str {
        match self {
            SpillStage::Merge => "te_merge",
            SpillStage::Walk => "te_walk",
            SpillStage::Average => "te_average",
        }
    }
}

/// Called as a spill stage runs with (stage, records done, the stage's records,
/// starting): once with `starting` true before the stage reads anything, then after
/// every [`PROGRESS_RECORDS`] records and once at its end. An error cancels the stage.
pub type Progress<'a> = dyn FnMut(SpillStage, u64, u64, bool) -> Result<()> + 'a;

/// How a term's values are computed from its statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TermSettings {
    pub prior: f64,
    pub prior_weight: f64,
    /// Totals of claims and exposure instead of target and count.
    pub exposure_weighted: bool,
    /// Labels; every code is below it.
    pub n_levels: usize,
}

/// One participating row of one permutation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub key: u64,
    /// The row's position among the participating rows. It increases with the
    /// global row, so `(key, row)` orders records as `(key, r)` does.
    pub row: u64,
    pub code: u32,
    pub target: f64,
    pub exposure: f64,
}

/// A record of a run file: a fixed size, little-endian, and a total order.
trait Fixed: Copy {
    const BYTES: usize;
    fn order(&self) -> (u64, u64);
    fn put(&self, out: &mut [u8]);
    fn get(bytes: &[u8]) -> Self;
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
}

impl Fixed for Record {
    const BYTES: usize = RECORD_BYTES;

    fn order(&self) -> (u64, u64) {
        (self.key, self.row)
    }

    fn put(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.key.to_le_bytes());
        out[8..16].copy_from_slice(&self.row.to_le_bytes());
        out[16..20].copy_from_slice(&self.code.to_le_bytes());
        out[20..24].copy_from_slice(&[0u8; 4]);
        out[24..32].copy_from_slice(&self.target.to_le_bytes());
        out[32..40].copy_from_slice(&self.exposure.to_le_bytes());
    }

    fn get(bytes: &[u8]) -> Self {
        Record {
            key: u64_at(bytes, 0),
            row: u64_at(bytes, 8),
            code: u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")),
            target: f64::from_bits(u64_at(bytes, 24)),
            exposure: f64::from_bits(u64_at(bytes, 32)),
        }
    }
}

/// A walked value and the row it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RowValue {
    row: u64,
    value: f64,
}

impl Fixed for RowValue {
    const BYTES: usize = ROW_VALUE_BYTES;

    fn order(&self) -> (u64, u64) {
        (self.row, 0)
    }

    fn put(&self, out: &mut [u8]) {
        out[0..8].copy_from_slice(&self.row.to_le_bytes());
        out[8..16].copy_from_slice(&self.value.to_le_bytes());
    }

    fn get(bytes: &[u8]) -> Self {
        RowValue {
            row: u64_at(bytes, 0),
            value: f64::from_bits(u64_at(bytes, 8)),
        }
    }
}

// ── The walk ─────────────────────────────────────────────────────────────────

/// Running per-label totals; each record's value uses the totals before it.
struct Walk {
    settings: TermSettings,
    sums: Vec<f64>,
    counts: Vec<usize>,
    exposures: Vec<f64>,
}

impl Walk {
    fn new(settings: TermSettings) -> Self {
        let n = settings.n_levels;
        Walk {
            settings,
            sums: vec![0.0; n],
            counts: if settings.exposure_weighted {
                Vec::new()
            } else {
                vec![0; n]
            },
            exposures: if settings.exposure_weighted {
                vec![0.0; n]
            } else {
                Vec::new()
            },
        }
    }

    /// The record's value, then its contribution to its label's totals: the
    /// expressions of `compute_ordered_target_stats` and
    /// `compute_ordered_exposure_weighted_stats`.
    #[inline]
    fn next(&mut self, record: &Record) -> f64 {
        let code = record.code as usize;
        let prior = self.settings.prior;
        let prior_weight = self.settings.prior_weight;
        if self.settings.exposure_weighted {
            let claims_before = self.sums[code];
            let exposure_before = self.exposures[code];
            let value = (claims_before + prior * prior_weight) / (exposure_before + prior_weight);
            self.sums[code] += record.target;
            self.exposures[code] += record.exposure;
            value
        } else {
            let sum_before = self.sums[code];
            let count_before = self.counts[code];
            let value = (sum_before + prior * prior_weight) / (count_before as f64 + prior_weight);
            self.sums[code] += record.target;
            self.counts[code] += 1;
            value
        }
    }
}

/// The mean of one row's permutation values, in permutation order, as
/// `target_encode` averages them.
#[inline]
fn average(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

// ── Spill files ──────────────────────────────────────────────────────────────

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

fn spill_error(path: &Path, error: std::io::Error) -> RustyStatsError {
    RustyStatsError::Spill(format!("{}: {error}", path.display()))
}

/// A term's files under the scratch directory. Files are named by
/// (stage, index), so only index ranges are kept, and every file still on disk
/// is removed when this is dropped.
struct Scratch {
    dir: PathBuf,
    prefix: String,
    /// Per stage: the indices of its files still on disk, `start..end`.
    live: Vec<(String, usize, usize)>,
}

impl Scratch {
    fn new(dir: &Path) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let prefix = format!(
            "rustystats-te-{}-{nanos:x}-{}",
            std::process::id(),
            SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        Scratch {
            dir: dir.to_path_buf(),
            prefix,
            live: Vec::new(),
        }
    }

    fn path(&self, stage: &str, index: usize) -> PathBuf {
        self.dir.join(format!("{}-{stage}-{index}", self.prefix))
    }

    /// Create the next file of `stage`, exclusively.
    fn create(&mut self, stage: &str) -> Result<(PathBuf, File)> {
        let index = match self.live.iter_mut().find(|(name, _, _)| name == stage) {
            Some(entry) => {
                entry.2 += 1;
                entry.2 - 1
            }
            None => {
                self.live.push((stage.to_string(), 0, 1));
                0
            }
        };
        let path = self.path(stage, index);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| spill_error(&path, error))?;
        Ok((path, file))
    }

    /// Indices of `stage`'s files on disk.
    fn range(&self, stage: &str) -> (usize, usize) {
        self.live
            .iter()
            .find(|(name, _, _)| name == stage)
            .map(|(_, start, end)| (*start, *end))
            .unwrap_or((0, 0))
    }

    /// Remove `stage`'s files below `end`, which have been consumed.
    fn release(&mut self, stage: &str, end: usize) -> Result<()> {
        let Some(position) = self.live.iter().position(|(name, _, _)| name == stage) else {
            return Ok(());
        };
        while self.live[position].1 < end.min(self.live[position].2) {
            let path = self.path(stage, self.live[position].1);
            self.live[position].1 += 1;
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(spill_error(&path, error)),
            }
        }
        if self.live[position].1 >= self.live[position].2 {
            self.live.remove(position);
        }
        Ok(())
    }

    /// Hand `stage`'s only file to the caller, who removes it.
    fn forget(&mut self, stage: &str) {
        self.live.retain(|(name, _, _)| name != stage);
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for (stage, start, end) in std::mem::take(&mut self.live) {
            for index in start..end {
                let _ = std::fs::remove_file(self.path(&stage, index));
            }
        }
    }
}

struct RunWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    buffer: Vec<u8>,
}

impl RunWriter {
    fn new(path: PathBuf, file: File) -> Self {
        RunWriter {
            path,
            writer: BufWriter::with_capacity(MERGE_BUFFER_BYTES, file),
            buffer: vec![0u8; RECORD_BYTES],
        }
    }

    fn push<T: Fixed>(&mut self, record: &T) -> Result<()> {
        record.put(&mut self.buffer[..T::BYTES]);
        self.writer
            .write_all(&self.buffer[..T::BYTES])
            .map_err(|error| spill_error(&self.path, error))
    }

    fn push_f64(&mut self, value: f64) -> Result<()> {
        self.writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| spill_error(&self.path, error))
    }

    fn finish(mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|error| spill_error(&self.path, error))
    }
}

struct RunReader {
    path: PathBuf,
    reader: BufReader<File>,
    buffer: Vec<u8>,
}

impl RunReader {
    fn open(path: PathBuf) -> Result<Self> {
        let file = File::open(&path).map_err(|error| spill_error(&path, error))?;
        Ok(RunReader {
            path,
            reader: BufReader::with_capacity(MERGE_BUFFER_BYTES, file),
            buffer: vec![0u8; RECORD_BYTES],
        })
    }

    /// The next `bytes` bytes, or None at the end of the file.
    fn next_bytes(&mut self, bytes: usize) -> Result<Option<&[u8]>> {
        let mut filled = 0;
        while filled < bytes {
            match self.reader.read(&mut self.buffer[filled..bytes]) {
                Ok(0) => break,
                Ok(read) => filled += read,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(spill_error(&self.path, error)),
            }
        }
        match filled {
            0 => Ok(None),
            n if n == bytes => Ok(Some(&self.buffer[..bytes])),
            _ => Err(RustyStatsError::Spill(format!(
                "{}: the file ends inside a record",
                self.path.display()
            ))),
        }
    }

    fn next<T: Fixed>(&mut self) -> Result<Option<T>> {
        Ok(self.next_bytes(T::BYTES)?.map(T::get))
    }

    fn next_f64(&mut self) -> Result<Option<f64>> {
        Ok(self
            .next_bytes(8)?
            .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("8 bytes"))))
    }
}

/// Counts a stage's records and reports progress every [`PROGRESS_RECORDS`].
struct Counter<'p, 'a> {
    stage: SpillStage,
    done: u64,
    total: u64,
    progress: &'p mut Progress<'a>,
}

impl<'p, 'a> Counter<'p, 'a> {
    /// Announce the stage and count its records from zero.
    fn start(stage: SpillStage, total: u64, progress: &'p mut Progress<'a>) -> Result<Self> {
        progress(stage, 0, total, true)?;
        Ok(Counter {
            stage,
            done: 0,
            total,
            progress,
        })
    }

    #[inline]
    fn tick(&mut self) -> Result<()> {
        self.done += 1;
        if self.done.is_multiple_of(PROGRESS_RECORDS) {
            (self.progress)(self.stage, self.done, self.total, false)?;
        }
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        (self.progress)(self.stage, self.done, self.total, false)
    }
}

/// Stream the records of sorted runs in their total order into `emit`.
fn merge_runs<T: Fixed>(
    readers: &mut [RunReader],
    mut emit: impl FnMut(T) -> Result<()>,
) -> Result<()> {
    let mut heads: Vec<Option<T>> = Vec::with_capacity(readers.len());
    let mut heap = BinaryHeap::with_capacity(readers.len());
    for (index, reader) in readers.iter_mut().enumerate() {
        let head = reader.next::<T>()?;
        if let Some(record) = head {
            heap.push(Reverse((record.order(), index)));
        }
        heads.push(head);
    }
    while let Some(Reverse((_, index))) = heap.pop() {
        let record = heads[index].take().expect("a queued run has a head");
        let next = readers[index].next::<T>()?;
        if let Some(following) = next {
            if following.order() <= record.order() {
                return Err(RustyStatsError::Spill(format!(
                    "{}: a run is not sorted",
                    readers[index].path.display()
                )));
            }
            heap.push(Reverse((following.order(), index)));
        }
        heads[index] = next;
        emit(record)?;
    }
    Ok(())
}

/// Merge `stage`'s runs in groups of [`MERGE_FAN_IN`] until at most that many
/// remain; returns the stage holding them. Each group's inputs are removed once
/// merged.
fn reduce_runs<T: Fixed>(
    scratch: &mut Scratch,
    mut stage: String,
    family: &str,
    records: u64,
    progress: &mut Progress<'_>,
) -> Result<String> {
    let mut level = 0usize;
    loop {
        let (start, end) = scratch.range(&stage);
        if end - start <= MERGE_FAN_IN {
            return Ok(stage);
        }
        level += 1;
        let next = format!("{family}{level}");
        let mut counter = Counter::start(SpillStage::Merge, records, &mut *progress)?;
        let mut group_start = start;
        while group_start < end {
            let group_end = (group_start + MERGE_FAN_IN).min(end);
            let mut readers = (group_start..group_end)
                .map(|index| RunReader::open(scratch.path(&stage, index)))
                .collect::<Result<Vec<_>>>()?;
            let (path, file) = scratch.create(&next)?;
            let mut writer = RunWriter::new(path, file);
            merge_runs::<T>(&mut readers, |record| {
                writer.push(&record)?;
                counter.tick()
            })?;
            writer.finish()?;
            drop(readers);
            scratch.release(&stage, group_end)?;
            group_start = group_end;
        }
        counter.end()?;
        stage = next;
    }
}

// ── A term's permutations ────────────────────────────────────────────────────

/// Where a term's values are kept.
enum Store {
    /// Every record in RAM; each finished permutation's values.
    Memory { columns: Vec<Vec<f64>> },
    /// Runs of `run_records` records under a scratch directory; finished
    /// permutations are column files there.
    Spill {
        scratch: Scratch,
        run_records: usize,
        columns: usize,
    },
}

/// A target-encoded term's training values, one permutation at a time.
pub struct TermEncoder {
    settings: TermSettings,
    seed: u64,
    n_permutations: usize,
    store: Store,
    /// The permutation being sorted, its unsorted records, and its participating
    /// rows so far.
    current: Option<usize>,
    records: Vec<Record>,
    rows: u64,
    /// Participating rows of the finished permutations.
    n_rows: Option<u64>,
}

/// A term's finished training values: in RAM, or a column file the caller now owns
/// (row-aligned little-endian float64, one value per participating row).
#[derive(Debug)]
pub enum TermValues {
    Memory(Vec<f64>),
    File(PathBuf),
}

impl TermEncoder {
    /// A term encoder holding its records in RAM (`spill` None), or spilling runs of
    /// `run_records` records under `spill`'s directory.
    pub fn new(
        settings: TermSettings,
        seed: u64,
        n_permutations: usize,
        spill: Option<(&Path, usize)>,
    ) -> Result<Self> {
        if n_permutations == 0 {
            return Err(RustyStatsError::InvalidValue(
                "target_encoding n_permutations must be a positive integer.".to_string(),
            ));
        }
        if u32::try_from(settings.n_levels).is_err() {
            return Err(RustyStatsError::InvalidValue(format!(
                "a target-encoded term has {} labels, more than a u32 code holds",
                settings.n_levels
            )));
        }
        let store = match spill {
            None => Store::Memory {
                columns: Vec::with_capacity(n_permutations),
            },
            Some((dir, run_records)) => {
                if run_records == 0 {
                    return Err(RustyStatsError::InvalidValue(
                        "te_run_records must be positive".to_string(),
                    ));
                }
                Store::Spill {
                    scratch: Scratch::new(dir),
                    run_records,
                    columns: 0,
                }
            }
        };
        Ok(TermEncoder {
            settings,
            seed,
            n_permutations,
            store,
            current: None,
            records: Vec::new(),
            rows: 0,
            n_rows: None,
        })
    }

    fn finished(&self) -> usize {
        match &self.store {
            Store::Memory { columns } => columns.len(),
            Store::Spill { columns, .. } => *columns,
        }
    }

    /// Start permutation `permutation` (the next one, in order).
    pub fn begin(&mut self, permutation: usize) -> Result<()> {
        if self.current.is_some() || permutation != self.finished() {
            return Err(RustyStatsError::InvalidValue(format!(
                "permutation {permutation} cannot start: {} are finished{}",
                self.finished(),
                if self.current.is_some() {
                    " and one is open"
                } else {
                    ""
                }
            )));
        }
        if permutation >= self.n_permutations {
            return Err(RustyStatsError::InvalidValue(format!(
                "the term has {} permutations",
                self.n_permutations
            )));
        }
        self.current = Some(permutation);
        self.records.clear();
        self.rows = 0;
        Ok(())
    }

    /// Add the next rows: global rows `first_row..first_row + codes.len()`, in
    /// order. Rows `include` marks false take no part.
    pub fn push(
        &mut self,
        first_row: u64,
        codes: &[i32],
        target: &[f64],
        exposure: Option<&[f64]>,
        include: Option<&[bool]>,
    ) -> Result<()> {
        let permutation = self
            .current
            .ok_or_else(|| RustyStatsError::InvalidValue("no permutation is open".to_string()))?
            as u64;
        let n = codes.len();
        if target.len() != n
            || exposure.is_some_and(|e| e.len() != n)
            || include.is_some_and(|m| m.len() != n)
        {
            return Err(RustyStatsError::dim_mismatch(
                n,
                target.len(),
                "target-encoding codes vs target, exposure and mask",
            ));
        }
        if self.settings.exposure_weighted && exposure.is_none() {
            return Err(RustyStatsError::InvalidValue(
                "an exposure-weighted term needs exposure".to_string(),
            ));
        }
        for i in 0..n {
            if include.is_some_and(|m| !m[i]) {
                continue;
            }
            let code = codes[i];
            if code < 0 || code as usize >= self.settings.n_levels {
                return Err(RustyStatsError::InvalidValue(format!(
                    "a training row has label code {code}, outside the term's {} labels",
                    self.settings.n_levels
                )));
            }
            self.records.push(Record {
                key: record_key(self.seed, permutation, first_row + i as u64),
                row: self.rows,
                code: code as u32,
                target: target[i],
                exposure: exposure.map_or(0.0, |e| e[i]),
            });
            self.rows += 1;
            if let Store::Spill {
                scratch,
                run_records,
                ..
            } = &mut self.store
            {
                if self.records.len() >= *run_records {
                    write_run(scratch, &mut self.records)?;
                }
            }
        }
        Ok(())
    }

    /// Finish the open permutation: walk its records in key order into its values.
    pub fn end(&mut self, progress: &mut Progress<'_>) -> Result<()> {
        let permutation = self
            .current
            .take()
            .ok_or_else(|| RustyStatsError::InvalidValue("no permutation is open".to_string()))?;
        if let Some(n) = self.n_rows {
            if n != self.rows {
                return Err(RustyStatsError::InvalidValue(format!(
                    "permutation {permutation} has {} rows; earlier ones had {n}",
                    self.rows
                )));
            }
        }
        self.n_rows = Some(self.rows);
        let rows = self.rows;
        let settings = self.settings;
        match &mut self.store {
            Store::Memory { columns } => {
                let mut records = std::mem::take(&mut self.records);
                records.par_sort_unstable_by_key(Fixed::order);
                let mut walk = Walk::new(settings);
                let mut values = vec![0.0; rows as usize];
                for record in &records {
                    values[record.row as usize] = walk.next(record);
                }
                columns.push(values);
            }
            Store::Spill {
                scratch,
                run_records,
                columns,
            } => {
                if !self.records.is_empty() {
                    write_run(scratch, &mut self.records)?;
                }
                self.records = Vec::new();
                walk_spilled(
                    scratch,
                    settings,
                    *run_records,
                    rows,
                    &format!("p{permutation}"),
                    progress,
                )?;
                *columns += 1;
            }
        }
        Ok(())
    }

    /// Average the permutations into the term's values (te_average). A spilled
    /// term's column file is handed to the caller.
    pub fn finish(mut self, progress: &mut Progress<'_>) -> Result<TermValues> {
        if self.current.is_some() || self.finished() != self.n_permutations {
            return Err(RustyStatsError::InvalidValue(format!(
                "{} of {} permutations are finished",
                self.finished(),
                self.n_permutations
            )));
        }
        let rows = self.n_rows.unwrap_or(0);
        match &mut self.store {
            Store::Memory { columns } => {
                let columns = std::mem::take(columns);
                if columns.len() == 1 {
                    return Ok(TermValues::Memory(
                        columns.into_iter().next().expect("one column"),
                    ));
                }
                let mut row_values = vec![0.0; columns.len()];
                let values = (0..rows as usize)
                    .map(|i| {
                        for (slot, column) in row_values.iter_mut().zip(columns.iter()) {
                            *slot = column[i];
                        }
                        average(&row_values)
                    })
                    .collect();
                Ok(TermValues::Memory(values))
            }
            Store::Spill { scratch, .. } => {
                let m = self.n_permutations;
                if m == 1 {
                    let path = scratch.path("p0", 0);
                    scratch.forget("p0");
                    return Ok(TermValues::File(path));
                }
                let mut readers = (0..m)
                    .map(|perm| RunReader::open(scratch.path(&format!("p{perm}"), 0)))
                    .collect::<Result<Vec<_>>>()?;
                let (path, file) = scratch.create("t")?;
                let mut writer = RunWriter::new(path.clone(), file);
                let mut counter = Counter::start(SpillStage::Average, rows, progress)?;
                let mut row_values = vec![0.0; m];
                for _ in 0..rows {
                    for (slot, reader) in row_values.iter_mut().zip(readers.iter_mut()) {
                        *slot = reader.next_f64()?.ok_or_else(|| {
                            RustyStatsError::Spill(format!(
                                "{}: a permutation column ends early",
                                reader.path.display()
                            ))
                        })?;
                    }
                    writer.push_f64(average(&row_values))?;
                    counter.tick()?;
                }
                writer.finish()?;
                counter.end()?;
                drop(readers);
                for perm in 0..m {
                    scratch.release(&format!("p{perm}"), 1)?;
                }
                scratch.forget("t");
                Ok(TermValues::File(path))
            }
        }
    }
}

/// Sort `records` by key and write them as the next key run (te_sort).
fn write_run(scratch: &mut Scratch, records: &mut Vec<Record>) -> Result<()> {
    records.par_sort_unstable_by_key(Fixed::order);
    let (path, file) = scratch.create("k0")?;
    let mut writer = RunWriter::new(path, file);
    for record in records.iter() {
        writer.push(record)?;
    }
    writer.finish()?;
    records.clear();
    Ok(())
}

/// Sort the walk's values by row and write them as the next row run (te_walk).
fn write_row_run(scratch: &mut Scratch, values: &mut Vec<RowValue>) -> Result<()> {
    values.par_sort_unstable_by_key(Fixed::order);
    let (path, file) = scratch.create("r0")?;
    let mut writer = RunWriter::new(path, file);
    for value in values.iter() {
        writer.push(value)?;
    }
    writer.finish()?;
    values.clear();
    Ok(())
}

/// The spilled permutation's runs to its column file (stage `column`, index 0):
/// key merges, the walk, row merges.
fn walk_spilled(
    scratch: &mut Scratch,
    settings: TermSettings,
    run_records: usize,
    rows: u64,
    column: &str,
    progress: &mut Progress<'_>,
) -> Result<()> {
    // te_merge by key, until the walk can merge what remains.
    let key_stage = reduce_runs::<Record>(scratch, "k0".to_string(), "k", rows, progress)?;

    // te_walk: the last key merge, walked in key order, written as row runs.
    {
        let (start, end) = scratch.range(&key_stage);
        let mut readers = (start..end)
            .map(|index| RunReader::open(scratch.path(&key_stage, index)))
            .collect::<Result<Vec<_>>>()?;
        let mut walk = Walk::new(settings);
        let mut buffer: Vec<RowValue> = Vec::with_capacity(run_records.min(rows as usize));
        let mut counter = Counter::start(SpillStage::Walk, rows, &mut *progress)?;
        merge_runs::<Record>(&mut readers, |record| {
            if record.code as usize >= settings.n_levels {
                return Err(RustyStatsError::Spill(
                    "a key run holds a label code outside the term".to_string(),
                ));
            }
            let value = walk.next(&record);
            buffer.push(RowValue {
                row: record.row,
                value,
            });
            if buffer.len() >= run_records {
                write_row_run(scratch, &mut buffer)?;
            }
            counter.tick()
        })?;
        if !buffer.is_empty() {
            write_row_run(scratch, &mut buffer)?;
        }
        counter.end()?;
        drop(readers);
        scratch.release(&key_stage, end)?;
    }

    // te_merge by row, then the final stage writes the row-aligned column.
    let row_stage = reduce_runs::<RowValue>(scratch, "r0".to_string(), "r", rows, progress)?;
    let (start, end) = scratch.range(&row_stage);
    let mut readers = (start..end)
        .map(|index| RunReader::open(scratch.path(&row_stage, index)))
        .collect::<Result<Vec<_>>>()?;
    let (path, file) = scratch.create(column)?;
    let mut writer = RunWriter::new(path.clone(), file);
    let mut counter = Counter::start(SpillStage::Merge, rows, progress)?;
    let mut expected = 0u64;
    merge_runs::<RowValue>(&mut readers, |value| {
        if value.row != expected {
            return Err(RustyStatsError::Spill(format!(
                "{}: row {expected} has no value",
                path.display()
            )));
        }
        expected += 1;
        writer.push_f64(value.value)?;
        counter.tick()
    })?;
    if expected != rows {
        return Err(RustyStatsError::Spill(format!(
            "{}: {expected} of {rows} rows have values",
            path.display()
        )));
    }
    writer.finish()?;
    counter.end()?;
    drop(readers);
    scratch.release(&row_stage, end)?;
    Ok(())
}

// ── Statistics ───────────────────────────────────────────────────────────────

/// Slots of [`PriorSums`]' block-ordered partial.
const SUM_TARGET: usize = 0;
const SUM_EXPOSURE: usize = 1;
const SUM_FINITE_TARGET: usize = 2;
const SUM_PAIR_EXPOSURE: usize = 3;
const SUM_PAIR_TARGET: usize = 4;

/// The prior and the "auto" prior weight's sums over a term's participating rows:
/// block-ordered reductions over global rows (§5.3), with exact counts and
/// extremes.
#[derive(Clone)]
pub struct PriorSums {
    reducer: BlockReducer<Vec<f64>>,
    exposure_weighted: bool,
    rows: u64,
    finite: u64,
    min: f64,
    max: f64,
}

impl PriorSums {
    pub fn new(exposure_weighted: bool) -> Self {
        PriorSums {
            reducer: BlockReducer::new(vec![0.0; 5]),
            exposure_weighted,
            rows: 0,
            finite: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add the next rows in global order; rows `include` marks false take no part
    /// but keep their place in the row order.
    pub fn update(
        &mut self,
        target: &[f64],
        exposure: Option<&[f64]>,
        include: Option<&[bool]>,
    ) -> Result<()> {
        let n = target.len();
        if exposure.is_some_and(|e| e.len() != n) || include.is_some_and(|m| m.len() != n) {
            return Err(RustyStatsError::dim_mismatch(
                n,
                exposure.map_or(n, <[f64]>::len),
                "target-encoding prior target vs exposure and mask",
            ));
        }
        if self.exposure_weighted && exposure.is_none() {
            return Err(RustyStatsError::InvalidValue(
                "an exposure-weighted term needs exposure".to_string(),
            ));
        }
        for i in 0..n {
            if include.is_some_and(|m| !m[i]) {
                continue;
            }
            self.rows += 1;
            let t = target[i];
            if t.is_finite() {
                self.finite += 1;
                self.min = self.min.min(t);
                self.max = self.max.max(t);
            }
        }
        self.reducer.feed(n, |sums, range| {
            for i in range {
                if include.is_some_and(|m| !m[i]) {
                    continue;
                }
                let t = target[i];
                sums[SUM_TARGET] += t;
                if t.is_finite() {
                    sums[SUM_FINITE_TARGET] += t;
                }
                if let Some(e) = exposure {
                    let e = e[i];
                    sums[SUM_EXPOSURE] += e;
                    if t.is_finite() && e.is_finite() && e > 0.0 {
                        sums[SUM_PAIR_EXPOSURE] += e;
                        sums[SUM_PAIR_TARGET] += t;
                    }
                }
            }
        });
        Ok(())
    }

    /// Participating rows so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    fn sums(&self) -> Vec<f64> {
        self.reducer.clone().finish()
    }

    /// The prior: the mean target, or total claims over total exposure.
    pub fn prior(&self) -> f64 {
        let sums = self.sums();
        if self.exposure_weighted {
            sums[SUM_TARGET] / sums[SUM_EXPOSURE].max(1e-10)
        } else {
            sums[SUM_TARGET] / self.rows as f64
        }
    }

    /// `prior_weight="auto"` (`interactions._resolve_target_encoding_prior_weight`):
    /// 5 / rate clipped to [20, 1000] with exposure, 5 / the minority share clipped
    /// to [20, 500] for a target in [0, 1], else 50.
    pub fn auto_prior_weight(&self) -> f64 {
        if self.finite == 0 {
            return 50.0;
        }
        let sums = self.sums();
        if self.exposure_weighted {
            let total_exposure = sums[SUM_PAIR_EXPOSURE];
            let total_target = sums[SUM_PAIR_TARGET];
            let rate = if total_exposure > 0.0 {
                total_target / total_exposure
            } else {
                f64::NAN
            };
            if rate.is_finite() && rate > 0.0 {
                return (5.0 / rate).clamp(20.0, 1000.0);
            }
            return 100.0;
        }
        if self.min >= 0.0 && self.max <= 1.0 {
            let p = sums[SUM_FINITE_TARGET] / self.finite as f64;
            let minority = p.min(1.0 - p);
            if minority.is_finite() && minority > 0.0 {
                return (5.0 / minority).clamp(20.0, 500.0);
            }
            return 500.0;
        }
        50.0
    }
}

/// Each label's totals over a term's participating rows, added in row order: the
/// statistics prediction encodes with.
#[derive(Clone, Debug)]
pub struct LevelTotals {
    exposure_weighted: bool,
    pub sums: Vec<f64>,
    pub counts: Vec<u64>,
    pub exposures: Vec<f64>,
}

impl LevelTotals {
    pub fn new(n_levels: usize, exposure_weighted: bool) -> Self {
        LevelTotals {
            exposure_weighted,
            sums: vec![0.0; n_levels],
            counts: vec![0; n_levels],
            exposures: vec![0.0; n_levels],
        }
    }

    /// Add the next rows in row order; rows `include` marks false take no part.
    pub fn update(
        &mut self,
        codes: &[i32],
        target: &[f64],
        exposure: Option<&[f64]>,
        include: Option<&[bool]>,
    ) -> Result<()> {
        let n = codes.len();
        if target.len() != n
            || exposure.is_some_and(|e| e.len() != n)
            || include.is_some_and(|m| m.len() != n)
        {
            return Err(RustyStatsError::dim_mismatch(
                n,
                target.len(),
                "target-encoding codes vs target, exposure and mask",
            ));
        }
        if self.exposure_weighted && exposure.is_none() {
            return Err(RustyStatsError::InvalidValue(
                "an exposure-weighted term needs exposure".to_string(),
            ));
        }
        for i in 0..n {
            if include.is_some_and(|m| !m[i]) {
                continue;
            }
            let code = codes[i];
            if code < 0 || code as usize >= self.sums.len() {
                return Err(RustyStatsError::InvalidValue(format!(
                    "a training row has label code {code}, outside the term's {} labels",
                    self.sums.len()
                )));
            }
            let code = code as usize;
            self.sums[code] += target[i];
            self.counts[code] += 1;
            if let Some(e) = exposure {
                self.exposures[code] += e[i];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(n_levels: usize, exposure_weighted: bool) -> TermSettings {
        TermSettings {
            prior: 0.25,
            prior_weight: 3.0,
            exposure_weighted,
            n_levels,
        }
    }

    fn data(n: usize) -> (Vec<i32>, Vec<f64>, Vec<f64>, Vec<bool>) {
        let codes = (0..n).map(|i| ((i * 7 + i / 3) % 5) as i32).collect();
        let target = (0..n).map(|i| ((i * 13) % 11) as f64 * 0.37).collect();
        let exposure = (0..n).map(|i| 0.1 + ((i * 17) % 9) as f64 * 0.11).collect();
        let include = (0..n).map(|i| i % 4 != 1).collect();
        (codes, target, exposure, include)
    }

    /// The reference: sort (key, r) of the participating rows, walk, average.
    fn reference(
        settings: TermSettings,
        seed: u64,
        perms: usize,
        (codes, target, exposure, include): &(Vec<i32>, Vec<f64>, Vec<f64>, Vec<bool>),
    ) -> Vec<f64> {
        let rows: Vec<usize> = (0..codes.len()).filter(|&r| include[r]).collect();
        let mut columns = Vec::new();
        for m in 0..perms {
            let mut order: Vec<(u64, usize, usize)> = rows
                .iter()
                .enumerate()
                .map(|(ordinal, &r)| (record_key(seed, m as u64, r as u64), r, ordinal))
                .collect();
            order.sort();
            let mut walk = Walk::new(settings);
            let mut values = vec![0.0; rows.len()];
            for (key, r, ordinal) in order {
                values[ordinal] = walk.next(&Record {
                    key,
                    row: ordinal as u64,
                    code: codes[r] as u32,
                    target: target[r],
                    exposure: exposure[r],
                });
            }
            columns.push(values);
        }
        if perms == 1 {
            return columns.pop().expect("one permutation");
        }
        (0..rows.len())
            .map(|i| average(&columns.iter().map(|c| c[i]).collect::<Vec<_>>()))
            .collect()
    }

    fn encode(
        settings: TermSettings,
        seed: u64,
        perms: usize,
        spill: Option<(&Path, usize)>,
        input: &(Vec<i32>, Vec<f64>, Vec<f64>, Vec<bool>),
        slice: usize,
    ) -> Vec<f64> {
        let (codes, target, exposure, include) = input;
        let mut encoder = TermEncoder::new(settings, seed, perms, spill).expect("encoder");
        let mut stages = Vec::new();
        let mut progress = |stage: SpillStage, done: u64, total: u64, _start: bool| {
            stages.push((stage, done, total));
            Ok(())
        };
        for m in 0..perms {
            encoder.begin(m).expect("begin");
            let mut start = 0;
            while start < codes.len() {
                let end = (start + slice).min(codes.len());
                encoder
                    .push(
                        start as u64,
                        &codes[start..end],
                        &target[start..end],
                        Some(&exposure[start..end]),
                        Some(&include[start..end]),
                    )
                    .expect("push");
                start = end;
            }
            encoder.end(&mut progress).expect("end");
        }
        match encoder.finish(&mut progress).expect("finish") {
            TermValues::Memory(values) => values,
            TermValues::File(path) => {
                let bytes = std::fs::read(&path).expect("column");
                std::fs::remove_file(&path).expect("remove");
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b))
                    .collect()
            }
        }
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rustystats-te-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn keys_are_unique_per_permutation() {
        let mut keys: Vec<u64> = (0..10_000).map(|r| record_key(9, 2, r)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 10_000);
    }

    #[test]
    fn in_memory_values_equal_the_reference() {
        let input = data(3000);
        for exposure_weighted in [false, true] {
            for perms in [1, 4] {
                let s = settings(5, exposure_weighted);
                let got = encode(s, 17, perms, None, &input, 1000);
                let want = reference(s, 17, perms, &input);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn spilled_values_equal_in_memory_for_every_run_count() {
        let input = data(5000);
        let dir = scratch_dir("runs");
        for exposure_weighted in [false, true] {
            let s = settings(5, exposure_weighted);
            let memory = encode(s, 3, 3, None, &input, 777);
            // 1 run, 2 runs, 130 runs (an intermediate merge), and 3 records per run.
            for run_records in [10_000, 2_000, 29, 3] {
                let spilled = encode(s, 3, 3, Some((&dir, run_records)), &input, 777);
                assert_eq!(
                    spilled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    memory.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "run_records = {run_records}"
                );
            }
        }
        assert_eq!(std::fs::read_dir(&dir).expect("dir").count(), 0);
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    #[test]
    fn two_intermediate_merge_stages_give_the_same_values() {
        // 7,500 participating rows in runs of one record: 7,500 runs merge to 118, then
        // to 2, before the walk.
        let input = data(10_000);
        let dir = scratch_dir("stages");
        let s = settings(5, true);
        let memory = encode(s, 11, 1, None, &input, 4096);
        let spilled = encode(s, 11, 1, Some((&dir, 1)), &input, 4096);
        assert_eq!(
            spilled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            memory.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(std::fs::read_dir(&dir).expect("dir").count(), 0);
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    #[test]
    fn a_cancelled_stage_removes_every_file() {
        let input = data(4000);
        let dir = scratch_dir("cancel");
        let s = settings(5, false);
        let mut encoder = TermEncoder::new(s, 1, 2, Some((&dir, 10))).expect("encoder");
        encoder.begin(0).expect("begin");
        let (codes, target, _exposure, _include) = &input;
        encoder.push(0, codes, target, None, None).expect("push");
        let mut cancel = |stage: SpillStage, _done: u64, _total: u64, _start: bool| {
            if stage == SpillStage::Walk {
                Err(RustyStatsError::PassAborted("stop".to_string()))
            } else {
                Ok(())
            }
        };
        assert!(encoder.end(&mut cancel).is_err());
        assert!(std::fs::read_dir(&dir).expect("dir").count() > 0);
        drop(encoder);
        assert_eq!(std::fs::read_dir(&dir).expect("dir").count(), 0);
        std::fs::remove_dir_all(&dir).expect("clean");
    }

    #[test]
    fn prior_sums_do_not_depend_on_slicing() {
        let (_codes, target, exposure, include) = data(10_000);
        let whole = {
            let mut sums = PriorSums::new(true);
            sums.update(&target, Some(&exposure), Some(&include))
                .expect("update");
            (sums.prior(), sums.auto_prior_weight(), sums.rows())
        };
        let mut sums = PriorSums::new(true);
        for chunk in (0..target.len()).collect::<Vec<_>>().chunks(997) {
            let (a, b) = (chunk[0], chunk[chunk.len() - 1] + 1);
            sums.update(&target[a..b], Some(&exposure[a..b]), Some(&include[a..b]))
                .expect("update");
        }
        assert_eq!(
            (
                sums.prior().to_bits(),
                sums.auto_prior_weight().to_bits(),
                sums.rows()
            ),
            (whole.0.to_bits(), whole.1.to_bits(), whole.2)
        );
    }

    #[test]
    fn auto_prior_weight_follows_the_target_kind() {
        let mut binary = PriorSums::new(false);
        binary
            .update(&[0.0, 1.0, 0.0, 0.0, f64::NAN], None, None)
            .expect("update");
        // p = 1/4, minority 0.25: 5 / 0.25 = 20.
        assert_eq!(binary.auto_prior_weight(), 20.0);
        let mut counts = PriorSums::new(false);
        counts.update(&[0.0, 3.0, 1.0], None, None).expect("update");
        assert_eq!(counts.auto_prior_weight(), 50.0);
        let mut empty = PriorSums::new(false);
        empty.update(&[f64::NAN], None, None).expect("update");
        assert_eq!(empty.auto_prior_weight(), 50.0);
        let mut rate = PriorSums::new(true);
        rate.update(&[1.0, 0.0], Some(&[0.5, 0.5]), None)
            .expect("update");
        // rate 1: 5 / 1 = 5, clipped to 20.
        assert_eq!(rate.auto_prior_weight(), 20.0);
    }

    #[test]
    fn level_totals_add_in_row_order() {
        let mut totals = LevelTotals::new(2, false);
        totals
            .update(
                &[0, 1, 0],
                &[1.0, 2.0, 3.0],
                None,
                Some(&[true, true, false]),
            )
            .expect("update");
        assert_eq!(totals.sums, vec![1.0, 2.0]);
        assert_eq!(totals.counts, vec![1, 1]);
        assert!(totals.update(&[2], &[1.0], None, None).is_err());
    }
}
