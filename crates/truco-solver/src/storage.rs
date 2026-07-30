use crate::bincode_v1;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Legacy on-disk layout before the `solved` bitset was added.
#[derive(Deserialize)]
struct LegacyMatchValueOnly {
    values: [[f64; 13]; 13],
}

use crate::abstraction::{AbstractHand, TurnupClass};
use crate::game_tree::{PolicyLookup, PolicyValueSource};
use crate::info_set::InfoSetKey;
use crate::info_set::{AbstractAction, ActionHistory, InfoSet};
use crate::match_value::MatchValueTable;
use crate::strategy::{InfoSetData, StrategyTable};

/// `InfoSet` layout before the `is_dealer` position field was added (bincode is
/// positional, so old artifacts need an explicit mirror struct to decode).
///
/// A legacy entry was shared between the two dealer trees wherever the visible
/// `(player, tc, hand, history)` coincided — i.e. it holds the position-AVERAGED
/// accumulators. On load it is expanded into BOTH position variants so legacy
/// 11x11 checkpoints/strategies stay usable for resume and warm-start (each
/// position starts from the averaged values and CFR differentiates from there).
#[derive(Deserialize)]
struct LegacyInfoSet {
    player: truco_engine::Player,
    turnup_class: TurnupClass,
    starting_hand: AbstractHand,
    history: ActionHistory,
}

impl LegacyInfoSet {
    fn expand(&self) -> [InfoSet; 2] {
        [false, true].map(|is_dealer| InfoSet {
            player: self.player,
            is_dealer,
            turnup_class: self.turnup_class,
            starting_hand: self.starting_hand.clone(),
            history: self.history.clone(),
        })
    }
}

/// Metadata for a solved score state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolvedStateMeta {
    pub score: (u8, u8),
    pub turnup_class: TurnupClass,
    pub iterations: u64,
    pub num_info_sets: usize,
}

/// Serializable format for a strategy table.
#[derive(Serialize, Deserialize)]
struct SerializedStrategyTable {
    meta: SolvedStateMeta,
    entries: Vec<SerializedEntry>,
}

#[derive(Serialize, Deserialize)]
struct SerializedEntry {
    key: u64,
    info_set: InfoSet,
    actions: Vec<crate::info_set::AbstractAction>,
    average_strategy: Vec<f32>,
}

const COMPACT_POLICY_MAX_ACTIONS: usize = 8;

#[derive(Clone, Copy, Debug)]
struct CompactAveragePolicyEntry {
    actions: [u8; COMPACT_POLICY_MAX_ACTIONS],
    probabilities: [f32; COMPACT_POLICY_MAX_ACTIONS],
    len: u8,
}

/// Average-policy-only lookup used by policy-aware tree censuses. It omits the
/// full `InfoSet`, regrets, Vec headers, and duplicate metadata retained by a
/// resumable `StrategyTable`; one fixed-size record is stored per key.
#[derive(Debug, Default)]
pub struct CompactAveragePolicy {
    entries: HashMap<InfoSetKey, CompactAveragePolicyEntry>,
}

impl PolicyLookup for CompactAveragePolicy {
    fn action_probability(
        &self,
        key: InfoSetKey,
        action: AbstractAction,
        values: PolicyValueSource,
    ) -> Option<f64> {
        if values != PolicyValueSource::Average {
            return None;
        }
        let entry = self.entries.get(&key)?;
        let action_code = action.to_u8();
        (0..entry.len as usize)
            .find(|&i| entry.actions[i] == action_code)
            .map(|i| entry.probabilities[i] as f64)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Solution index file.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SolutionIndex {
    pub solved_states: Vec<SolvedStateMeta>,
}

/// Borrowed mirror of [`SerializedEntry`] for streaming writes; the per-row
/// average is tiny and owned, everything else borrows.
#[derive(Serialize)]
struct StrategyEntryRef<'a> {
    key: u64,
    info_set: &'a InfoSet,
    actions: &'a [AbstractAction],
    average_strategy: Vec<f32>,
}

/// Save a strategy table to a binary file.
pub fn save_strategy(
    path: &Path,
    table: &StrategyTable,
    meta: SolvedStateMeta,
) -> Result<(), StorageError> {
    let rows: Vec<_> = table
        .data
        .iter()
        .map(|(key, data)| {
            let info_set = table.get_info_set(*key).ok_or_else(|| {
                StorageError::Serialize(format!("missing info set for key {}", key.0))
            })?;
            Ok((
                key.0,
                info_set,
                data.actions.as_slice(),
                data.cumulative_strategy.as_slice(),
            ))
        })
        .collect::<Result<Vec<_>, StorageError>>()?;
    save_strategy_rows(path, meta, rows.into_iter())
}

/// Element type of a solve-time accumulator row (`f64` default, `f32` under
/// the `accum-f32` feature). The on-disk formats stay f64/f32 exactly as
/// before — rows are widened through `to_f64` at write time, which is a no-op
/// for `f64` and keeps artifacts from both builds byte-compatible.
pub trait AccumElem: Copy {
    fn to_f64(self) -> f64;
}

impl AccumElem for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl AccumElem for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Like [`save_strategy`], but from an iterator of
/// `(key, info_set, actions, cumulative_strategy)` borrows — lets the solver
/// write the average-strategy artifact directly from its dense accumulators
/// without rebuilding a `StrategyTable`. The per-row average is normalized
/// exactly like `InfoSetData::average_strategy` (uniform when the cumulative
/// sum is zero) and rows stream through a `BufWriter` in key order, matching
/// the historical byte layout.
pub fn save_strategy_rows<'a, F: AccumElem + 'a>(
    path: &Path,
    meta: SolvedStateMeta,
    entries: impl Iterator<Item = (u64, &'a InfoSet, &'a [AbstractAction], &'a [F])>,
) -> Result<(), StorageError> {
    let mut rows: Vec<(u64, &InfoSet, &[AbstractAction], &[F])> = entries.collect();
    rows.sort_by_key(|row| row.0);

    ensure_parent_dir(path)?;
    let tmp_path = atomic_tmp_path(path);
    {
        let file = fs::File::create(&tmp_path).map_err(|e| StorageError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);
        bincode_v1::serialize_into(&mut writer, &meta)
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        bincode_v1::serialize_into(&mut writer, &(rows.len() as u64))
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        for (key, info_set, actions, cumulative_strategy) in rows {
            let total: f64 = cumulative_strategy.iter().map(|s| s.to_f64()).sum();
            let average_strategy: Vec<f32> = if total > 0.0 {
                cumulative_strategy
                    .iter()
                    .map(|&s| (s.to_f64() / total) as f32)
                    .collect()
            } else {
                // Match `uniform_probs` exactly: f64 division, then f32 cast.
                vec![(1.0f64 / actions.len() as f64) as f32; actions.len()]
            };
            let entry = StrategyEntryRef {
                key,
                info_set,
                actions,
                average_strategy,
            };
            bincode_v1::serialize_into(&mut writer, &entry)
                .map_err(|e| StorageError::Serialize(e.to_string()))?;
        }
        use std::io::Write as _;
        writer
            .flush()
            .map_err(|e| StorageError::Io(e.to_string()))?;
    }
    fs::rename(&tmp_path, path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Incremental writer for the same on-disk layout [`save_strategy_rows`]
/// produces, for callers that know the row count up front but CANNOT hold the
/// rows: the deep path's composed profile is ~757 M rows at 0×0 (the
/// 2026-07-23 post-certificate OOM), and it is assembled subgame by subgame.
///
/// Rows are written in call order — NOT key-sorted like `save_strategy_rows` —
/// so the bytes differ from a materialized write even though every row and the
/// meta header are identical. Readers (`load_strategy`, `stream_strategy_rows`,
/// `load_compact_average_policy`) are order-independent; where two rows share a
/// key the LAST one wins, matching the deep composition's "subgame rows
/// override the inert trunk boundary roots".
pub struct StrategyRowWriter {
    writer: std::io::BufWriter<fs::File>,
    tmp_path: std::path::PathBuf,
    path: std::path::PathBuf,
    expected_rows: u64,
    written_rows: u64,
}

impl StrategyRowWriter {
    /// Open `path` for streaming; `expected_rows` is written into the header and
    /// enforced by [`StrategyRowWriter::finish`].
    pub fn create(
        path: &Path,
        meta: SolvedStateMeta,
        expected_rows: u64,
    ) -> Result<Self, StorageError> {
        ensure_parent_dir(path)?;
        let tmp_path = atomic_tmp_path(path);
        let file = fs::File::create(&tmp_path).map_err(|e| StorageError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);
        bincode_v1::serialize_into(&mut writer, &meta)
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        bincode_v1::serialize_into(&mut writer, &expected_rows)
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        Ok(Self {
            writer,
            tmp_path,
            path: path.to_path_buf(),
            expected_rows,
            written_rows: 0,
        })
    }

    /// Append one row. `weights` is normalized exactly like
    /// [`save_strategy_rows`] (uniform when the sum is zero), so an already
    /// normalized probability row round-trips through the same arithmetic a
    /// materialized write would have applied.
    pub fn write_row<F: AccumElem>(
        &mut self,
        key: u64,
        info_set: &InfoSet,
        actions: &[AbstractAction],
        weights: &[F],
    ) -> Result<(), StorageError> {
        let total: f64 = weights.iter().map(|s| s.to_f64()).sum();
        let average_strategy: Vec<f32> = if total > 0.0 {
            weights
                .iter()
                .map(|&s| (s.to_f64() / total) as f32)
                .collect()
        } else {
            vec![(1.0f64 / actions.len() as f64) as f32; actions.len()]
        };
        let entry = StrategyEntryRef {
            key,
            info_set,
            actions,
            average_strategy,
        };
        bincode_v1::serialize_into(&mut self.writer, &entry)
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        self.written_rows += 1;
        Ok(())
    }

    /// Flush and atomically rename into place. Errors (leaving the temp file
    /// behind, never the destination) if the row count missed the header.
    pub fn finish(mut self) -> Result<(), StorageError> {
        if self.written_rows != self.expected_rows {
            return Err(StorageError::Serialize(format!(
                "strategy row count mismatch: header {} rows, wrote {}",
                self.expected_rows, self.written_rows
            )));
        }
        use std::io::Write as _;
        self.writer
            .flush()
            .map_err(|e| StorageError::Io(e.to_string()))?;
        drop(self.writer);
        fs::rename(&self.tmp_path, &self.path).map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(())
    }
}

/// Legacy strategy-file layout: entries carry a [`LegacyInfoSet`].
#[derive(Deserialize)]
struct LegacySerializedStrategyTable {
    meta: SolvedStateMeta,
    entries: Vec<LegacySerializedEntry>,
}

#[derive(Deserialize)]
struct LegacySerializedEntry {
    #[allow(dead_code)]
    key: u64,
    info_set: LegacyInfoSet,
    actions: Vec<AbstractAction>,
    average_strategy: Vec<f32>,
}

/// Load a strategy table from a file.
///
/// The loaded table preserves the serialized average strategy and info-set
/// metadata, but does not reconstruct cumulative regrets. Files written before
/// `InfoSet` encoded position are decoded via the legacy layout and each entry
/// is expanded into both position variants (see [`LegacyInfoSet`]).
pub fn load_strategy(path: &Path) -> Result<(StrategyTable, SolvedStateMeta), StorageError> {
    let bytes = fs::read(path).map_err(|e| StorageError::Io(e.to_string()))?;

    let mut table = StrategyTable::new();
    match bincode_v1::deserialize::<SerializedStrategyTable>(&bytes) {
        Ok(serialized) => {
            for entry in serialized.entries {
                let avg: Vec<f64> = entry.average_strategy.iter().map(|&x| x as f64).collect();
                let data = InfoSetData {
                    cumulative_regret: vec![0.0; entry.actions.len()],
                    cumulative_strategy: avg,
                    pending_regret: Vec::new(),
                    last_regret: Vec::new(),
                    actions: entry.actions,
                };
                // Rekey from the info set with the current (deterministic) hasher
                // rather than trusting the stored key, so files written by another
                // process are usable for resume / warm-start. (The stored entry.key
                // may be from an older per-process random-seed hasher.)
                let key = entry.info_set.key();
                table.insert_serialized(key, entry.info_set, data);
            }
            Ok((table, serialized.meta))
        }
        Err(_) => {
            let legacy: LegacySerializedStrategyTable = bincode_v1::deserialize(&bytes)
                .map_err(|e| StorageError::Deserialize(e.to_string()))?;
            for entry in legacy.entries {
                let avg: Vec<f64> = entry.average_strategy.iter().map(|&x| x as f64).collect();
                for info_set in entry.info_set.expand() {
                    let data = InfoSetData {
                        cumulative_regret: vec![0.0; entry.actions.len()],
                        cumulative_strategy: avg.clone(),
                        pending_regret: Vec::new(),
                        last_regret: Vec::new(),
                        actions: entry.actions.clone(),
                    };
                    let key = info_set.key();
                    table.insert_serialized(key, info_set, data);
                }
            }
            Ok((table, legacy.meta))
        }
    }
}

/// One streamed strategy row: full info-set metadata plus the saved average.
pub struct StrategyRow {
    pub info_set: InfoSet,
    pub actions: Vec<crate::info_set::AbstractAction>,
    pub average_strategy: Vec<f32>,
}

/// Stream a current-format strategy file row by row without materializing the
/// table or a second info-set map. Rows arrive in serialized (key-sorted)
/// order. Like [`load_compact_average_policy`] this intentionally accepts only
/// the current positioned format; use `load_strategy` for legacy artifacts.
pub fn stream_strategy_rows(
    path: &Path,
    mut row_fn: impl FnMut(StrategyRow),
) -> Result<SolvedStateMeta, StorageError> {
    let file = fs::File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let meta: SolvedStateMeta = bincode_v1::deserialize_from(&mut reader)
        .map_err(|e| StorageError::Deserialize(e.to_string()))?;
    let entry_count: u64 = bincode_v1::deserialize_from(&mut reader)
        .map_err(|e| StorageError::Deserialize(e.to_string()))?;
    for _ in 0..entry_count {
        let entry: SerializedEntry = bincode_v1::deserialize_from(&mut reader)
            .map_err(|e| StorageError::Deserialize(e.to_string()))?;
        if entry.actions.len() != entry.average_strategy.len() {
            return Err(StorageError::Deserialize(
                "strategy action/probability length mismatch".into(),
            ));
        }
        row_fn(StrategyRow {
            info_set: entry.info_set,
            actions: entry.actions,
            average_strategy: entry.average_strategy,
        });
    }
    Ok(meta)
}

/// Stream a current positioned full-state checkpoint as policy rows, with each
/// row's cumulative strategy normalized to the average strategy (uniform when
/// the cumulative sum is zero, matching `load_compact_average_checkpoint`).
/// Regrets are discarded row by row; nothing is materialized.
pub fn stream_checkpoint_policy_rows(
    path: &Path,
    mut row_fn: impl FnMut(StrategyRow),
) -> Result<CheckpointMeta, StorageError> {
    let mut stream = CheckpointStream::open(path)?;
    let meta = stream.meta().clone();
    while let Some(entry) = stream.next_entry()? {
        let total: f64 = entry.cumulative_strategy.iter().sum();
        let uniform = 1.0 / entry.actions.len().max(1) as f64;
        let average_strategy: Vec<f32> = entry
            .cumulative_strategy
            .iter()
            .map(|&weight| {
                if total > 0.0 {
                    (weight / total) as f32
                } else {
                    uniform as f32
                }
            })
            .collect();
        row_fn(StrategyRow {
            info_set: entry.info_set,
            actions: entry.actions,
            average_strategy,
        });
    }
    Ok(meta)
}

/// Stream a current-format strategy file into a compact average-policy map.
/// Unlike [`load_strategy`], this never materializes the serialized entry Vec
/// or a second `InfoSet` map, which keeps a 10x10 policy census practical on a
/// workstation. Every key is still recomputed from the serialized `InfoSet`,
/// preserving the cross-process/legacy-hasher safety of the full loader.
///
/// This loader intentionally accepts only the current positioned format. Use
/// `load_strategy` for legacy artifacts that require position expansion.
pub fn load_compact_average_policy(
    path: &Path,
) -> Result<(CompactAveragePolicy, SolvedStateMeta), StorageError> {
    let file = fs::File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let meta: SolvedStateMeta = bincode_v1::deserialize_from(&mut reader)
        .map_err(|e| StorageError::Deserialize(e.to_string()))?;
    let entry_count: u64 = bincode_v1::deserialize_from(&mut reader)
        .map_err(|e| StorageError::Deserialize(e.to_string()))?;
    let entry_count = usize::try_from(entry_count)
        .map_err(|_| StorageError::Deserialize("strategy entry count exceeds usize".into()))?;
    let mut entries = HashMap::with_capacity(entry_count);

    for _ in 0..entry_count {
        let entry: SerializedEntry = bincode_v1::deserialize_from(&mut reader)
            .map_err(|e| StorageError::Deserialize(e.to_string()))?;
        if entry.actions.len() != entry.average_strategy.len() {
            return Err(StorageError::Deserialize(
                "strategy action/probability length mismatch".into(),
            ));
        }
        if entry.actions.len() > COMPACT_POLICY_MAX_ACTIONS {
            return Err(StorageError::Deserialize(format!(
                "strategy entry has {} actions; compact maximum is {}",
                entry.actions.len(),
                COMPACT_POLICY_MAX_ACTIONS
            )));
        }
        let mut compact = CompactAveragePolicyEntry {
            actions: [0; COMPACT_POLICY_MAX_ACTIONS],
            probabilities: [0.0; COMPACT_POLICY_MAX_ACTIONS],
            len: entry.actions.len() as u8,
        };
        for (i, (&action, &probability)) in entry
            .actions
            .iter()
            .zip(entry.average_strategy.iter())
            .enumerate()
        {
            compact.actions[i] = action.to_u8();
            compact.probabilities[i] = probability;
        }
        entries.insert(entry.info_set.key(), compact);
    }

    Ok((CompactAveragePolicy { entries }, meta))
}

/// Stream a current positioned full-state checkpoint into the same compact
/// average-policy representation used by policy censuses and compact BR tools.
/// Regrets and full `InfoSet` metadata are discarded row by row.
pub fn load_compact_average_checkpoint(
    path: &Path,
) -> Result<(CompactAveragePolicy, CheckpointMeta), StorageError> {
    load_compact_average_checkpoint_with_player_swap(path, false)
}

/// As [`load_compact_average_checkpoint`], optionally adding a player-swapped
/// alias for every row. A single-dealer solve contains both strategic
/// positions but only one assignment of player labels; the aliases let a
/// score-DAG sampling scout reuse that position policy after dealer
/// alternation. This is an explicit symmetry projection, not checkpoint
/// resume semantics.
pub fn load_compact_average_checkpoint_with_player_swap(
    path: &Path,
    add_player_swap: bool,
) -> Result<(CompactAveragePolicy, CheckpointMeta), StorageError> {
    let mut stream = CheckpointStream::open(path)?;
    let meta = stream.meta().clone();
    let mut entries = HashMap::with_capacity(meta.num_info_sets);
    while let Some(entry) = stream.next_entry()? {
        if entry.actions.len() > COMPACT_POLICY_MAX_ACTIONS {
            return Err(StorageError::Deserialize(format!(
                "checkpoint entry has {} actions; compact maximum is {}",
                entry.actions.len(),
                COMPACT_POLICY_MAX_ACTIONS
            )));
        }
        let total: f64 = entry.cumulative_strategy.iter().sum();
        let uniform = 1.0 / entry.actions.len().max(1) as f64;
        let mut compact = CompactAveragePolicyEntry {
            actions: [0; COMPACT_POLICY_MAX_ACTIONS],
            probabilities: [0.0; COMPACT_POLICY_MAX_ACTIONS],
            len: entry.actions.len() as u8,
        };
        for (i, (&action, &weight)) in entry
            .actions
            .iter()
            .zip(entry.cumulative_strategy.iter())
            .enumerate()
        {
            compact.actions[i] = action.to_u8();
            compact.probabilities[i] = if total > 0.0 {
                (weight / total) as f32
            } else {
                uniform as f32
            };
        }
        entries.insert(entry.info_set.key(), compact);
        if add_player_swap {
            let mut swapped = entry.info_set;
            swapped.player = 1 - swapped.player;
            entries.entry(swapped.key()).or_insert(compact);
        }
    }
    Ok((CompactAveragePolicy { entries }, meta))
}

/// Metadata for a full-state solver checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub score: (u8, u8),
    pub turnup_class: TurnupClass,
    pub algo: String,
    pub iteration: u64,
    pub num_info_sets: usize,
    /// `Some(d)` when the solve was restricted to dealer `d`'s game
    /// (`--dealer`); `None` for a joint both-dealers solve.
    pub dealer_filter: Option<u8>,
}

/// Serializable format for a full-state checkpoint.
///
/// Unlike `SerializedStrategyTable` (which only stores the average strategy),
/// this preserves the complete CFR accumulators so a run can resume and keep
/// improving from where it left off.
#[derive(Serialize, Deserialize)]
struct SerializedCheckpoint {
    meta: CheckpointMeta,
    entries: Vec<SerializedCheckpointEntry>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SerializedCheckpointEntry {
    key: u64,
    pub(crate) info_set: InfoSet,
    pub(crate) actions: Vec<AbstractAction>,
    pub(crate) cumulative_regret: Vec<f64>,
    pub(crate) cumulative_strategy: Vec<f64>,
}

/// Row-at-a-time reader for the current positioned checkpoint format.
///
/// Bincode serializes `SerializedCheckpoint { meta, entries: Vec<_> }` as the
/// metadata, a `u64` vector length, then each entry consecutively. Reading that
/// layout directly avoids the serialized byte buffer, entry Vec, and complete
/// source `StrategyTable` that a warm start does not need. Legacy checkpoints
/// still use [`load_checkpoint`] because they require position expansion.
pub(crate) struct CheckpointStream {
    reader: BufReader<fs::File>,
    meta: CheckpointMeta,
    remaining: usize,
}

impl CheckpointStream {
    pub(crate) fn open(path: &Path) -> Result<Self, StorageError> {
        let file = fs::File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
        let mut reader = BufReader::with_capacity(1024 * 1024, file);
        let meta: CheckpointMeta = bincode_v1::deserialize_from(&mut reader)
            .map_err(|e| StorageError::Deserialize(e.to_string()))?;
        let entry_count: u64 = bincode_v1::deserialize_from(&mut reader)
            .map_err(|e| StorageError::Deserialize(e.to_string()))?;
        let remaining = usize::try_from(entry_count).map_err(|_| {
            StorageError::Deserialize("checkpoint entry count exceeds usize".into())
        })?;
        if remaining != meta.num_info_sets {
            return Err(StorageError::Deserialize(format!(
                "checkpoint metadata says {} rows but stream contains {}",
                meta.num_info_sets, remaining
            )));
        }
        Ok(Self {
            reader,
            meta,
            remaining,
        })
    }

    pub(crate) fn meta(&self) -> &CheckpointMeta {
        &self.meta
    }

    pub(crate) fn next_entry(&mut self) -> Result<Option<SerializedCheckpointEntry>, StorageError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let entry: SerializedCheckpointEntry = bincode_v1::deserialize_from(&mut self.reader)
            .map_err(|e| StorageError::Deserialize(e.to_string()))?;
        if entry.actions.len() != entry.cumulative_regret.len()
            || entry.actions.len() != entry.cumulative_strategy.len()
        {
            return Err(StorageError::Deserialize(
                "checkpoint action/accumulator length mismatch".into(),
            ));
        }
        self.remaining -= 1;
        Ok(Some(entry))
    }
}

/// Save a full-state solver checkpoint to a binary file.
///
/// Stores per-info-set cumulative regrets and cumulative strategy sums so a
/// later `load_checkpoint` can resume the solve exactly. The write is atomic:
/// bytes go to `<path>.tmp` and are then renamed onto `path`, so a process kill
/// mid-write cannot corrupt an existing checkpoint.
pub fn save_checkpoint(
    path: &Path,
    table: &StrategyTable,
    meta: CheckpointMeta,
) -> Result<(), StorageError> {
    let mut pairs = Vec::with_capacity(table.len());
    for (key, data) in table.data.iter() {
        let info_set = table.get_info_set(*key).ok_or_else(|| {
            StorageError::Serialize(format!("missing info set for key {}", key.0))
        })?;
        pairs.push((
            key.0,
            info_set,
            data.actions.as_slice(),
            data.cumulative_regret.as_slice(),
            data.cumulative_strategy.as_slice(),
        ));
    }
    save_checkpoint_iter(path, meta, pairs.into_iter())
}

/// Borrowed mirror of [`SerializedCheckpointEntry`]: bincode encodes `&T` /
/// `&[T]` exactly like `T` / `Vec<T>`, so streaming these one at a time after
/// the meta and a `u64` count produces byte-identical files to serializing the
/// whole owned `SerializedCheckpoint` — without an owned copy of every row or
/// a whole-file byte buffer (both multi-GiB at production scale).
#[derive(Serialize)]
struct CheckpointEntryRef<'a> {
    key: u64,
    info_set: &'a InfoSet,
    actions: &'a [AbstractAction],
    cumulative_regret: &'a [f64],
    cumulative_strategy: &'a [f64],
}

fn atomic_tmp_path(path: &Path) -> std::path::PathBuf {
    path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| format!("{}.", e))
            .unwrap_or_default()
    ))
}

fn ensure_parent_dir(path: &Path) -> Result<(), StorageError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| StorageError::Io(e.to_string()))?;
        }
    }
    Ok(())
}

/// Like [`save_checkpoint`], but from an iterator of
/// `(key, info_set, actions, regret, strategy)` slices — lets the solver write
/// checkpoints directly from its SoA dense accumulators without rebuilding a
/// `StrategyTable` first. Rows are buffered only as borrows (sorted by key to
/// keep the on-disk layout identical to the historical whole-struct
/// serialization) and stream through a `BufWriter`.
pub fn save_checkpoint_iter<'a, F: AccumElem + 'a>(
    path: &Path,
    meta: CheckpointMeta,
    entries: impl Iterator<Item = (u64, &'a InfoSet, &'a [AbstractAction], &'a [F], &'a [F])>,
) -> Result<(), StorageError> {
    let mut rows: Vec<(u64, &InfoSet, &[AbstractAction], &[F], &[F])> = entries.collect();
    rows.sort_by_key(|row| row.0);

    ensure_parent_dir(path)?;
    let tmp_path = atomic_tmp_path(path);
    {
        let file = fs::File::create(&tmp_path).map_err(|e| StorageError::Io(e.to_string()))?;
        let mut writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);
        bincode_v1::serialize_into(&mut writer, &meta)
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        bincode_v1::serialize_into(&mut writer, &(rows.len() as u64))
            .map_err(|e| StorageError::Serialize(e.to_string()))?;
        // The checkpoint format stays f64 regardless of the in-memory
        // accumulator type: rows widen through reusable scratch buffers, so
        // the on-disk bytes match the historical layout exactly.
        let mut regret_buf: Vec<f64> = Vec::new();
        let mut strategy_buf: Vec<f64> = Vec::new();
        for (key, info_set, actions, cumulative_regret, cumulative_strategy) in rows {
            regret_buf.clear();
            regret_buf.extend(cumulative_regret.iter().map(|x| x.to_f64()));
            strategy_buf.clear();
            strategy_buf.extend(cumulative_strategy.iter().map(|x| x.to_f64()));
            let entry = CheckpointEntryRef {
                key,
                info_set,
                actions,
                cumulative_regret: &regret_buf,
                cumulative_strategy: &strategy_buf,
            };
            bincode_v1::serialize_into(&mut writer, &entry)
                .map_err(|e| StorageError::Serialize(e.to_string()))?;
        }
        use std::io::Write as _;
        writer
            .flush()
            .map_err(|e| StorageError::Io(e.to_string()))?;
    }
    fs::rename(&tmp_path, path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Legacy checkpoint layout: pre-`dealer_filter` meta and [`LegacyInfoSet`]
/// entries.
#[derive(Deserialize)]
struct LegacySerializedCheckpoint {
    meta: LegacyCheckpointMeta,
    entries: Vec<LegacySerializedCheckpointEntry>,
}

#[derive(Deserialize)]
struct LegacyCheckpointMeta {
    score: (u8, u8),
    turnup_class: TurnupClass,
    algo: String,
    iteration: u64,
    num_info_sets: usize,
}

#[derive(Deserialize)]
struct LegacySerializedCheckpointEntry {
    #[allow(dead_code)]
    key: u64,
    info_set: LegacyInfoSet,
    actions: Vec<AbstractAction>,
    cumulative_regret: Vec<f64>,
    cumulative_strategy: Vec<f64>,
}

/// Load a full-state solver checkpoint, rebuilding a `StrategyTable` with the
/// complete CFR accumulators (regrets + strategy sums + actions) intact.
/// Checkpoints written before `InfoSet` encoded position are decoded via the
/// legacy layout and each entry is expanded into both position variants (see
/// [`LegacyInfoSet`]).
pub fn load_checkpoint(path: &Path) -> Result<(StrategyTable, CheckpointMeta), StorageError> {
    let bytes = fs::read(path).map_err(|e| StorageError::Io(e.to_string()))?;

    let mut table = StrategyTable::new();
    match bincode_v1::deserialize::<SerializedCheckpoint>(&bytes) {
        Ok(serialized) => {
            for entry in serialized.entries {
                let data = InfoSetData {
                    cumulative_regret: entry.cumulative_regret,
                    cumulative_strategy: entry.cumulative_strategy,
                    pending_regret: Vec::new(),
                    last_regret: Vec::new(),
                    actions: entry.actions,
                };
                // Rekey deterministically on load (see load_strategy) so checkpoints
                // from another process resume / warm-start correctly.
                let key = entry.info_set.key();
                table.insert_serialized(key, entry.info_set, data);
            }
            Ok((table, serialized.meta))
        }
        Err(_) => {
            let legacy: LegacySerializedCheckpoint = bincode_v1::deserialize(&bytes)
                .map_err(|e| StorageError::Deserialize(e.to_string()))?;
            for entry in legacy.entries {
                for info_set in entry.info_set.expand() {
                    let data = InfoSetData {
                        cumulative_regret: entry.cumulative_regret.clone(),
                        cumulative_strategy: entry.cumulative_strategy.clone(),
                        pending_regret: Vec::new(),
                        last_regret: Vec::new(),
                        actions: entry.actions.clone(),
                    };
                    let key = info_set.key();
                    table.insert_serialized(key, info_set, data);
                }
            }
            let meta = CheckpointMeta {
                score: legacy.meta.score,
                turnup_class: legacy.meta.turnup_class,
                algo: legacy.meta.algo,
                iteration: legacy.meta.iteration,
                num_info_sets: legacy.meta.num_info_sets,
                dealer_filter: None,
            };
            Ok((table, meta))
        }
    }
}

/// Save match values to a file.
pub fn save_match_values(path: &Path, values: &MatchValueTable) -> Result<(), StorageError> {
    let bytes =
        bincode_v1::serialize(values).map_err(|e| StorageError::Serialize(e.to_string()))?;
    fs::write(path, bytes).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Load match values from a file.
///
/// Supports the current format (`values` + `solved` bitset) and legacy files
/// that only stored `values`.
pub fn load_match_values(path: &Path) -> Result<MatchValueTable, StorageError> {
    let bytes = fs::read(path).map_err(|e| StorageError::Io(e.to_string()))?;
    match bincode_v1::deserialize::<MatchValueTable>(&bytes) {
        Ok(t) => Ok(t),
        Err(_) => {
            let legacy: LegacyMatchValueOnly = bincode_v1::deserialize(&bytes)
                .map_err(|e| StorageError::Deserialize(e.to_string()))?;
            Ok(MatchValueTable::from_values_legacy(legacy.values))
        }
    }
}

/// Save solution index.
pub fn save_index(path: &Path, index: &SolutionIndex) -> Result<(), StorageError> {
    let json =
        serde_json::to_string_pretty(index).map_err(|e| StorageError::Serialize(e.to_string()))?;
    fs::write(path, json).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Load solution index.
pub fn load_index(path: &Path) -> Result<SolutionIndex, StorageError> {
    let json = fs::read_to_string(path).map_err(|e| StorageError::Io(e.to_string()))?;
    serde_json::from_str(&json).map_err(|e| StorageError::Deserialize(e.to_string()))
}

#[derive(Debug)]
pub enum StorageError {
    Io(String),
    Serialize(String),
    Deserialize(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(e) => write!(f, "IO error: {}", e),
            StorageError::Serialize(e) => write!(f, "serialization error: {}", e),
            StorageError::Deserialize(e) => write!(f, "deserialization error: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    use crate::abstraction::{AbstractCard, TurnupClass};
    use crate::info_set::{AbstractAction, InfoSet};

    #[test]
    fn test_match_values_roundtrip() {
        let dir = std::env::temp_dir().join("truco_solver_test_mv");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("match_values.bin");

        let mut mv = MatchValueTable::new();
        mv.set(11, 11, 0, 0.48);
        mv.set(11, 11, 1, 0.52);
        mv.set(10, 11, 0, 0.35);

        save_match_values(&path, &mv).unwrap();
        let loaded = load_match_values(&path).unwrap();

        // Per-dealer cells round-trip independently.
        assert!((loaded.get(11, 11, 0) - 0.48).abs() < 1e-10);
        assert!((loaded.get(11, 11, 1) - 0.52).abs() < 1e-10);
        assert!((loaded.get(10, 11, 0) - 0.35).abs() < 1e-10);
        // Terminal cells are dealer-independent.
        assert_eq!(loaded.get(12, 0, 0), 1.0);
        assert_eq!(loaded.get(12, 0, 1), 1.0);

        let _ = fs::remove_dir_all(&dir);
    }

    /// The streaming writers must produce the exact bytes of the historical
    /// whole-struct `bincode_v1::serialize` layout: same meta, u64 row count,
    /// key-sorted entries, f32 averages with the uniform fallback.
    #[test]
    fn streaming_writers_match_whole_struct_serialization() {
        let dir = std::env::temp_dir().join("truco_solver_test_stream_writers");
        let _ = fs::create_dir_all(&dir);

        let tc = TurnupClass {
            blocked_plain_level: 2,
        };
        let mut table = StrategyTable::new();
        let mut infos = Vec::new();
        for (player, cards, cumulative) in [
            (0u8, [1u8, 4, 6], vec![3.0, 1.0]),
            (1u8, [0u8, 2, 5], vec![0.0, 0.0]), // zero mass: uniform fallback
            (0u8, [2u8, 3, 6], vec![1.0, 7.0]),
        ] {
            let info_set = InfoSet::new(
                player,
                player == 1,
                tc,
                smallvec![
                    AbstractCard::Plain(cards[0]),
                    AbstractCard::Plain(cards[1]),
                    AbstractCard::Plain(cards[2])
                ],
            );
            let data = table.get_or_insert(
                &info_set,
                &[AbstractAction::AcceptEleven, AbstractAction::FoldEleven],
            );
            data.cumulative_strategy = cumulative.clone();
            data.cumulative_regret = vec![2.5, -0.5];
            infos.push(info_set);
        }
        let meta = SolvedStateMeta {
            score: (11, 11),
            turnup_class: tc,
            iterations: 40,
            num_info_sets: infos.len(),
        };

        // Strategy artifact: streaming writer vs historical owned layout.
        let strategy_path = dir.join("stream.bin");
        save_strategy(&strategy_path, &table, meta.clone()).unwrap();
        let streamed = fs::read(&strategy_path).unwrap();
        let mut entries: Vec<SerializedEntry> = table
            .data
            .iter()
            .map(|(key, data)| SerializedEntry {
                key: key.0,
                info_set: table.get_info_set(*key).cloned().unwrap(),
                actions: data.actions.clone(),
                average_strategy: data.average_strategy().iter().map(|&x| x as f32).collect(),
            })
            .collect();
        entries.sort_by_key(|entry| entry.key);
        let expected = bincode_v1::serialize(&SerializedStrategyTable {
            meta: meta.clone(),
            entries,
        })
        .unwrap();
        assert_eq!(streamed, expected, "strategy artifact bytes changed");

        // Checkpoint artifact: streaming writer vs historical owned layout.
        let ckpt_meta = CheckpointMeta {
            score: (11, 11),
            turnup_class: tc,
            algo: "SyncCfrPlus".to_string(),
            iteration: 40,
            num_info_sets: infos.len(),
            dealer_filter: Some(0),
        };
        let ckpt_path = dir.join("stream.ckpt.bin");
        save_checkpoint(&ckpt_path, &table, ckpt_meta.clone()).unwrap();
        let streamed = fs::read(&ckpt_path).unwrap();
        let mut entries: Vec<SerializedCheckpointEntry> = table
            .data
            .iter()
            .map(|(key, data)| SerializedCheckpointEntry {
                key: key.0,
                info_set: table.get_info_set(*key).cloned().unwrap(),
                actions: data.actions.clone(),
                cumulative_regret: data.cumulative_regret.clone(),
                cumulative_strategy: data.cumulative_strategy.clone(),
            })
            .collect();
        entries.sort_by_key(|entry| entry.key);
        let expected = bincode_v1::serialize(&SerializedCheckpoint {
            meta: ckpt_meta,
            entries,
        })
        .unwrap();
        assert_eq!(streamed, expected, "checkpoint artifact bytes changed");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_strategy_roundtrip_preserves_info_sets() {
        let dir = std::env::temp_dir().join("truco_solver_test_strategy");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("strategy.bin");

        let mut table = StrategyTable::new();
        let info_set = InfoSet::new(
            1,
            false,
            TurnupClass {
                blocked_plain_level: 2,
            },
            smallvec![
                AbstractCard::Plain(1),
                AbstractCard::Plain(5),
                AbstractCard::Manilha(3)
            ],
        );
        let data = table.get_or_insert(
            &info_set,
            &[AbstractAction::AcceptEleven, AbstractAction::FoldEleven],
        );
        data.cumulative_strategy = vec![3.0, 1.0];

        let meta = SolvedStateMeta {
            score: (11, 11),
            turnup_class: TurnupClass {
                blocked_plain_level: 2,
            },
            iterations: 120,
            num_info_sets: 1,
        };

        save_strategy(&path, &table, meta.clone()).unwrap();
        let (loaded, loaded_meta) = load_strategy(&path).unwrap();

        assert_eq!(loaded_meta.score, meta.score);
        assert_eq!(loaded_meta.turnup_class, meta.turnup_class);
        assert_eq!(loaded_meta.iterations, meta.iterations);

        let loaded_data = loaded.get(&info_set).unwrap();
        let loaded_info_set = loaded.get_info_set(info_set.key()).unwrap();
        assert_eq!(loaded_info_set, &info_set);
        assert_eq!(loaded_data.actions.len(), 2);

        let avg = loaded_data.average_strategy();
        assert!((avg[0] - 0.75).abs() < 1e-6);
        assert!((avg[1] - 0.25).abs() < 1e-6);

        let (compact, compact_meta) = load_compact_average_policy(&path).unwrap();
        assert_eq!(compact_meta.score, meta.score);
        assert_eq!(compact.len(), 1);
        assert!(
            (compact
                .action_probability(
                    info_set.key(),
                    AbstractAction::AcceptEleven,
                    PolicyValueSource::Average,
                )
                .unwrap()
                - 0.75)
                .abs()
                < 1e-6
        );
        assert!(
            (compact
                .action_probability(
                    info_set.key(),
                    AbstractAction::FoldEleven,
                    PolicyValueSource::Average,
                )
                .unwrap()
                - 0.25)
                .abs()
                < 1e-6
        );
        assert!(compact
            .action_probability(
                info_set.key(),
                AbstractAction::AcceptEleven,
                PolicyValueSource::Current,
            )
            .is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_roundtrip_preserves_full_state() {
        let dir = std::env::temp_dir().join("truco_solver_test_checkpoint");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("tc0.ckpt.bin");

        let mut table = StrategyTable::new();
        let info_set = InfoSet::new(
            0,
            true,
            TurnupClass {
                blocked_plain_level: 1,
            },
            smallvec![
                AbstractCard::Plain(2),
                AbstractCard::Plain(6),
                AbstractCard::Manilha(0)
            ],
        );
        let data = table.get_or_insert(
            &info_set,
            &[AbstractAction::AcceptRaise, AbstractAction::Fold],
        );
        data.cumulative_regret = vec![4.5, -2.0];
        data.cumulative_strategy = vec![7.0, 3.0];
        let expected_actions = data.actions.clone();

        let meta = CheckpointMeta {
            score: (11, 11),
            turnup_class: TurnupClass {
                blocked_plain_level: 1,
            },
            algo: "CFR+".to_string(),
            iteration: 4242,
            num_info_sets: 1,
            dealer_filter: Some(1),
        };

        save_checkpoint(&path, &table, meta.clone()).unwrap();

        let (compact, compact_meta) = load_compact_average_checkpoint(&path).unwrap();
        assert_eq!(compact_meta.iteration, meta.iteration);
        assert_eq!(compact.len(), 1);
        assert!(
            (compact
                .action_probability(
                    info_set.key(),
                    AbstractAction::AcceptRaise,
                    PolicyValueSource::Average,
                )
                .unwrap()
                - 0.7)
                .abs()
                < 1e-6
        );
        let (symmetrized, _) =
            load_compact_average_checkpoint_with_player_swap(&path, true).unwrap();
        let mut swapped_info_set = info_set.clone();
        swapped_info_set.player = 1;
        assert_eq!(symmetrized.len(), 2);
        assert!(
            (symmetrized
                .action_probability(
                    swapped_info_set.key(),
                    AbstractAction::AcceptRaise,
                    PolicyValueSource::Average,
                )
                .unwrap()
                - 0.7)
                .abs()
                < 1e-6
        );

        let mut stream = CheckpointStream::open(&path).unwrap();
        assert_eq!(stream.meta().score, meta.score);
        assert_eq!(stream.meta().iteration, meta.iteration);
        let streamed = stream.next_entry().unwrap().unwrap();
        assert_eq!(streamed.info_set, info_set);
        assert_eq!(streamed.actions, expected_actions);
        assert_eq!(streamed.cumulative_regret, vec![4.5, -2.0]);
        assert_eq!(streamed.cumulative_strategy, vec![7.0, 3.0]);
        assert!(stream.next_entry().unwrap().is_none());

        let (loaded, loaded_meta) = load_checkpoint(&path).unwrap();

        assert_eq!(loaded_meta.score, meta.score);
        assert_eq!(loaded_meta.turnup_class, meta.turnup_class);
        assert_eq!(loaded_meta.algo, meta.algo);
        assert_eq!(loaded_meta.iteration, meta.iteration);
        assert_eq!(loaded_meta.num_info_sets, meta.num_info_sets);
        assert_eq!(loaded_meta.dealer_filter, meta.dealer_filter);

        let loaded_data = loaded.get(&info_set).unwrap();
        let loaded_info_set = loaded.get_info_set(info_set.key()).unwrap();
        assert_eq!(loaded_info_set, &info_set);
        assert_eq!(loaded_data.actions, expected_actions);
        assert_eq!(loaded_data.cumulative_regret, vec![4.5, -2.0]);
        assert_eq!(loaded_data.cumulative_strategy, vec![7.0, 3.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Serialize-side mirrors of the pre-position on-disk layouts, used to
    /// fabricate legacy artifact bytes for the fallback-loader tests.
    mod legacy_writers {
        use super::*;
        use serde::Serialize;

        #[derive(Serialize)]
        pub struct WInfoSet {
            pub player: truco_engine::Player,
            pub turnup_class: TurnupClass,
            pub starting_hand: crate::abstraction::AbstractHand,
            pub history: ActionHistory,
        }

        #[derive(Serialize)]
        pub struct WStrategyEntry {
            pub key: u64,
            pub info_set: WInfoSet,
            pub actions: Vec<AbstractAction>,
            pub average_strategy: Vec<f32>,
        }

        #[derive(Serialize)]
        pub struct WStrategyTable {
            pub meta: SolvedStateMeta,
            pub entries: Vec<WStrategyEntry>,
        }

        #[derive(Serialize)]
        pub struct WCheckpointMeta {
            pub score: (u8, u8),
            pub turnup_class: TurnupClass,
            pub algo: String,
            pub iteration: u64,
            pub num_info_sets: usize,
        }

        #[derive(Serialize)]
        pub struct WCheckpointEntry {
            pub key: u64,
            pub info_set: WInfoSet,
            pub actions: Vec<AbstractAction>,
            pub cumulative_regret: Vec<f64>,
            pub cumulative_strategy: Vec<f64>,
        }

        #[derive(Serialize)]
        pub struct WCheckpoint {
            pub meta: WCheckpointMeta,
            pub entries: Vec<WCheckpointEntry>,
        }
    }

    fn legacy_hand() -> crate::abstraction::AbstractHand {
        smallvec![
            AbstractCard::Plain(1),
            AbstractCard::Plain(5),
            AbstractCard::Manilha(3)
        ]
    }

    fn expected_position_variants(tc: TurnupClass) -> [InfoSet; 2] {
        [false, true].map(|is_dealer| InfoSet::new(1, is_dealer, tc, legacy_hand()))
    }

    #[test]
    fn test_legacy_strategy_loads_with_position_expansion() {
        let dir = std::env::temp_dir().join("truco_solver_test_legacy_strategy");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("legacy.bin");

        let tc = TurnupClass {
            blocked_plain_level: 2,
        };
        let legacy = legacy_writers::WStrategyTable {
            meta: SolvedStateMeta {
                score: (11, 11),
                turnup_class: tc,
                iterations: 100,
                num_info_sets: 1,
            },
            entries: vec![legacy_writers::WStrategyEntry {
                key: 42,
                info_set: legacy_writers::WInfoSet {
                    player: 1,
                    turnup_class: tc,
                    starting_hand: legacy_hand(),
                    history: ActionHistory::new(),
                },
                actions: vec![AbstractAction::AcceptEleven, AbstractAction::FoldEleven],
                average_strategy: vec![0.75, 0.25],
            }],
        };
        fs::write(&path, bincode_v1::serialize(&legacy).unwrap()).unwrap();

        let (table, meta) = load_strategy(&path).unwrap();
        assert_eq!(meta.score, (11, 11));
        // One legacy (position-averaged) entry expands into both positions.
        assert_eq!(table.len(), 2);
        for info_set in expected_position_variants(tc) {
            let data = table
                .get(&info_set)
                .expect("legacy entry expanded to this position");
            let avg = data.average_strategy();
            assert!((avg[0] - 0.75).abs() < 1e-6);
            assert!((avg[1] - 0.25).abs() < 1e-6);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_legacy_checkpoint_loads_with_position_expansion() {
        let dir = std::env::temp_dir().join("truco_solver_test_legacy_ckpt");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("legacy.ckpt.bin");

        let tc = TurnupClass {
            blocked_plain_level: 2,
        };
        let legacy = legacy_writers::WCheckpoint {
            meta: legacy_writers::WCheckpointMeta {
                score: (11, 11),
                turnup_class: tc,
                algo: "CFR+".to_string(),
                iteration: 77,
                num_info_sets: 1,
            },
            entries: vec![legacy_writers::WCheckpointEntry {
                key: 42,
                info_set: legacy_writers::WInfoSet {
                    player: 1,
                    turnup_class: tc,
                    starting_hand: legacy_hand(),
                    history: ActionHistory::new(),
                },
                actions: vec![AbstractAction::AcceptEleven, AbstractAction::FoldEleven],
                cumulative_regret: vec![4.5, 0.0],
                cumulative_strategy: vec![7.0, 3.0],
            }],
        };
        fs::write(&path, bincode_v1::serialize(&legacy).unwrap()).unwrap();

        let (table, meta) = load_checkpoint(&path).unwrap();
        assert_eq!(meta.iteration, 77);
        assert_eq!(meta.dealer_filter, None);
        assert_eq!(table.len(), 2);
        for info_set in expected_position_variants(tc) {
            let data = table
                .get(&info_set)
                .expect("legacy entry expanded to this position");
            assert_eq!(data.cumulative_regret, vec![4.5, 0.0]);
            assert_eq!(data.cumulative_strategy, vec![7.0, 3.0]);
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
