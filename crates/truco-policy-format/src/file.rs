//! Bot-policy artifact: a disk-resident, mmap-able average-strategy lookup.
//!
//! `export-bot-policy` (bin/solve.rs) writes one file per solved
//! `(score, turnup class, dealer)` profile; the live-match solver bot mmaps
//! them and binary-searches per decision. The format trades everything for
//! lookup locality: fixed 24-byte records sorted by `InfoSetKey`, u8-quantized
//! probabilities (the bot samples, and quantization noise at 1/255 sits far
//! below the certified transfer eps), no in-RAM index. A full 10x10 profile is
//! ~1 GB on disk and costs ~25 page touches per human-paced decision.
//!
//! Layout (little-endian):
//!   magic  b"TPB1"            4 bytes
//!   version u16 = 1           2 bytes
//!   reserved u16 = 0          2 bytes
//!   count  u64                8 bytes
//!   records: count x 24 bytes, ascending by key
//!     key    u64
//!     actions [u8; 8]         AbstractAction::to_u8 codes, 0xFF = unused slot
//!     probs   [u8; 8]         quantized; renormalized over used slots on read

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;
use smallvec::SmallVec;

use crate::info_set::{AbstractAction, InfoSetKey};
/// Errors while validating, reading, or writing a policy artifact.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PolicyFormatError {
    #[error("policy I/O error: {0}")]
    Io(String),
    #[error("policy serialization error: {0}")]
    Serialize(String),
    #[error("policy deserialization error: {0}")]
    Deserialize(String),
}

const MAGIC: &[u8; 4] = b"TPB1";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 16;
const RECORD_LEN: usize = 24;
pub const MAX_ACTIONS: usize = 8;
const UNUSED_SLOT: u8 = 0xFF;

/// One decoded policy entry: the info set's tree actions and their
/// (renormalized) average-strategy probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct BotPolicyEntry {
    pub actions: SmallVec<[AbstractAction; MAX_ACTIONS]>,
    pub probabilities: SmallVec<[f32; MAX_ACTIONS]>,
}

/// Write a bot-policy file from `(key, action codes, probabilities)` rows.
/// Rows may arrive in any order; probabilities need not be normalized.
pub fn write_bot_policy(
    path: &Path,
    entries: impl Iterator<
        Item = (
            u64,
            SmallVec<[u8; MAX_ACTIONS]>,
            SmallVec<[f32; MAX_ACTIONS]>,
        ),
    >,
) -> Result<u64, PolicyFormatError> {
    let mut records: Vec<[u8; RECORD_LEN]> = Vec::new();
    for (key, actions, probs) in entries {
        if actions.len() != probs.len() {
            return Err(PolicyFormatError::Serialize(
                "bot policy action/probability length mismatch".into(),
            ));
        }
        if actions.is_empty() || actions.len() > MAX_ACTIONS {
            return Err(PolicyFormatError::Serialize(format!(
                "bot policy entry has {} actions; expected 1..={MAX_ACTIONS}",
                actions.len()
            )));
        }
        let mut record = [0u8; RECORD_LEN];
        record[..8].copy_from_slice(&key.to_le_bytes());
        let total: f64 = probs.iter().map(|&p| p.max(0.0) as f64).sum();
        for i in 0..MAX_ACTIONS {
            if i < actions.len() {
                if actions[i] == UNUSED_SLOT {
                    return Err(PolicyFormatError::Serialize(
                        "bot policy action byte 0xFF collides with the unused-slot sentinel".into(),
                    ));
                }
                if AbstractAction::try_from_u8(actions[i]).is_none() {
                    return Err(PolicyFormatError::Serialize(format!(
                        "invalid bot policy action byte {}",
                        actions[i]
                    )));
                }
                record[8 + i] = actions[i];
                let p = if total > 0.0 {
                    (probs[i].max(0.0) as f64) / total
                } else {
                    1.0 / actions.len() as f64
                };
                record[16 + i] = (p * 255.0).round().clamp(0.0, 255.0) as u8;
            } else {
                record[8 + i] = UNUSED_SLOT;
                record[16 + i] = 0;
            }
        }
        records.push(record);
    }
    records.sort_unstable_by(|a, b| a[..8].cmp(&b[..8]));
    for pair in records.windows(2) {
        if pair[0][..8] == pair[1][..8] {
            return Err(PolicyFormatError::Serialize(format!(
                "duplicate bot policy key {}",
                u64::from_le_bytes(pair[0][..8].try_into().expect("8-byte key slice"))
            )));
        }
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| PolicyFormatError::Io(e.to_string()))?;
    }
    let tmp_path = path.with_extension("tmp");
    {
        let file = fs::File::create(&tmp_path).map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
        writer
            .write_all(MAGIC)
            .map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        writer
            .write_all(&VERSION.to_le_bytes())
            .map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        writer
            .write_all(&0u16.to_le_bytes())
            .map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        writer
            .write_all(&(records.len() as u64).to_le_bytes())
            .map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        for record in &records {
            writer
                .write_all(record)
                .map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        }
        writer
            .flush()
            .map_err(|e| PolicyFormatError::Io(e.to_string()))?;
    }
    fs::rename(&tmp_path, path).map_err(|e| PolicyFormatError::Io(e.to_string()))?;
    Ok(records.len() as u64)
}

/// An open, mmap-ed bot-policy file. Cheap to clone-share behind an `Arc`;
/// lookups touch O(log n) pages and allocate nothing beyond the entry.
#[derive(Debug)]
pub struct BotPolicyFile {
    map: Mmap,
    count: usize,
}

impl BotPolicyFile {
    pub fn open(path: &Path) -> Result<Self, PolicyFormatError> {
        let file = fs::File::open(path).map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        // Safety: the file is written atomically by `write_bot_policy` and
        // treated as immutable once published; remapping on truncation is the
        // standard mmap caveat accepted across this codebase's readers.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| PolicyFormatError::Io(e.to_string()))?;
        if map.len() < HEADER_LEN || &map[..4] != MAGIC {
            return Err(PolicyFormatError::Deserialize(
                "not a bot-policy file".into(),
            ));
        }
        let version = u16::from_le_bytes(map[4..6].try_into().expect("2-byte version"));
        if version != VERSION {
            return Err(PolicyFormatError::Deserialize(format!(
                "unsupported bot-policy version {version}"
            )));
        }
        let reserved = u16::from_le_bytes(map[6..8].try_into().expect("2-byte reserved field"));
        if reserved != 0 {
            return Err(PolicyFormatError::Deserialize(format!(
                "bot-policy reserved header field is {reserved}, expected 0"
            )));
        }
        let count_u64 = u64::from_le_bytes(map[8..16].try_into().expect("8-byte count"));
        let count = usize::try_from(count_u64).map_err(|_| {
            PolicyFormatError::Deserialize(format!(
                "bot-policy record count {count_u64} exceeds this platform"
            ))
        })?;
        let expected_len = count
            .checked_mul(RECORD_LEN)
            .and_then(|records_len| HEADER_LEN.checked_add(records_len))
            .ok_or_else(|| {
                PolicyFormatError::Deserialize(format!(
                    "bot-policy record count {count_u64} overflows file length"
                ))
            })?;
        if map.len() != expected_len {
            return Err(PolicyFormatError::Deserialize(format!(
                "bot-policy length {} does not match {count} records",
                map.len()
            )));
        }
        Ok(Self { map, count })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn record(&self, index: usize) -> &[u8] {
        let start = HEADER_LEN + index * RECORD_LEN;
        &self.map[start..start + RECORD_LEN]
    }

    pub fn lookup_checked(
        &self,
        key: InfoSetKey,
    ) -> Result<Option<BotPolicyEntry>, PolicyFormatError> {
        let target = key.0.to_le_bytes();
        let mut lo = 0usize;
        let mut hi = self.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let record = self.record(mid);
            match record[..8].cmp(&target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return decode_record(record).map(Some),
            }
        }
        Ok(None)
    }

    /// Compatibility convenience for trusted, checksum-verified artifacts.
    ///
    /// New runtime code should prefer [`Self::lookup_checked`] so a malformed
    /// matching record is distinguishable from a missing key.
    pub fn lookup(&self, key: InfoSetKey) -> Option<BotPolicyEntry> {
        self.lookup_checked(key).ok().flatten()
    }
}

fn decode_record(record: &[u8]) -> Result<BotPolicyEntry, PolicyFormatError> {
    let mut actions = SmallVec::new();
    let mut raw = SmallVec::<[f32; MAX_ACTIONS]>::new();
    for i in 0..MAX_ACTIONS {
        let code = record[8 + i];
        if code == UNUSED_SLOT {
            if record[8 + i..16]
                .iter()
                .any(|remaining| *remaining != UNUSED_SLOT)
            {
                return Err(PolicyFormatError::Deserialize(
                    "bot-policy record has an action after an unused slot".into(),
                ));
            }
            break;
        }
        let action = AbstractAction::try_from_u8(code).ok_or_else(|| {
            PolicyFormatError::Deserialize(format!("invalid bot policy action byte {code}"))
        })?;
        actions.push(action);
        raw.push(record[16 + i] as f32);
    }
    if actions.is_empty() {
        return Err(PolicyFormatError::Deserialize(
            "bot-policy record has no actions".into(),
        ));
    }
    let total: f32 = raw.iter().sum();
    let probabilities = if total > 0.0 {
        raw.iter().map(|&q| q / total).collect()
    } else {
        let uniform = 1.0 / actions.len().max(1) as f32;
        raw.iter().map(|_| uniform).collect()
    };
    Ok(BotPolicyEntry {
        actions,
        probabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        key: u64,
        actions: &[u8],
        probs: &[f32],
    ) -> (
        u64,
        SmallVec<[u8; MAX_ACTIONS]>,
        SmallVec<[f32; MAX_ACTIONS]>,
    ) {
        (
            key,
            actions.iter().copied().collect(),
            probs.iter().copied().collect(),
        )
    }

    #[test]
    fn roundtrip_lookup_and_sorting() {
        let dir = std::env::temp_dir().join(format!("tpb-test-{}", std::process::id()));
        let path = dir.join("roundtrip.tpb");
        let rows = vec![
            entry(42, &[0, 27, 32], &[0.5, 0.25, 0.25]),
            entry(7, &[31], &[1.0]),
            entry(u64::MAX, &[13, 26], &[0.0, 0.0]),
        ];
        let written = write_bot_policy(&path, rows.into_iter()).expect("write");
        assert_eq!(written, 3);

        let file = BotPolicyFile::open(&path).expect("open");
        assert_eq!(file.len(), 3);

        let mixed = file.lookup(InfoSetKey(42)).expect("key 42");
        assert_eq!(
            mixed.actions.as_slice(),
            &[
                AbstractAction::from_u8(0),
                AbstractAction::from_u8(27),
                AbstractAction::from_u8(32)
            ]
        );
        let probs = mixed.probabilities.as_slice();
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((probs[0] - 0.5).abs() < 0.01);
        assert!((probs[1] - 0.25).abs() < 0.01);

        let pure = file.lookup(InfoSetKey(7)).expect("key 7");
        assert_eq!(pure.probabilities.as_slice(), &[1.0]);

        // all-zero cumulative mass renormalizes to uniform
        let zeroed = file.lookup(InfoSetKey(u64::MAX)).expect("max key");
        assert_eq!(zeroed.probabilities.as_slice(), &[0.5, 0.5]);

        assert_eq!(file.lookup(InfoSetKey(1)), None);
        assert_eq!(file.lookup(InfoSetKey(u64::MAX - 1)), None);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn duplicate_keys_refuse() {
        let dir = std::env::temp_dir().join(format!("tpb-test-dup-{}", std::process::id()));
        let path = dir.join("dup.tpb");
        let rows = vec![entry(1, &[0], &[1.0]), entry(1, &[1], &[1.0])];
        assert!(write_bot_policy(&path, rows.into_iter()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn key_lookup_uses_the_v1_encoded_byte_order() {
        // Little-endian byte order differs from integer order. TPB1 v1
        // canonically sorts encoded bytes, so 0x0100 precedes 0x00FF.
        let dir = std::env::temp_dir().join(format!("tpb-test-ord-{}", std::process::id()));
        let path = dir.join("ord.tpb");
        let rows = vec![entry(0x0100, &[0], &[1.0]), entry(0x00FF, &[1], &[1.0])];
        write_bot_policy(&path, rows.into_iter()).expect("write");
        let file = BotPolicyFile::open(&path).expect("open");
        assert_eq!(
            u64::from_le_bytes(file.record(0)[..8].try_into().unwrap()),
            0x0100
        );
        assert!(file.lookup(InfoSetKey(0x0100)).is_some());
        assert!(file.lookup(InfoSetKey(0x00FF)).is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn golden_single_record_bytes_v1() {
        let dir = std::env::temp_dir().join(format!("tpb-test-golden-{}", std::process::id()));
        let path = dir.join("golden.tpb");
        write_bot_policy(
            &path,
            vec![entry(0x0102_0304_0506_0708, &[0, 27], &[0.75, 0.25])].into_iter(),
        )
        .expect("write");

        let expected = [
            0x54, 0x50, 0x42, 0x31, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00, 0x1b, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xbf, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(std::fs::read(&path).unwrap(), expected);
        std::fs::remove_dir_all(&dir).ok();
    }
}
