//! Serde (de)serialization pinned to the bincode 1 wire format.
//!
//! Every artifact this crate has ever written — strategy tables, checkpoints,
//! deep-solve checkpoints, treepacks, teacher band sidecars, match-value
//! tables — used bincode 1's default options: little-endian, fixed-width
//! integers, no length limit. bincode 2 removed the free functions
//! (`serialize`, `serialize_into`, `deserialize`, `deserialize_from`) and its
//! new default config is varint-encoded, which would silently change the
//! on-disk format and orphan existing GCS artifacts. `bincode::config::legacy()`
//! reproduces the v1 format exactly.
//!
//! All bincode traffic in this crate must go through these wrappers so the
//! wire format never drifts. Signatures mirror the bincode 1 free functions
//! (writer-first `serialize_into`, trailing bytes tolerated on slice
//! decode) to keep call sites unchanged apart from the module path. Format
//! parity with real bincode-1 bytes is asserted in this module's tests.

use bincode::config::{Configuration, Fixint, LittleEndian, NoLimit};
use bincode::error::{DecodeError, EncodeError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::{Read, Write};

/// Little-endian, fixed-width integers, no limit: bincode 1's default format.
const V1_CONFIG: Configuration<LittleEndian, Fixint, NoLimit> = bincode::config::legacy();

pub fn serialize<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, EncodeError> {
    bincode::serde::encode_to_vec(value, V1_CONFIG)
}

pub fn serialize_into<W: Write, T: Serialize + ?Sized>(
    mut writer: W,
    value: &T,
) -> Result<(), EncodeError> {
    bincode::serde::encode_into_std_write(value, &mut writer, V1_CONFIG).map(|_| ())
}

/// Trailing bytes after the value are ignored, matching bincode 1's
/// `deserialize` (the legacy-layout fallbacks in `storage.rs` depend on this).
pub fn deserialize<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, DecodeError> {
    bincode::serde::decode_from_slice(bytes, V1_CONFIG).map(|(value, _read)| value)
}

pub fn deserialize_from<R: Read, T: DeserializeOwned>(mut reader: R) -> Result<T, DecodeError> {
    bincode::serde::decode_from_std_read(&mut reader, V1_CONFIG)
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    enum GoldenAction {
        PlayFaceUp(u8),
        Hidden,
        Raise(u8),
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct GoldenEntry {
        key: u64,
        score: (u8, u8),
        algo: String,
        num_info_sets: usize,
        dealer_filter: Option<u8>,
        is_dealer: bool,
        actions: Vec<GoldenAction>,
        cumulative_regret: Vec<f64>,
        average_strategy: Vec<f32>,
    }

    fn golden_entry() -> GoldenEntry {
        GoldenEntry {
            key: 0xDEAD_BEEF_CAFE_F00D,
            score: (10, 7),
            algo: "sync-cfr+".to_string(),
            num_info_sets: 11_000_000,
            dealer_filter: Some(1),
            is_dealer: true,
            actions: vec![
                GoldenAction::PlayFaceUp(4),
                GoldenAction::Hidden,
                GoldenAction::Raise(9),
            ],
            cumulative_regret: vec![0.0, -1.5, 42.42],
            average_strategy: vec![0.25, 0.75],
        }
    }

    /// Bytes produced by `bincode 1.3.3::serialize(&golden_entry())` before
    /// the bincode 2 migration. If this test fails, the wire format changed
    /// and existing artifacts (GCS checkpoints, treepacks, teacher sidecars)
    /// are no longer readable — that is a release blocker, not a snapshot to
    /// regenerate.
    const GOLDEN_V1_BYTES: [u8; 108] = [
        0x0D, 0xF0, 0xFE, 0xCA, 0xEF, 0xBE, 0xAD, 0xDE, // key (u64 LE)
        0x0A, 0x07, // score (u8, u8)
        0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // algo len (u64 LE)
        0x73, 0x79, 0x6E, 0x63, 0x2D, 0x63, 0x66, 0x72, 0x2B, // "sync-cfr+"
        0xC0, 0xD8, 0xA7, 0x00, 0x00, 0x00, 0x00, 0x00, // num_info_sets (usize as u64 LE)
        0x01, 0x01, // Some(1)
        0x01, // is_dealer
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // actions len
        0x00, 0x00, 0x00, 0x00, 0x04, // PlayFaceUp(4): variant 0 (u32 LE) + payload
        0x01, 0x00, 0x00, 0x00, // Hidden: variant 1
        0x02, 0x00, 0x00, 0x00, 0x09, // Raise(9): variant 2 + payload
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // regrets len
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 0.0f64
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0xBF, // -1.5f64
        0xF6, 0x28, 0x5C, 0x8F, 0xC2, 0x35, 0x45, 0x40, // 42.42f64
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // strategy len
        0x00, 0x00, 0x80, 0x3E, // 0.25f32
        0x00, 0x00, 0x40, 0x3F, // 0.75f32
    ];

    #[test]
    fn golden_v1_bytes_roundtrip() {
        let encoded = serialize(&golden_entry()).unwrap();
        assert_eq!(encoded.as_slice(), GOLDEN_V1_BYTES.as_slice());

        let decoded: GoldenEntry = deserialize(&GOLDEN_V1_BYTES).unwrap();
        assert_eq!(decoded, golden_entry());
    }

    #[test]
    fn streaming_matches_slice_and_tolerates_trailing_bytes() {
        let mut streamed: Vec<u8> = Vec::new();
        serialize_into(&mut streamed, &golden_entry()).unwrap();
        serialize_into(&mut streamed, &2u64).unwrap();
        assert_eq!(
            &streamed[..GOLDEN_V1_BYTES.len()],
            GOLDEN_V1_BYTES.as_slice()
        );

        let mut reader = std::io::Cursor::new(&streamed);
        let entry: GoldenEntry = deserialize_from(&mut reader).unwrap();
        let count: u64 = deserialize_from(&mut reader).unwrap();
        assert_eq!(entry, golden_entry());
        assert_eq!(count, 2);

        // Slice decode must ignore trailing bytes (legacy-fallback decodes
        // pass whole-file buffers that keep decoding as older layouts).
        let entry: GoldenEntry = deserialize(&streamed).unwrap();
        assert_eq!(entry, golden_entry());
    }
}
