//! Treepack: the on-disk / GCS-cacheable form of a prebuilt tree set.
//!
//! Trees are identical for every score state within a ladder band (see
//! `game_tree::band_signature`), so one treepack serves a whole band: 5 bands
//! × 9 turn-up classes × 2 dealers ≈ 90 artifacts for the entire game.
//!
//! The node/edge arenas are stored as raw packed bytes at 8-aligned offsets,
//! so `load_treepack` memory-maps the file and hands out `GameTree` handles
//! directly over the mapping — no deserialization of the arenas, and the OS
//! pages a larger-than-RAM arena in and out at SSD speed (the deep-state
//! streaming path). Only the info-set metadata is bincode-decoded.

use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use crate::game_tree::{assemble_entries, band_signature, PrebuiltTrees, TreeBytes};
use crate::info_set::{InfoSet, InfoSetKey};
use crate::storage::StorageError;
use truco_engine::{Player, Score};

const MAGIC: u64 = 0x5452_5543_4f54_5245; // "TRUCOTRE"
const VERSION: u64 = 1;
/// Header layout (u64 words):
/// [magic, version, sig_hash, n_deals, n_nodes, n_edges,
///  spans_off, weights_off, nodes_off, edges_off, info_off, info_len]
const HEADER_WORDS: usize = 12;

fn align8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

fn sig_hash(sig: &str, tc: u8) -> u64 {
    // Fixed-seed hash (must be stable across processes, like InfoSet::key).
    let state = ahash::RandomState::with_seeds(1, 2, 3, 4);
    state.hash_one((sig, tc))
}

/// Stable hash binding an artifact to its (band signature, tc); shared by
/// treepacks and `.teach` teacher files.
pub fn band_sig_hash(score: &Score, tc: u8, dealer_filter: Option<Player>) -> u64 {
    sig_hash(&band_signature(score, dealer_filter), tc)
}

/// Canonical treepack filename for a (score-band, tc, dealer-filter).
pub fn treepack_name(score: &Score, tc: u8, dealer_filter: Option<Player>) -> String {
    format!(
        "{}-tc{}-v{}.trees",
        band_signature(score, dealer_filter),
        tc,
        VERSION
    )
}

/// Serialize a freshly built tree set. The arenas are written verbatim from
/// the owned build buffers.
pub fn save_treepack(
    path: &Path,
    prebuilt: &PrebuiltTrees,
    score: &Score,
    tc: u8,
    dealer_filter: Option<Player>,
) -> Result<(), StorageError> {
    use std::io::Write;

    let sig = band_signature(score, dealer_filter);
    let info_bin: Vec<u8> = bincode::serialize(
        &prebuilt
            .info_sets
            .iter()
            .map(|(k, i, a)| (k.0, i, a))
            .collect::<Vec<_>>(),
    )
    .map_err(|e| StorageError::Serialize(e.to_string()))?;

    let n_deals = prebuilt.spans.len();
    let nodes_bytes = prebuilt.nodes_buf.as_bytes();
    let edges_bytes = prebuilt.edges_buf.as_bytes();
    let (n_nodes, n_edges) = {
        let mut nn = 0u64;
        let mut ne = 0u64;
        for span in &prebuilt.spans {
            for d in span {
                nn = nn.max(d.0 + d.1);
                ne = ne.max(d.2 + d.3);
            }
        }
        (nn, ne)
    };

    let spans_off = HEADER_WORDS * 8;
    let weights_off = spans_off + n_deals * 8 * 8;
    let nodes_off = align8(weights_off + n_deals * 8);
    let edges_off = align8(nodes_off + (n_nodes as usize) * 12);
    let info_off = align8(edges_off + (n_edges as usize) * 8);

    let header: [u64; HEADER_WORDS] = [
        MAGIC,
        VERSION,
        sig_hash(&sig, tc),
        n_deals as u64,
        n_nodes,
        n_edges,
        spans_off as u64,
        weights_off as u64,
        nodes_off as u64,
        edges_off as u64,
        info_off as u64,
        info_bin.len() as u64,
    ];

    let tmp = path.with_extension("trees.tmp");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut w = std::io::BufWriter::with_capacity(
        8 << 20,
        File::create(&tmp).map_err(|e| StorageError::Io(e.to_string()))?,
    );
    let mut written = 0usize;
    let put = |w: &mut dyn Write, bytes: &[u8]| -> Result<(), StorageError> {
        w.write_all(bytes)
            .map_err(|e| StorageError::Io(e.to_string()))
    };
    for h in header {
        put(&mut w, &h.to_le_bytes())?;
    }
    written += HEADER_WORDS * 8;
    for span in &prebuilt.spans {
        for d in span {
            for v in [d.0, d.1, d.2, d.3] {
                put(&mut w, &v.to_le_bytes())?;
            }
        }
    }
    written += n_deals * 8 * 8;
    for e in &prebuilt.entries {
        put(&mut w, &e.weight.to_le_bytes())?;
    }
    written += n_deals * 8;
    put(&mut w, &vec![0u8; nodes_off - written])?;
    written = nodes_off;
    put(&mut w, &nodes_bytes[..(n_nodes as usize) * 12])?;
    written += (n_nodes as usize) * 12;
    put(&mut w, &vec![0u8; edges_off - written])?;
    written = edges_off;
    put(&mut w, &edges_bytes[..(n_edges as usize) * 8])?;
    written += (n_edges as usize) * 8;
    put(&mut w, &vec![0u8; info_off - written])?;
    put(&mut w, &info_bin)?;
    drop(w);
    fs::rename(&tmp, path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// Memory-map a treepack and reconstruct `PrebuiltTrees` with `GameTree`
/// handles pointing straight into the mapping. Validates magic, version,
/// band signature, and section bounds before trusting any offset.
pub fn load_treepack(
    path: &Path,
    score: &Score,
    tc: u8,
    dealer_filter: Option<Player>,
) -> Result<PrebuiltTrees, StorageError> {
    let file = File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
    // SAFETY: read-only private mapping of a regular file; we validate every
    // offset/length against the mapping size before use. Concurrent external
    // modification of a cache artifact is outside the threat model.
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| StorageError::Io(e.to_string()))?;
    let bytes: &[u8] = &map;
    let word = |i: usize| -> Result<u64, StorageError> {
        let off = i * 8;
        bytes
            .get(off..off + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
            .ok_or_else(|| StorageError::Deserialize("treepack truncated header".into()))
    };
    if word(0)? != MAGIC {
        return Err(StorageError::Deserialize("treepack bad magic".into()));
    }
    if word(1)? != VERSION {
        return Err(StorageError::Deserialize(
            "treepack version mismatch".into(),
        ));
    }
    let sig = band_signature(score, dealer_filter);
    if word(2)? != sig_hash(&sig, tc) {
        return Err(StorageError::Deserialize(format!(
            "treepack band-signature mismatch (want {sig})"
        )));
    }
    let n_deals = word(3)? as usize;
    let n_nodes = word(4)? as usize;
    let n_edges = word(5)? as usize;
    let spans_off = word(6)? as usize;
    let weights_off = word(7)? as usize;
    let nodes_off = word(8)? as usize;
    let edges_off = word(9)? as usize;
    let info_off = word(10)? as usize;
    let info_len = word(11)? as usize;
    let need = info_off + info_len;
    if bytes.len() < need
        || nodes_off + n_nodes * 12 > bytes.len()
        || edges_off + n_edges * 8 > bytes.len()
        || nodes_off % 8 != 0
        || edges_off % 8 != 0
    {
        return Err(StorageError::Deserialize(
            "treepack truncated/misaligned".into(),
        ));
    }

    let mut spans = Vec::with_capacity(n_deals);
    let mut weights = Vec::with_capacity(n_deals);
    for i in 0..n_deals {
        let base = spans_off / 8 + i * 8;
        let mut span = [(0u64, 0u64, 0u64, 0u64); 2];
        for (d, item) in span.iter_mut().enumerate() {
            *item = (
                word(base + d * 4)?,
                word(base + d * 4 + 1)?,
                word(base + d * 4 + 2)?,
                word(base + d * 4 + 3)?,
            );
            // bounds: every span must sit inside the declared arenas
            if item.0 + item.1 > n_nodes as u64 || item.2 + item.3 > n_edges as u64 {
                return Err(StorageError::Deserialize(
                    "treepack span out of bounds".into(),
                ));
            }
        }
        spans.push(span);
        weights.push(f64::from_le_bytes(
            bytes[weights_off + i * 8..weights_off + i * 8 + 8]
                .try_into()
                .expect("8 bytes"),
        ));
    }

    let info_raw: Vec<(u64, InfoSet, Vec<crate::info_set::AbstractAction>)> =
        bincode::deserialize(&bytes[info_off..info_off + info_len])
            .map_err(|e| StorageError::Deserialize(e.to_string()))?;
    let info_sets = info_raw
        .into_iter()
        .map(|(k, i, a)| (InfoSetKey(k), i, a.into_iter().collect()))
        .collect();

    let buf = TreeBytes::Mapped(Arc::new(map));
    let entries = assemble_entries(&buf, &buf, nodes_off, edges_off, &spans, &weights);

    Ok(PrebuiltTrees {
        entries,
        info_sets,
        nodes_buf: buf.clone(),
        edges_buf: buf,
        spans,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{enumerate_deals, TurnupClass};
    use crate::game_tree::build_all_trees_with_dealer;

    #[test]
    fn test_treepack_roundtrip_and_band_identity() {
        let tc = TurnupClass {
            blocked_plain_level: 0,
        };
        let deals: Vec<_> = enumerate_deals(&tc).into_iter().take(60).collect();
        let s88 = Score { zero: 8, one: 8 };
        let s66 = Score { zero: 6, one: 6 };

        let built = build_all_trees_with_dealer(&s88, tc, &deals, Some(0)).unwrap();
        let dir = std::env::temp_dir().join("truco_treepack_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join(treepack_name(&s88, 0, Some(0)));
        save_treepack(&path, &built, &s88, 0, Some(0)).unwrap();

        // Round-trip: mapped handles must describe the identical tree set.
        let loaded = load_treepack(&path, &s88, 0, Some(0)).unwrap();
        assert_eq!(loaded.entries.len(), built.entries.len());
        assert_eq!(loaded.info_sets.len(), built.info_sets.len());
        for (a, b) in built.entries.iter().zip(loaded.entries.iter()) {
            assert_eq!(a.weight, b.weight);
            assert_eq!(a.tree_dealer_0.nodes.len(), b.tree_dealer_0.nodes.len());
            assert_eq!(a.tree_dealer_0.edges.len(), b.tree_dealer_0.edges.len());
        }
        for ((k1, i1, a1), (k2, i2, a2)) in built.info_sets.iter().zip(loaded.info_sets.iter()) {
            assert_eq!(k1, k2);
            assert_eq!(i1, i2);
            assert_eq!(a1, a2);
        }

        // Band identity: 6x6 shares 8x8's signature (same {1,3,6} ladder), so
        // the SAME artifact loads for it; while 10x10 ({1,3}) must be refused.
        assert_eq!(band_signature(&s88, Some(0)), band_signature(&s66, Some(0)));
        let for_66 = load_treepack(&path, &s66, 0, Some(0)).unwrap();
        assert_eq!(for_66.info_sets.len(), built.info_sets.len());
        let s1010 = Score { zero: 10, one: 10 };
        assert!(load_treepack(&path, &s1010, 0, Some(0)).is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}
