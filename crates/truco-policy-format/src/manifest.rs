//! Versioned manifest for a directory of TPB1 policy profiles.

use std::collections::HashSet;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

/// Manifest discriminator for the first public policy bundle contract.
pub const MANIFEST_FORMAT: &str = "truco-policy-bot/v1";

/// A policy bundle manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyManifest {
    pub format: String,
    pub profiles: Vec<PolicyProfile>,
}

/// One solve-frame profile in a policy bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyProfile {
    /// `(seat 0 score, seat 1 score)`.
    pub score: [u8; 2],
    /// Turnup class, in `0..=8`.
    pub tc: u8,
    /// Dealer seat, either `0` or `1`.
    pub dealer: u8,
    /// Safe bundle-local TPB1 filename.
    pub file: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PolicyManifestError {
    #[error("invalid policy manifest JSON: {0}")]
    Json(String),
    #[error("unsupported policy manifest format {0:?}")]
    UnsupportedFormat(String),
    #[error("profile {index} has invalid turnup class {tc}; expected 0..=8")]
    InvalidTurnupClass { index: usize, tc: u8 },
    #[error("profile {index} has invalid dealer {dealer}; expected 0 or 1")]
    InvalidDealer { index: usize, dealer: u8 },
    #[error("profile {index} has unsafe or invalid TPB1 filename {file:?}")]
    InvalidFile { index: usize, file: String },
    #[error("duplicate profile for score {score:?}, tc {tc}, dealer {dealer}")]
    DuplicateProfile { score: [u8; 2], tc: u8, dealer: u8 },
}

impl PolicyManifest {
    pub fn parse(raw: &str) -> Result<Self, PolicyManifestError> {
        let manifest: Self = serde_json::from_str(raw)
            .map_err(|error| PolicyManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), PolicyManifestError> {
        if self.format != MANIFEST_FORMAT {
            return Err(PolicyManifestError::UnsupportedFormat(self.format.clone()));
        }

        let mut profiles = HashSet::with_capacity(self.profiles.len());
        for (index, profile) in self.profiles.iter().enumerate() {
            if profile.tc > 8 {
                return Err(PolicyManifestError::InvalidTurnupClass {
                    index,
                    tc: profile.tc,
                });
            }
            if profile.dealer > 1 {
                return Err(PolicyManifestError::InvalidDealer {
                    index,
                    dealer: profile.dealer,
                });
            }
            if !is_safe_tpb_filename(&profile.file) {
                return Err(PolicyManifestError::InvalidFile {
                    index,
                    file: profile.file.clone(),
                });
            }
            if !profiles.insert((profile.score, profile.tc, profile.dealer)) {
                return Err(PolicyManifestError::DuplicateProfile {
                    score: profile.score,
                    tc: profile.tc,
                    dealer: profile.dealer,
                });
            }
        }
        Ok(())
    }
}

fn is_safe_tpb_filename(file: &str) -> bool {
    let path = Path::new(file);
    let mut components = path.components();
    matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && path.extension().is_some_and(|extension| extension == "tpb")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> PolicyManifest {
        PolicyManifest {
            format: MANIFEST_FORMAT.into(),
            profiles: vec![PolicyProfile {
                score: [10, 10],
                tc: 5,
                dealer: 0,
                file: "s10x10-tc5-d0.tpb".into(),
            }],
        }
    }

    #[test]
    fn parses_the_v1_contract() {
        let raw = serde_json::to_string(&valid_manifest()).unwrap();
        assert_eq!(PolicyManifest::parse(&raw).unwrap(), valid_manifest());
    }

    #[test]
    fn rejects_unknown_fields() {
        let raw = r#"{"format":"truco-policy-bot/v1","profiles":[],"extra":true}"#;
        assert!(matches!(
            PolicyManifest::parse(raw),
            Err(PolicyManifestError::Json(_))
        ));
    }

    #[test]
    fn rejects_traversal_and_absolute_paths() {
        for file in ["../secret.tpb", "nested/profile.tpb", "/tmp/profile.tpb"] {
            let mut manifest = valid_manifest();
            manifest.profiles[0].file = file.into();
            assert!(matches!(
                manifest.validate(),
                Err(PolicyManifestError::InvalidFile { .. })
            ));
        }
    }

    #[test]
    fn rejects_duplicate_profile_keys() {
        let mut manifest = valid_manifest();
        manifest.profiles.push(manifest.profiles[0].clone());
        assert!(matches!(
            manifest.validate(),
            Err(PolicyManifestError::DuplicateProfile { .. })
        ));
    }
}
