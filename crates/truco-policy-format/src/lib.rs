//! Stable policy interchange contract shared by policy producers and runtimes.
//!
//! This crate deliberately contains no CFR implementation. It owns the
//! abstract card and action vocabulary, deterministic information-set keys,
//! and the versioned TPB1 policy file codec. A solver can produce these files
//! and a gameplay bot can consume them without the bot depending on training,
//! checkpoint, experiment, or research code.

pub mod abstraction;
pub mod file;
pub mod info_set;
pub mod manifest;
