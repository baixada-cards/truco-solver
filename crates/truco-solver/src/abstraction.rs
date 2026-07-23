//! Solver-facing compatibility re-export for the shared policy abstraction.
//!
//! The canonical types live in `truco-policy-format` so runtime consumers do
//! not need to depend on this solver crate.

pub use truco_policy_format::abstraction::*;
