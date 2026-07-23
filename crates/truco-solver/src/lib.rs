//! Tabular CFR solver for abstracted Truco.
//!
//! Clippy: traversal and regret-update code uses explicit indices and
//! multi-argument recursion by design; suppress noisy style lints workspace-wide.
#![allow(clippy::derivable_impls)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::new_without_default)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

pub mod abstraction;
pub mod allocation_scout;
pub mod bot_policy;
pub mod cfr;
pub mod cfr_experiment;
pub mod compact_br;
pub mod deep;
pub mod game_tree;
pub mod info_set;
pub mod match_value;
pub mod resolve;
pub mod simulate;
pub mod storage;
pub mod strategy;
pub mod subgame;
pub mod teacher_export;
pub mod treepack;
