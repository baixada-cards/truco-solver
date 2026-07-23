// Throwaway: synthesize a COMPLETE match-value table (all reachable successor
// scores solved with placeholder values) so the deep-solve CLI probe can run
// its successor-solvedness assertion at toy scale. Not for production use.
use truco_solver::match_value::MatchValueTable;
use truco_solver::storage::save_match_values;

fn main() {
    let mut mv = MatchValueTable::new();
    for s0 in 0..12u8 {
        for s1 in 0..12u8 {
            for dealer in 0..2u8 {
                // Placeholder: dealer edge, mild score dependence, in [-1,1] q-scale.
                let v =
                    0.10 * (s0 as f64 - s1 as f64) / 12.0 + if dealer == 0 { 0.05 } else { -0.05 };
                mv.set(s0, s1, dealer, v.clamp(-0.99, 0.99));
            }
        }
    }
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/mv_full.bin".into());
    save_match_values(std::path::Path::new(&path), &mv).unwrap();
    eprintln!("wrote {path}");
}
