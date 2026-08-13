//! Test-only crate: no public API. All tests live in `tests/golden.rs` and
//! run the real simulation pipeline in-process (see IMPLEMENTATION_PLAN.md
//! §3 WP12) — end-to-end determinism, golden-planet stat ranges for seeds
//! 42 and 1337, mass-ledger balance, config sanitize round-tripping, and
//! checkpoint-resume equivalence.
