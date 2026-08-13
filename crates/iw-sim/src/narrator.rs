//! Progress narration: the Maxis-grade snark pool (DESIGN.md §9).
//!
//! Lines live in `assets/snark.txt` so adding jokes never touches code. The
//! asset is embedded at compile time; a runtime override can be parsed with
//! [`Narrator::from_str`].

use iw_core::{rng_for, Phase};
use rand::RngCore;

/// The narration pool shipped with the binary (`assets/snark.txt`).
pub const EMBEDDED_SNARK: &str = include_str!("../../../assets/snark.txt");

/// RNG stream name used for line selection. Never collides with a process
/// stream because no process may be called `narrator`.
pub const NARRATOR_STREAM: &str = "narrator";

/// Section names in `Phase::ALL` order, followed by the always-mixed-in
/// `generic` section.
pub const SECTIONS: [&str; 5] = ["crust", "drift", "refine", "recent", "generic"];

/// Index of the `generic` section within [`SECTIONS`].
const GENERIC: usize = 4;

/// Parsed narration pool: one line list per section of `assets/snark.txt`.
#[derive(Debug, Clone)]
pub struct Narrator {
    sections: [Vec<String>; 5],
}

impl Default for Narrator {
    /// The embedded pool. Panics only if the shipped asset is malformed, which
    /// a unit test rules out.
    fn default() -> Self {
        Narrator::from_str(EMBEDDED_SNARK).expect("embedded snark.txt is valid")
    }
}

impl Narrator {
    /// Parse a narration pool.
    ///
    /// Sections start with `[name]`; blank lines and `#` comments are skipped.
    /// Unknown section names are an error, as is a line before any section, so
    /// a typo in the asset fails loudly instead of silently losing jokes.
    #[allow(clippy::should_implement_trait)] // not FromStr: we want an inherent, non-generic ctor
    pub fn from_str(text: &str) -> anyhow::Result<Narrator> {
        let mut sections: [Vec<String>; 5] = Default::default();
        let mut current: Option<usize> = None;
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                let name = name.trim();
                let idx = SECTIONS.iter().position(|s| *s == name).ok_or_else(|| {
                    anyhow::anyhow!("snark: unknown section [{name}] on line {}", lineno + 1)
                })?;
                current = Some(idx);
                continue;
            }
            let idx = current.ok_or_else(|| {
                anyhow::anyhow!("snark: line {} appears before any [section]", lineno + 1)
            })?;
            sections[idx].push(line.to_string());
        }
        for (name, lines) in SECTIONS.iter().zip(sections.iter()) {
            if lines.is_empty() {
                anyhow::bail!("snark: section [{name}] is empty");
            }
        }
        Ok(Narrator { sections })
    }

    /// Lines of one section, or `None` if the name is not a known section.
    pub fn section(&self, name: &str) -> Option<&[String]> {
        SECTIONS
            .iter()
            .position(|s| *s == name)
            .map(|i| self.sections[i].as_slice())
    }

    /// Lines specific to a phase (excluding `generic`).
    pub fn phase_lines(&self, phase: Phase) -> &[String] {
        &self.sections[phase.index()]
    }

    /// Lines of the `generic` section.
    pub fn generic_lines(&self) -> &[String] {
        &self.sections[GENERIC]
    }

    /// Pick a line for `phase`, deterministically from `(seed, step_index)`.
    ///
    /// The candidate pool is the phase section followed by `generic`, so both
    /// flavours appear and the choice is stable across runs and machines.
    pub fn line(&self, phase: Phase, seed: u64, step_index: u64) -> &str {
        let phase_lines = self.phase_lines(phase);
        let generic = self.generic_lines();
        let total = phase_lines.len() + generic.len();
        let mut rng = rng_for(seed, NARRATOR_STREAM, step_index);
        let pick = (rng.next_u64() % total as u64) as usize;
        if pick < phase_lines.len() {
            &phase_lines[pick]
        } else {
            &generic[pick - phase_lines.len()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_asset_parses_with_full_sections() {
        let n = Narrator::default();
        for name in SECTIONS {
            let lines = n
                .section(name)
                .unwrap_or_else(|| panic!("missing [{name}]"));
            assert!(
                lines.len() >= 12,
                "[{name}] has {} lines, need >= 12",
                lines.len()
            );
            assert!(lines.iter().all(|l| !l.is_empty()));
        }
        assert!(
            n.generic_lines()
                .iter()
                .any(|l| l == "Reticulating splines"),
            "the one obligatory line is missing from [generic]"
        );
    }

    #[test]
    fn selection_is_deterministic_and_varies() {
        let n = Narrator::default();
        let a: Vec<&str> = (0..40).map(|i| n.line(Phase::Drift, 42, i)).collect();
        let b: Vec<&str> = (0..40).map(|i| n.line(Phase::Drift, 42, i)).collect();
        assert_eq!(a, b);
        let c: Vec<&str> = (0..40).map(|i| n.line(Phase::Drift, 43, i)).collect();
        assert_ne!(a, c);
        // Every pick must come from the drift or generic pools.
        let pool: Vec<&str> = n
            .phase_lines(Phase::Drift)
            .iter()
            .chain(n.generic_lines())
            .map(|s| s.as_str())
            .collect();
        assert!(a.iter().all(|l| pool.contains(l)));
        assert!(a.iter().collect::<std::collections::HashSet<_>>().len() > 5);
    }

    #[test]
    fn parser_rejects_bad_input() {
        assert!(Narrator::from_str("[nope]\nline\n").is_err());
        assert!(Narrator::from_str("orphan line\n").is_err());
        assert!(Narrator::from_str("[crust]\nonly crust\n").is_err());
    }

    #[test]
    fn parser_skips_comments_and_blanks() {
        let text = "# header\n\n[crust]\n a \n# note\n[drift]\nb\n[refine]\nc\n[recent]\nd\n[generic]\ne\n";
        let n = Narrator::from_str(text).unwrap();
        assert_eq!(n.section("crust").unwrap(), ["a".to_string()]);
        assert_eq!(n.section("generic").unwrap(), ["e".to_string()]);
        assert!(n.section("nope").is_none());
    }
}
