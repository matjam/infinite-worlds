//! The progress log: narration, milestones and phase events from the
//! simulation, folded into a bounded list of lines for the UI.

use std::collections::VecDeque;

use iw_core::{Phase, ProgressEvent};

use crate::panel::phase_name;

/// Lines kept for the collapsible log.
pub const LOG_CAPACITY: usize = 200;

/// Rolling event log plus the derived progress readout.
pub struct EventLog {
    lines: VecDeque<String>,
    narration: String,
    /// `(phase, step, of)` from the most recent step event.
    pub phase_progress: Option<(Phase, u64, u64)>,
    /// Set when a phase completes; the app drains it into its checkpoint list.
    pub last_completed_phase: Option<Phase>,
}

impl Default for EventLog {
    fn default() -> Self {
        EventLog {
            lines: VecDeque::with_capacity(LOG_CAPACITY),
            narration: "Warming up the mantle...".to_string(),
            phase_progress: None,
            last_completed_phase: None,
        }
    }
}

impl EventLog {
    /// Append a line of our own (not from the simulation).
    pub fn push(&mut self, line: String) {
        if self.lines.len() == LOG_CAPACITY {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    /// Fold one progress event in.
    pub fn event(&mut self, ev: ProgressEvent) {
        match ev {
            ProgressEvent::PhaseStarted { phase, time_myr } => {
                self.phase_progress = Some((phase, 0, 1));
                self.push(format!("[{time_myr:.1} Myr] {} started", phase_name(phase)));
            }
            ProgressEvent::PhaseCompleted { phase, time_myr } => {
                self.last_completed_phase = Some(phase);
                self.push(format!(
                    "[{time_myr:.1} Myr] {} complete (checkpoint saved)",
                    phase_name(phase)
                ));
            }
            ProgressEvent::Step {
                phase,
                step,
                of,
                time_myr: _,
            } => {
                self.phase_progress = Some((phase, step, of));
            }
            ProgressEvent::Narration(line) => {
                self.narration = line.clone();
                self.push(line);
            }
            ProgressEvent::Milestone(line) => {
                self.push(format!("* {line}"));
            }
        }
    }

    /// The most recent narration line.
    pub fn latest_narration(&self) -> &str {
        &self.narration
    }

    /// The last `n` lines, oldest first.
    pub fn recent(&self, n: usize) -> Vec<&str> {
        let skip = self.lines.len().saturating_sub(n);
        self.lines.iter().skip(skip).map(|s| s.as_str()).collect()
    }

    /// Total lines held.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.lines.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_events_become_lines_and_progress() {
        let mut log = EventLog::default();
        log.event(ProgressEvent::PhaseStarted {
            phase: Phase::Drift,
            time_myr: 200.0,
        });
        log.event(ProgressEvent::Step {
            phase: Phase::Drift,
            step: 40,
            of: 400,
            time_myr: 220.0,
        });
        assert_eq!(log.phase_progress, Some((Phase::Drift, 40, 400)));
        log.event(ProgressEvent::Narration("Reticulating splines".into()));
        assert_eq!(log.latest_narration(), "Reticulating splines");
        log.event(ProgressEvent::Milestone("a rift opened".into()));
        log.event(ProgressEvent::PhaseCompleted {
            phase: Phase::Drift,
            time_myr: 400.0,
        });
        assert_eq!(log.last_completed_phase, Some(Phase::Drift));

        let lines = log.recent(10);
        assert!(lines
            .iter()
            .any(|l| l.contains("Continental drift started")));
        assert!(lines.iter().any(|l| l.contains("Reticulating splines")));
        assert!(lines.iter().any(|l| l.starts_with("* a rift opened")));
        assert!(lines.iter().any(|l| l.contains("complete")));
        // Step events are progress only; they must not spam the log.
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn the_log_is_bounded_and_keeps_the_newest() {
        let mut log = EventLog::default();
        for i in 0..(LOG_CAPACITY + 50) {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.len(), LOG_CAPACITY);
        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[2], format!("line {}", LOG_CAPACITY + 49));
        // Asking for more than we have is fine.
        assert_eq!(log.recent(10_000).len(), LOG_CAPACITY);
    }
}
