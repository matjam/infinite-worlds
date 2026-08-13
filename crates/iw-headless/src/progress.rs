//! [`ProgressSink`] that logs to stderr via the `log` crate, throttling the
//! high-frequency `Step` event so a long run doesn't flood the terminal.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use iw_core::{ProgressEvent, ProgressSink};

/// Minimum wall time between printed `Step` lines.
const STEP_PRINT_INTERVAL: Duration = Duration::from_secs(2);

/// Prints phase/narration/milestone events immediately; `Step` events at most
/// once every [`STEP_PRINT_INTERVAL`].
pub struct CliProgress {
    last_step_print: Mutex<Instant>,
}

impl Default for CliProgress {
    fn default() -> Self {
        CliProgress {
            // Ensure the very first Step event prints immediately.
            last_step_print: Mutex::new(Instant::now() - STEP_PRINT_INTERVAL),
        }
    }
}

impl ProgressSink for CliProgress {
    fn event(&self, ev: ProgressEvent) {
        match ev {
            ProgressEvent::PhaseStarted { phase, time_myr } => {
                log::info!("=== phase {phase:?} started at {time_myr:.3} Myr ===");
            }
            ProgressEvent::PhaseCompleted { phase, time_myr } => {
                log::info!("=== phase {phase:?} completed at {time_myr:.3} Myr ===");
            }
            ProgressEvent::Narration(line) => log::info!("{line}"),
            ProgressEvent::Milestone(line) => log::info!("milestone: {line}"),
            ProgressEvent::Step {
                phase,
                step,
                of,
                time_myr,
            } => {
                let mut last = self.last_step_print.lock().unwrap();
                if last.elapsed() >= STEP_PRINT_INTERVAL {
                    log::info!("{phase:?} step {step}/{of} ({time_myr:.3} Myr)");
                    *last = Instant::now();
                }
            }
        }
    }
}
