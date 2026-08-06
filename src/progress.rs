//! A single-line stderr progress bar for the solve.
//!
//! The rayon workers never touch the terminal: they bump one shared
//! `AtomicUsize` per solved banner, and a watcher thread redraws the line on a
//! 100 ms tick — one writer, no per-cell I/O cost, no interleaved output. The
//! bar only exists when stderr is a terminal, so piped and CI runs stay clean.
//!
//! [`Progress::finish`] joins the watcher and erases the line before the
//! summary `info!` lines print, so a half-drawn bar never collides with them.

use std::io::{stderr, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

const BAR_WIDTH: usize = 24;
const TICK: Duration = Duration::from_millis(100);

pub struct Progress {
    count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    watcher: Option<JoinHandle<()>>,
}

impl Progress {
    /// Start counting towards `total` banners; the watcher thread (and any
    /// drawing at all) exists only when stderr is a terminal.
    pub fn start(total: usize) -> Self {
        let count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let watcher = (total > 0 && stderr().is_terminal()).then(|| {
            let count = Arc::clone(&count);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut last = usize::MAX;
                while !stop.load(Ordering::Relaxed) {
                    let n = count.load(Ordering::Relaxed).min(total);
                    if n != last {
                        last = n;
                        draw(n, total);
                    }
                    std::thread::sleep(TICK);
                }
                erase();
            })
        });
        Progress {
            count,
            stop,
            watcher,
        }
    }

    /// The counter a worker bumps once per solved banner.
    pub fn counter(&self) -> &AtomicUsize {
        &self.count
    }

    /// Stop the watcher and erase the line. Call before printing anything.
    pub fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(watcher) = self.watcher.take() {
            let _ = watcher.join();
        }
    }
}

fn draw(n: usize, total: usize) {
    let filled = n * BAR_WIDTH / total;
    eprint!(
        "\r[{}{}] {}/{} banners",
        "#".repeat(filled),
        "-".repeat(BAR_WIDTH - filled),
        n,
        total,
    );
    let _ = stderr().flush();
}

fn erase() {
    // The line's worst-case width: the bar plus two full-width counters.
    let width = BAR_WIDTH + 2 + 2 * 20 + " banners/".len();
    eprint!("\r{}\r", " ".repeat(width));
    let _ = stderr().flush();
}
