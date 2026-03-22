use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};

use colored::Colorize;

pub struct Logger {}

pub const LOGGER: Logger = Logger {};

impl Logger {
    pub fn info(&self, message: impl AsRef<str>) {
        println!("{}: {}", "info".green().bold(), message.as_ref());
    }

    pub fn error(&self, message: impl AsRef<str>) {
        eprintln!("{}: {}", "error".red().bold(), message.as_ref());
    }
}

macro_rules! info {
    ($($arg:tt)*) => {{
        use $crate::logger::LOGGER;
        LOGGER.info(format!($($arg)*));
    }};
}

macro_rules! error {
    ($($arg:tt)*) => {{
        use $crate::logger::LOGGER;
        LOGGER.error(format!($($arg)*));
    }};
}

pub(crate) use {error, info};

static TIMER: LazyLock<Instant> = LazyLock::new(Instant::now);
static LAST: RwLock<Option<Instant>> = RwLock::new(None);

pub fn init_clock() {
    LazyLock::force(&TIMER);
    *LAST.write().unwrap() = Some(Instant::now());
}

pub fn elapsed() -> Duration {
    TIMER.elapsed()
}

pub fn since_last() -> Duration {
    let now = Instant::now();
    let prev = LAST.write().unwrap().replace(now).unwrap();
    now.duration_since(prev)
}
