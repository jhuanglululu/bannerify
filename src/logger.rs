//! Coloured `info`/`error` output, ported from `../bannerify-old/src/logger`.
//!
//! [`error_out!`] is the friendly-validation-failure path: print one clear line
//! and exit non-zero, instead of a clap panic or a raw `unwrap`.

use colored::Colorize;

/// Print an `info: ...` line to stdout.
macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::logger::info_print(format!($($arg)*));
    }};
}

/// Print an `error: ...` line to stderr and exit with status 1.
macro_rules! error_out {
    ($($arg:tt)*) => {{
        $crate::logger::error_print(format!($($arg)*));
        std::process::exit(1);
    }};
}

pub fn info_print(message: String) {
    println!("{}: {}", "info".green().bold(), message);
}

pub fn error_print(message: String) {
    eprintln!("{}: {}", "error".red().bold(), message);
}

pub(crate) use {error_out, info};
