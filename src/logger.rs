//! Coloured `info`/`error` output.

use colored::Colorize;

macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::logger::info_print(format!($($arg)*));
    }};
}

/// Prints an `error: ...` line to stderr and exits with status 1.
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
