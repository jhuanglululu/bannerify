use colored::Colorize;

pub mod profiler;

macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::logger::info_print(format!($($arg)*));
    }};
}

pub fn info_print(message: String) {
    println!("{:}: {}", "info".green().bold(), message);
}

macro_rules! error {
    ($($arg:tt)*) => {{
        $crate::logger::error_print(format!($($arg)*));
    }};
}

pub fn error_print(message: String) {
    eprintln!("{:}: {}", "error".red().bold(), message);
}

pub fn profile_print(message: String) {
    println!("{:}: {}", "profiling".blue().bold(), message);
}

pub(crate) use {error, info};
