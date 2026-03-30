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

macro_rules! error_out {
    ($($arg:tt)*) => {{
        $crate::logger::error_print(format!($($arg)*));
        std::process::exit(1);
    }};
}

pub fn error_print(message: String) {
    eprintln!("{:}: {}", "error".red().bold(), message);
}

pub(crate) use {error, error_out, info};
