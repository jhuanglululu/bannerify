//! Everything the CLI says, and which stream it goes to.
//!
//! Lines for people (`info`, `warning`, `debug`, `error`) go to stderr. stdout
//! carries only results: the single `--json` object or a `--list` listing, so
//! a caller can parse stdout without filtering log lines out of it.
//!
//! A problem is a [`Diagnostic`]: a [`Kind`] (which decides the exit code), a
//! message, and the option, value or path at fault. The same diagnostic prints
//! as an `error:` line in text mode and as an `errors[]` entry in `--json`
//! mode.

use std::fmt::Display;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

use colored::Colorize;
use serde::Serialize;
use serde_json::{Value, json};

/// Version of both JSON shapes: the `--json` result object and the
/// `--format json` artifact. Bumped on any breaking change to either.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// Human output on stderr.
    Text = 0,
    /// Text without the `info:` lines.
    Quiet = 1,
    /// One JSON object on stdout, nothing on stderr but the TTY progress bar.
    Json = 2,
}

static MODE: AtomicU8 = AtomicU8::new(Mode::Text as u8);
static WARNINGS: Mutex<Vec<Diagnostic>> = Mutex::new(Vec::new());

pub fn set_mode(mode: Mode) {
    MODE.store(mode as u8, Ordering::Relaxed);
}

pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        0 => Mode::Text,
        1 => Mode::Quiet,
        _ => Mode::Json,
    }
}

/// What went wrong, as a caller would branch on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The command line itself could not be parsed (clap).
    Usage,
    /// A flag or config value is out of range or malformed.
    InvalidValue,
    /// Mutually exclusive options were given together.
    Conflict,
    /// A required option is absent.
    Missing,
    /// An excluded pattern or block id names nothing.
    UnknownId,
    /// The config file is missing, unreadable or not valid TOML.
    ConfigError,
    /// The input image is missing, unreadable or not an image.
    InputError,
    /// The output or `--render` file could not be written.
    WriteError,
    /// An AVX2 build on a CPU without AVX2+FMA.
    UnsupportedCpu,
    /// A bug.
    Internal,
    /// Warning: an option was accepted but changes nothing.
    NoEffect,
}

impl Kind {
    pub fn exit_code(self) -> i32 {
        match self {
            Kind::Usage
            | Kind::InvalidValue
            | Kind::Conflict
            | Kind::Missing
            | Kind::UnknownId
            | Kind::ConfigError => 2,
            Kind::InputError => 3,
            Kind::WriteError => 4,
            Kind::UnsupportedCpu => 5,
            Kind::Internal | Kind::NoEffect => 1,
        }
    }
}

/// One error or warning. Optional fields are left out of the JSON when unset.
#[derive(Clone, Debug, Serialize)]
pub struct Diagnostic {
    pub kind: Kind,
    pub message: String,
    /// The CLI flag at fault, e.g. `--row`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub option: Option<String>,
    /// The config-file key at fault, e.g. `window_size`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// The config file the key was read from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Valid values close to a mistyped one.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl Diagnostic {
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Diagnostic {
            kind,
            message: message.into(),
            option: None,
            key: None,
            config: None,
            value: None,
            path: None,
            suggestions: Vec::new(),
            hint: None,
        }
    }

    pub fn option(mut self, option: impl Into<String>) -> Self {
        self.option = Some(option.into());
        self
    }

    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    pub fn config(mut self, config: impl Into<String>) -> Self {
        self.config = Some(config.into());
        self
    }

    pub fn value(mut self, value: impl Display) -> Self {
        self.value = Some(value.to_string());
        self
    }

    pub fn path(mut self, path: impl Display) -> Self {
        self.path = Some(path.to_string());
        self
    }

    pub fn suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions = suggestions;
        self
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// The one-line text form: message, then suggestions, then the hint.
    pub fn text(&self) -> String {
        let mut out = self.message.clone();
        if !self.suggestions.is_empty() {
            let quoted: Vec<String> = self.suggestions.iter().map(|s| format!("'{s}'")).collect();
            out.push_str(&format!(" (did you mean {}?)", quoted.join(" or ")));
        }
        if let Some(hint) = &self.hint {
            out.push_str("; ");
            out.push_str(hint);
        }
        out
    }
}

macro_rules! info {
    ($($arg:tt)*) => {{
        $crate::logger::info_print(format!($($arg)*));
    }};
}

pub fn info_print(message: String) {
    if mode() == Mode::Text {
        eprintln!("{}: {}", "info".green().bold(), message);
    }
}

/// A `debug:` line; `--debug` asked for it, so `--quiet` keeps it. In JSON mode
/// the timings go into the result object instead.
pub fn debug_print(message: String) {
    if mode() != Mode::Json {
        eprintln!("{}: {}", "debug".blue().bold(), message);
    }
}

/// Print a warning (text modes) and keep it for the JSON `warnings[]`.
pub fn warn(warning: Diagnostic) {
    if mode() != Mode::Json {
        eprintln!("{}: {}", "warning".yellow().bold(), warning.text());
    }
    WARNINGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(warning);
}

/// Every warning issued so far.
pub fn warnings() -> Vec<Diagnostic> {
    WARNINGS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Print the one `--json` result object on stdout.
pub fn print_result(result: &Value) {
    println!("{result}");
}

/// Report one error and exit with its kind's code.
pub fn fatal(error: Diagnostic) -> ! {
    fail(vec![error])
}

/// Report every error and exit with the first one's code.
pub fn fail(errors: Vec<Diagnostic>) -> ! {
    let code = errors.first().map_or(1, |e| e.kind.exit_code());
    if mode() == Mode::Json {
        print_result(&json!({
            "ok": false,
            "schema_version": SCHEMA_VERSION,
            "errors": errors,
            "warnings": warnings(),
        }));
    } else {
        for error in &errors {
            eprintln!("{}: {}", "error".red().bold(), error.text());
        }
    }
    std::process::exit(code);
}

/// A bug, not a user mistake.
pub fn internal(message: impl Display) -> ! {
    fatal(Diagnostic::new(
        Kind::Internal,
        format!("internal error: {message}"),
    ))
}

pub(crate) use info;
