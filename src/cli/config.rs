//! Merging CLI flags with the TOML config file, and validating the result.
//!
//! CLI values always win over file values. Validation failures print one
//! friendly coloured line and exit (see
//! [`error_out!`](crate::logger::error_out)).

use std::collections::HashSet;
use std::path::PathBuf;

use crate::layout::{Dimension, ResizingMethod};
use colored::Colorize;
use serde::Deserialize;

use crate::cli::Args;
use crate::logger::error_out;

const DEFAULT_EXACT_CANDIDATES: usize = 20;

/// The validated, merged configuration the pipeline runs on.
pub struct Config {
    pub input: PathBuf,
    pub output: PathBuf,
    pub dimension: Dimension,
    pub workers: Option<usize>,
    pub resizing_method: ResizingMethod,
    pub debug: bool,
    pub exclude_patterns: HashSet<String>,
    pub exclude_blocks: HashSet<String>,
    /// Largest dimension of the embedded preview images, or `None` for the
    /// input image's own size. Clamped to the wall either way
    /// ([`crate::preview::dimensions`]).
    pub preview: Option<usize>,
    pub render: Option<PathBuf>,
    /// `(min, max)` layers per banner, spread by the variance pre-pass.
    pub n_layers: (usize, usize),
    pub refinement: RefinementConfig,
    /// `(top_n, duplicates, rounds)` perturbation search, or `None` when off.
    pub perturbations: Option<(usize, usize, usize)>,
    pub seed: u64,
}

/// Windowed beam refinement settings.
pub struct RefinementConfig {
    pub refinement_pass: usize,
    /// Layers adjusted at once.
    pub window_size: usize,
    /// Candidate survival threshold, `0.0..=1.0`.
    pub error_threshold: f32,
    pub refinement_candidate: usize,
    /// Candidates per window step re-scored exactly in OKLab; `0` disables the
    /// exact rung and leaves refinement on the closed-form sRGB error.
    pub exact_candidates: usize,
}

/// The TOML config file schema (all keys optional).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConfigToml {
    pub workers: Option<usize>,
    pub fit: Option<bool>,
    pub stretch: Option<bool>,
    pub fill: Option<String>,
    pub exclude_patterns: Option<Vec<String>>,
    pub exclude_blocks: Option<Vec<String>>,
    pub layer_range: Option<Vec<usize>>,
    pub refinement_pass: Option<usize>,
    pub window_size: Option<usize>,
    pub error_threshold: Option<f32>,
    pub refinement_candidate: Option<usize>,
    pub exact_candidates: Option<usize>,
    pub perturbations: Option<Vec<usize>>,
    pub seed: Option<u64>,
    pub preview: Option<usize>,
    pub render: Option<PathBuf>,
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let (config, config_path) = load_config(args.config.as_deref());
        let n_layers = parse_n_layers(&config_path, &args.layer_range, config.layer_range);

        Config {
            input: validate_input(args.input),
            output: args.output,
            dimension: parse_dimension(args.row, args.columns),
            workers: args.workers.or(config.workers),
            resizing_method: parse_resizing_method(
                args.fit,
                args.stretch,
                args.fill.as_deref(),
                (config.fit, config.stretch, config.fill.as_deref()),
            ),
            debug: args.debug,
            exclude_patterns: HashSet::from_iter(
                args.exclude_patterns
                    .map(|pat| pat.split(',').map(str::to_string).collect())
                    .or(config.exclude_patterns)
                    .unwrap_or_default(),
            ),
            exclude_blocks: HashSet::from_iter(
                args.exclude_blocks
                    .map(|pat| pat.split(',').map(str::to_string).collect())
                    .or(config.exclude_blocks)
                    .unwrap_or_default(),
            ),
            preview: parse_preview(args.preview.or(config.preview)),
            render: args.render.or(config.render),
            n_layers,
            refinement: RefinementConfig {
                refinement_pass: args.refinement_pass.or(config.refinement_pass).unwrap_or(2),
                window_size: parse_window_size(
                    &config_path,
                    n_layers.0,
                    args.window_size,
                    config.window_size,
                ),
                error_threshold: parse_error_threshold(
                    &config_path,
                    args.error_threshold,
                    config.error_threshold,
                ),
                refinement_candidate: args
                    .refinement_candidate
                    .or(config.refinement_candidate)
                    .unwrap_or(5),
                exact_candidates: args
                    .exact_candidates
                    .or(config.exact_candidates)
                    .unwrap_or(DEFAULT_EXACT_CANDIDATES),
            },
            perturbations: parse_perturbation(
                &config_path,
                &args.perturbations,
                config.perturbations,
            ),
            seed: args.seed.or(config.seed).unwrap_or(0),
        }
    }
}

fn load_config(path: Option<&std::path::Path>) -> (ConfigToml, String) {
    let Some(path) = path else {
        return (ConfigToml::default(), String::new());
    };
    let shown = path.display().to_string();

    if !path.exists() {
        error_out!("'{}' does not exists", shown.yellow());
    }

    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        error_out!(
            "error reading '{}': {}",
            shown.yellow(),
            e.to_string().red()
        );
    });

    let config = toml::from_str::<ConfigToml>(&text).unwrap_or_else(|e| {
        error_out!(
            "error parsing '{}': {}",
            shown.yellow(),
            e.to_string().red()
        );
    });

    (config, shown)
}

/// `--preview 0` is meaningless — a zero-pixel image — and almost certainly a
/// typo, so it is refused rather than silently clamped to one pixel.
fn parse_preview(preview: Option<usize>) -> Option<usize> {
    if preview == Some(0) {
        error_out!(
            "'{}' needs to be at least '{}'",
            "--preview".yellow(),
            "1".yellow()
        );
    }
    preview
}

fn validate_input(input: PathBuf) -> PathBuf {
    if !input.exists() {
        error_out!("'{}' does not exists", input.display().to_string().yellow());
    }
    input
}

fn parse_dimension(row: Option<usize>, col: Option<usize>) -> Dimension {
    match (row, col) {
        (Some(r), None) => {
            if r < 1 {
                error_out!(
                    "'{}' needs to be at least '{}'",
                    "--row".yellow(),
                    "1".yellow()
                );
            }
            Dimension::Row(r)
        }
        (None, Some(c)) => {
            if c < 1 {
                error_out!(
                    "'{}' needs to be at least '{}'",
                    "--columns".yellow(),
                    "1".yellow()
                );
            }
            Dimension::Column(c)
        }
        (Some(_), Some(_)) => {
            error_out!(
                "only one of '{}' or '{}' can be entered",
                "--row".yellow(),
                "--columns".yellow()
            );
        }
        (None, None) => {
            error_out!(
                "one of '{}' or '{}' is required",
                "--row".yellow(),
                "--columns".yellow()
            );
        }
    }
}

fn parse_resizing_method(
    fit: bool,
    stretch: bool,
    fill: Option<&str>,
    config_settings: (Option<bool>, Option<bool>, Option<&str>),
) -> ResizingMethod {
    match (fit, stretch, fill) {
        (true, false, None) => ResizingMethod::Fit,
        (false, true, None) => ResizingMethod::Stretch,
        (false, false, Some(color_str)) => match parse_color(color_str) {
            Ok(color) => ResizingMethod::Fill(color),
            Err(e) => error_out!("{}", e),
        },
        (false, false, None) => match config_settings {
            (Some(true), Some(false) | None, None) => ResizingMethod::Fit,
            (Some(false) | None, Some(true), None) => ResizingMethod::Stretch,
            (Some(false) | None, Some(false) | None, Some(color_str)) => {
                match parse_color(color_str) {
                    Ok(color) => ResizingMethod::Fill(color),
                    Err(e) => error_out!("{}", e),
                }
            }
            (Some(false) | None, Some(false) | None, None) => ResizingMethod::Fit,
            _ => {
                error_out!(
                    "only one of '{}', '{}' or '{}' can exist in config",
                    "fit".yellow(),
                    "stretch".yellow(),
                    "fill".yellow()
                );
            }
        },
        _ => {
            error_out!(
                "only one of '{}', '{}' or '{}' can be entered",
                "--fit".yellow(),
                "--stretch".yellow(),
                "--fill".yellow()
            );
        }
    }
}

fn parse_n_layers(
    config_path: &str,
    layers_vec: &[usize],
    config_vec: Option<Vec<usize>>,
) -> (usize, usize) {
    let layers = if !layers_vec.is_empty() {
        (layers_vec[0], layers_vec[1])
    } else if let Some(ref layer_range) = config_vec {
        if layer_range.len() != 2 {
            error_out!(
                "'{}' in '{}' can only have two elements: '{}'",
                "layer_range".yellow(),
                config_path.yellow(),
                "[MIN, MAX]".yellow(),
            );
        }
        (layer_range[0], layer_range[1])
    } else {
        (4, 6)
    };

    if layers.0 > layers.1 {
        error_out!(
            "'{}' can not be greater than '{}' in '{}'",
            "MIN".yellow(),
            "MAX".yellow(),
            "layer_range".yellow(),
        );
    }

    if layers.0 < 1 {
        error_out!(
            "'{}' can not be less than '{}' in '{}'",
            "MIN".yellow(),
            "1".yellow(),
            "layer_range".yellow(),
        );
    }

    layers
}

fn parse_window_size(
    config_path: &str,
    min_layer: usize,
    window_size: Option<usize>,
    config: Option<usize>,
) -> usize {
    if let Some(k) = window_size {
        if k < 1 {
            error_out!(
                "'{}' value need to be greater than '{}'",
                "--window-size".yellow(),
                "1".yellow(),
            );
        }
        if min_layer < k {
            error_out!(
                "'{}' value need to be less than '{}': '{}'",
                "--window-size".yellow(),
                "MIN-LAYERS".yellow(),
                min_layer.to_string().yellow(),
            );
        }
        k
    } else if let Some(k) = config {
        if k < 1 {
            error_out!(
                "'{}' in '{}' needs to be greater '{}'",
                "window_size".yellow(),
                config_path.yellow(),
                "1".yellow(),
            );
        }
        if min_layer < k {
            error_out!(
                "'{}' in '{}' needs to be less than '{}': '{}'",
                "window_size".yellow(),
                config_path.yellow(),
                "MIN-LAYERS".yellow(),
                min_layer.to_string().yellow(),
            );
        }
        k
    } else {
        2
    }
}

fn parse_error_threshold(config_path: &str, threshold: Option<f32>, config: Option<f32>) -> f32 {
    if let Some(thresh) = threshold {
        if !(0.0..=1.0).contains(&thresh) {
            error_out!(
                "'{}' value need to be within '{}' and '{}'",
                "--error-threshold".yellow(),
                "0.0".yellow(),
                "1.0".yellow()
            );
        }
        thresh
    } else if let Some(thresh) = config {
        if !(0.0..=1.0).contains(&thresh) {
            error_out!(
                "'{}' in '{}' needs to be within '{}' and '{}'",
                "error_threshold".yellow(),
                config_path.yellow(),
                "0.0".yellow(),
                "1.0".yellow()
            );
        }
        thresh
    } else {
        0.7
    }
}

fn parse_perturbation(
    config_path: &str,
    args: &[usize],
    config: Option<Vec<usize>>,
) -> Option<(usize, usize, usize)> {
    if !args.is_empty() {
        if args.contains(&0) {
            None
        } else {
            Some((args[0], args[1], args[2]))
        }
    } else if let Some(config_vec) = config {
        if config_vec.len() != 3 {
            error_out!(
                "'{}' in '{}' can only have '{}' elements: '{}'",
                "perturbations".yellow(),
                config_path.yellow(),
                "three".yellow(),
                "[TOP_N, DUPLICATES, ROUNDS]".yellow(),
            );
        }
        if config_vec.contains(&0) {
            None
        } else {
            Some((config_vec[0], config_vec[1], config_vec[2]))
        }
    } else {
        None
    }
}

fn valid_color_str() -> String {
    format!(
        "\n       valid color format includes: '{}', '{}' and '{}'",
        "#ff9453".yellow(),
        "9,4,87".yellow(),
        "rgb(11, 45, 14)".yellow()
    )
}

/// Parse `#rrggbb`, `rrggbb`, `r,g,b` or `rgb(r, g, b)`.
pub fn parse_color(s: &str) -> Result<[u8; 3], String> {
    if let Some(hex) = s
        .strip_prefix('#')
        .or_else(|| (!s.contains(',')).then_some(s))
    {
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "invalid hex color: '{}'. {}",
                s.yellow(),
                valid_color_str()
            ));
        }
        let r = u8::from_str_radix(&hex[0..2], 16).expect("checked hex digits");
        let g = u8::from_str_radix(&hex[2..4], 16).expect("checked hex digits");
        let b = u8::from_str_radix(&hex[4..6], 16).expect("checked hex digits");
        return Ok([r, g, b]);
    }

    let inner = s
        .strip_prefix("rgb(")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(s);

    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected 3 components, got {} from '{}'. {}",
            parts.len().to_string().yellow(),
            s.yellow(),
            valid_color_str()
        ));
    }

    Ok([
        parse_component(parts[0], s)?,
        parse_component(parts[1], s)?,
        parse_component(parts[2], s)?,
    ])
}

fn parse_component(p: &str, s: &str) -> Result<u8, String> {
    p.parse::<u8>().map_err(|_| {
        format!(
            "invalid color component '{}' in '{}'. {}",
            p.yellow(),
            s.yellow(),
            valid_color_str()
        )
    })
}
