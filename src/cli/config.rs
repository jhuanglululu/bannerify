use crate::cli::Args;
use crate::color::NUM_COLORS;
use crate::logger::error_out;
use clap::ValueEnum;
use colored::Colorize;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::LazyLock;

pub struct Config {
    pub input: PathBuf,
    pub output: PathBuf,
    pub dimension: Dimension,
    pub workers: Option<usize>,
    pub resizing_method: ResizingMethod,
    pub exclude_patterns: HashSet<String>,
    pub exclude_blocks: HashSet<String>,
    pub complexity: ComplexityOptions,
    pub optimal: OptimalOption,
    pub perturbations: Option<(usize, usize, usize)>,
    pub lab_refine: Option<usize>,
}

#[derive(Deserialize)]
pub struct ConfigToml {
    pub workers: Option<usize>,
    pub fit: Option<bool>,
    pub stretch: Option<bool>,
    pub fill: Option<String>,
    pub exclude_patterns: Option<Vec<String>>,
    pub exclude_blocks: Option<Vec<String>>,
    pub layer_range: Option<Vec<usize>>,
    pub color_range: Option<Vec<usize>>,
    pub optimal: Option<Vec<OptimalEnum>>,
    pub perturbations: Option<Vec<usize>>,
    pub lab_refine: Option<usize>,
}

#[derive(Clone, Copy)]
pub enum Dimension {
    Row(usize),
    Column(usize),
}

#[derive(Clone, Copy)]
pub struct ComplexityOptions {
    pub layers: (usize, usize),
    pub colors: (usize, usize),
}

#[derive(Clone, Copy)]
pub enum ResizingMethod {
    Fit,
    Fill([u8; 3]),
    Stretch,
}

#[derive(Debug, Clone, PartialEq, ValueEnum, Deserialize)]
pub enum OptimalEnum {
    Initial,
    Refine,
    Perturbation,
    LabRefine,
}

#[derive(Clone, Copy)]
pub struct OptimalOption {
    pub initial: bool,
    pub refine: bool,
    pub perturbation: bool,
    pub lab_refine: bool,
}

impl OptimalOption {
    pub fn has_greedy(&self) -> bool {
        !(self.initial && self.refine && self.perturbation && self.lab_refine)
    }
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        if let Some(ref config_path) = args.config {
            if !config_path.exists() {
                error_out!(
                    "'{}' does not exists",
                    config_path.display().to_string().yellow()
                );
            }

            let mut config_buf = Vec::new();
            File::open(config_path)
                .unwrap_or_else(|e| {
                    error_out!(
                        "error opening '{}': {}",
                        config_path.display().to_string().yellow(),
                        e.to_string().red()
                    );
                })
                .read_to_end(&mut config_buf)
                .unwrap_or_else(|e| {
                    error_out!(
                        "error reading '{}': {}",
                        config_path.display().to_string().yellow(),
                        e.to_string().red()
                    );
                });

            let mut config = match toml::from_slice::<ConfigToml>(&config_buf) {
                Ok(table) => table,
                Err(e) => {
                    error_out!(
                        "error parsing '{}': {}",
                        config_path.display().to_string().yellow(),
                        e.to_string().red()
                    );
                }
            };

            Config {
                input: validate_input(args.input),
                output: args.output,
                dimension: parse_dimension(args.row, args.columns),
                workers: args.workers.or(config.workers),
                resizing_method: parse_resizing_method(
                    args.fit,
                    args.stretch,
                    args.fill.as_deref(),
                    Some(&config),
                ),
                exclude_patterns: HashSet::from_iter(
                    args.exclude_patterns
                        .map(|pat| pat.split(',').map(|p| p.to_string()).collect())
                        .or(config.exclude_patterns.take())
                        .unwrap_or_default(),
                ),
                exclude_blocks: HashSet::from_iter(
                    args.exclude_blocks
                        .map(|pat| pat.split(',').map(|p| p.to_string()).collect())
                        .or(config.exclude_blocks.take())
                        .unwrap_or_default(),
                ),
                complexity: parse_complextiy(
                    Some(&config_path.display().to_string()),
                    &args.layer_range,
                    &args.color_range,
                    Some(&config),
                ),
                optimal: parse_optimal(&args.optimal, config.optimal.as_deref()),
                perturbations: parse_perturbation(
                    Some(&config_path.display().to_string()),
                    &args.perturbations,
                    config.perturbations,
                ),
                lab_refine: args.lab_refine.or(config.lab_refine),
            }
        } else {
            Config {
                input: validate_input(args.input),
                output: args.output,
                dimension: parse_dimension(args.row, args.columns),
                workers: args.workers,
                resizing_method: parse_resizing_method(
                    args.fit,
                    args.stretch,
                    args.fill.as_deref(),
                    None,
                ),
                exclude_patterns: args
                    .exclude_patterns
                    .map(|pat| {
                        pat.rsplit(',')
                            .map(|p| p.to_string())
                            .collect::<HashSet<String>>()
                    })
                    .unwrap_or_default(),
                exclude_blocks: args
                    .exclude_blocks
                    .map(|pat| {
                        pat.rsplit(',')
                            .map(|p| p.to_string())
                            .collect::<HashSet<String>>()
                    })
                    .unwrap_or_default(),
                complexity: parse_complextiy(None, &args.layer_range, &args.color_range, None),
                optimal: parse_optimal(&args.optimal, None),
                perturbations: parse_perturbation(None, &args.perturbations, None),
                lab_refine: args.lab_refine,
            }
        }
    }
}

fn validate_input(input: PathBuf) -> PathBuf {
    if !input.exists() {
        error_out!("'{}' does not exists", input.display().to_string().yellow(),);
    }
    input
}

fn parse_dimension(row: Option<usize>, col: Option<usize>) -> Dimension {
    match (row, col) {
        (Some(r), None) => Dimension::Row(r),
        (None, Some(c)) => Dimension::Column(c),
        (Some(_), Some(_)) => {
            error_out!(
                "only one of '{}' or '{}' can be entered",
                "--rows".yellow(),
                "--columns".yellow()
            );
        }
        (None, None) => {
            error_out!(
                "one of '{}' or '{}' is required",
                "--rows".yellow(),
                "--columns".yellow()
            );
        }
    }
}

fn parse_resizing_method(
    fit: bool,
    stretch: bool,
    fill: Option<&str>,
    config_toml: Option<&ConfigToml>,
) -> ResizingMethod {
    match (fit, stretch, fill) {
        (true, false, None) => ResizingMethod::Fit,
        (false, true, None) => ResizingMethod::Stretch,
        (false, false, Some(color_str)) => match parse_color(color_str) {
            Ok(color) => ResizingMethod::Fill(color),
            Err(e) => {
                error_out!("{}", e);
            }
        },
        (false, false, None) => {
            if let Some(config) = config_toml {
                match (config.fit, config.stretch, &config.fill) {
                    (Some(true), Some(false) | None, None) => ResizingMethod::Fit,
                    (Some(false) | None, Some(true), None) => ResizingMethod::Stretch,
                    (Some(false) | None, Some(false) | None, Some(color_str)) => {
                        match parse_color(color_str) {
                            Ok(color) => ResizingMethod::Fill(color),
                            Err(e) => {
                                error_out!("{}", e);
                            }
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
                }
            } else {
                ResizingMethod::Fit
            }
        }
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

fn parse_complextiy(
    config_name: Option<&str>,
    layers_vec: &[usize],
    colors_vec: &[usize],
    config_toml: Option<&ConfigToml>,
) -> ComplexityOptions {
    let layers = if !layers_vec.is_empty() {
        (layers_vec[0], layers_vec[1])
    } else if let Some(config) = config_toml
        && let Some(ref layer_range) = config.layer_range
    {
        if layer_range.len() != 2 {
            error_out!(
                "'{}' in '{}' can only have two elements: '{}'",
                "layer_range".yellow(),
                config_name.unwrap().yellow(),
                "[MIN, MAX]".yellow(),
            );
        }
        (layer_range[0], layer_range[1])
    } else {
        (4, 6)
    };

    let colors = if !colors_vec.is_empty() {
        (colors_vec[0], colors_vec[1])
    } else if let Some(config) = config_toml
        && let Some(ref color_range) = config.color_range
    {
        if color_range.len() != 2 {
            error_out!(
                "'{}' in '{}' can only have '{}' elements: '{}'",
                "color_range".yellow(),
                config_name.unwrap().yellow(),
                "two".yellow(),
                "[MIN, MAX]".yellow(),
            );
        }
        (color_range[0], color_range[1])
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

    if colors.0 > colors.1 {
        error_out!(
            "'{}' can not be greater than '{}' in '{}'",
            "MIN".yellow(),
            "MAX".yellow(),
            "color_range".yellow(),
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

    if colors.0 < 1 {
        error_out!(
            "'{}' can not be less than '{}' in '{}'",
            "MIN".yellow(),
            "1".yellow(),
            "color_range".yellow(),
        );
    }

    if colors.1 > NUM_COLORS {
        error_out!(
            "'{}' can not be greater than '{}' in '{}'",
            "MIN".yellow(),
            NUM_COLORS.to_string().yellow(),
            "color_range".yellow(),
        );
    }

    ComplexityOptions { layers, colors }
}

fn parse_optimal(args: &[OptimalEnum], config: Option<&[OptimalEnum]>) -> OptimalOption {
    if !args.is_empty() {
        OptimalOption {
            initial: args.contains(&OptimalEnum::Initial),
            refine: args.contains(&OptimalEnum::Refine),
            perturbation: args.contains(&OptimalEnum::Perturbation),
            lab_refine: args.contains(&OptimalEnum::LabRefine),
        }
    } else if let Some(optimal_vec) = config {
        OptimalOption {
            initial: optimal_vec.contains(&OptimalEnum::Initial),
            refine: optimal_vec.contains(&OptimalEnum::Refine),
            perturbation: optimal_vec.contains(&OptimalEnum::Perturbation),
            lab_refine: optimal_vec.contains(&OptimalEnum::LabRefine),
        }
    } else {
        OptimalOption {
            initial: false,
            refine: false,
            perturbation: false,
            lab_refine: false,
        }
    }
}

fn parse_perturbation(
    config_name: Option<&str>,
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
                "perturbation".yellow(),
                config_name.unwrap().yellow(),
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

static VALID_COLOR_STR: LazyLock<String> = LazyLock::new(|| {
    format!(
        "\n       valid color format includes: '{}', '{}' and '{}'",
        "#ff9453".yellow(),
        "9,4,87".yellow(),
        "rgb(11, 45, 14)".yellow()
    )
});

fn parse_color(s: &str) -> Result<[u8; 3], String> {
    // "#rrggbb" or "rrggbb"

    if let Some(hex) = s
        .strip_prefix('#')
        .or_else(|| (!s.contains(',')).then_some(s))
    {
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "invalid hex color: '{}'. {}",
                s.yellow(),
                *VALID_COLOR_STR
            ));
        }
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap();
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap();
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap();
        return Ok([r, g, b]);
    }

    // "rgb(r, g, b)" or "r,g,b"
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
            *VALID_COLOR_STR
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
            *VALID_COLOR_STR
        )
    })
}
