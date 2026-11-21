//! Merging CLI flags with the TOML config file, and validating the result.
//!
//! CLI values always win over file values. Validation collects every problem
//! before giving up, so one run reports all of them (see
//! [`fail`](crate::logger::fail)); each problem names the flag or the config
//! key it came from, with the same wording for both.

use std::collections::HashSet;
use std::fmt::Display;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::block;
use crate::cli::{Args, BackendChoice, OutputFormat};
use crate::layout::{Dimension, ResizingMethod};
use crate::logger::{Diagnostic, Kind, fail, warn};
use crate::pattern;
use crate::solver::feature::LAYER_COUNTS;
use crate::solver::shift::MAX_SHIFT;

const DEFAULT_EXACT_CANDIDATES: usize = 20;

/// Provisional: picked before the A/B of `context/plans/6-feature-map.md`
/// step 6 has been run.
const DEFAULT_FEATURE_WEIGHT: f32 = 0.5;

const DEFAULT_PURITY_WEIGHT: f32 = 0.0;

const DEFAULT_FEATURE_LAYERS: usize = 3;

const DEFAULT_LAYERS: (usize, usize) = (4, 6);

/// Default GPU memory budget, in MiB: the GPU backend's own default.
const DEFAULT_GPU_MEMORY: usize = 2048;

/// The validated, merged configuration the pipeline runs on.
pub struct Config {
    pub input: PathBuf,
    pub output: PathBuf,
    pub format: OutputFormat,
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
    /// Weight of the feature-map term in the solver's error, `0` to disable it
    /// entirely ([`crate::solver::feature`]).
    pub feature_weight: f32,
    /// Weight of the purity term in the exact rung's ranking error, `0` to
    /// disable it ([`crate::solver::refine`]).
    pub purity_weight: f32,
    /// Most layers of the feature map, one of [`LAYER_COUNTS`]
    /// ([`crate::solver::feature`]).
    pub feature_layers: usize,
    /// `(X, Y)`: how far each banner's target patch may be displaced, in wall
    /// pixels; `(0, 0)` disables the shift ([`crate::solver::shift`]).
    pub shift_range: (usize, usize),
    /// `(top_n, duplicates, rounds)` perturbation search, or `None` when off.
    pub perturbations: Option<(usize, usize, usize)>,
    pub seed: u64,
    /// The requested solver backend; `Auto` is resolved when the pipeline
    /// starts.
    pub backend: BackendChoice,
    /// The GPU backend's memory budget, in MiB.
    pub gpu_memory: usize,
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
    pub format: Option<OutputFormat>,
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
    pub feature_weight: Option<f32>,
    pub purity_weight: Option<f32>,
    pub feature_layers: Option<usize>,
    pub shift_range: Option<Vec<usize>>,
    pub perturbations: Option<Vec<usize>>,
    pub seed: Option<u64>,
    pub preview: Option<usize>,
    pub render: Option<PathBuf>,
    pub backend: Option<BackendChoice>,
    pub gpu_memory: Option<usize>,
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let mut c = Checker::default();
        let config = c.load_config(args.config.as_deref());

        let input = args.input.expect("clap requires <INPUT> without --list");
        if !input.exists() {
            c.errors.push(
                Diagnostic::new(
                    Kind::InputError,
                    format!("input '{}' does not exist", input.display()),
                )
                .path(input.display()),
            );
        }

        let dimension = c.dimension(args.row, args.columns);
        let resizing_method = c.resizing_method(
            args.fit,
            args.stretch,
            args.fill.as_deref(),
            (config.fit, config.stretch, config.fill.as_deref()),
        );
        let n_layers = c.n_layers(&args.layer_range, config.layer_range);
        let window_size = c.window_size(
            n_layers.map(|l| l.0),
            pick(
                args.window_size,
                "--window-size",
                config.window_size,
                "window_size",
            ),
        );
        let error_threshold = c.error_threshold(pick(
            args.error_threshold,
            "--error-threshold",
            config.error_threshold,
            "error_threshold",
        ));
        let feature_weight = c.feature_weight(pick(
            args.feature_weight,
            "--feature-weight",
            config.feature_weight,
            "feature_weight",
        ));
        let purity_weight = c.purity_weight(pick(
            args.purity_weight,
            "--purity-weight",
            config.purity_weight,
            "purity_weight",
        ));
        let feature_layers = c.feature_layers(pick(
            args.feature_layers,
            "--feature-layers",
            config.feature_layers,
            "feature_layers",
        ));
        let shift_range = c.shift_range(&args.shift_range, config.shift_range);
        let perturbations = c.perturbations(&args.perturbations, config.perturbations);
        let refinement = RefinementConfig {
            refinement_pass: args.refinement_pass.or(config.refinement_pass).unwrap_or(2),
            window_size,
            error_threshold,
            refinement_candidate: args
                .refinement_candidate
                .or(config.refinement_candidate)
                .unwrap_or(5),
            exact_candidates: args
                .exact_candidates
                .or(config.exact_candidates)
                .unwrap_or(DEFAULT_EXACT_CANDIDATES),
        };
        let preview = pick(args.preview, "--preview", config.preview, "preview");
        c.preview(preview);
        let exclude_patterns = c.exclude(
            pick(
                args.exclude_patterns.map(split_ids),
                "--exclude-patterns",
                config.exclude_patterns,
                "exclude_patterns",
            ),
            &pattern::ids(),
            "pattern",
            "patterns",
        );
        let exclude_blocks = c.exclude(
            pick(
                args.exclude_blocks.map(split_ids),
                "--exclude-blocks",
                config.exclude_blocks,
                "exclude_blocks",
            ),
            &block::ids(),
            "block",
            "blocks",
        );
        let backend = args.backend.or(config.backend).unwrap_or_default();
        let gpu_memory = pick(
            args.gpu_memory,
            "--gpu-memory",
            config.gpu_memory,
            "gpu_memory",
        );
        c.gpu_memory(
            gpu_memory,
            backend,
            pattern::ids().len().saturating_sub(exclude_patterns.len()),
            &refinement,
            n_layers,
        );

        if !c.errors.is_empty() {
            fail(c.errors);
        }

        let format = args.format.or(config.format).unwrap_or_default();
        let seed = pick(args.seed, "--seed", config.seed, "seed");
        // Warnings only once the configuration is known to be valid.
        if let Some((_, origin)) = preview
            && format != OutputFormat::Html
        {
            warn(c.diag(
                Kind::NoEffect,
                origin,
                format!(
                    "{} has no effect with --format {}",
                    c.name(origin),
                    format_name(format)
                ),
            ));
        }
        if let Some((_, origin)) = seed
            && perturbations.is_none()
        {
            warn(c.diag(
                Kind::NoEffect,
                origin,
                format!("{} has no effect without --perturbations", c.name(origin)),
            ));
        }
        if let Some((_, origin)) = gpu_memory
            && backend == BackendChoice::Cpu
        {
            warn(c.diag(
                Kind::NoEffect,
                origin,
                format!("{} has no effect with --backend cpu", c.name(origin)),
            ));
        }

        Config {
            input,
            output: args.output.expect("clap requires <OUTPUT> without --list"),
            format,
            dimension,
            workers: args.workers.or(config.workers),
            resizing_method,
            debug: args.debug,
            exclude_patterns,
            exclude_blocks,
            preview: preview.map(|(p, _)| p),
            render: args.render.or(config.render),
            n_layers: n_layers.unwrap_or(DEFAULT_LAYERS),
            refinement,
            feature_weight,
            purity_weight,
            feature_layers,
            shift_range,
            perturbations,
            seed: seed.map_or(0, |(s, _)| s),
            backend,
            gpu_memory: gpu_memory.map_or(DEFAULT_GPU_MEMORY, |(m, _)| m),
        }
    }
}

impl Config {
    /// The effective settings, for the `--json` result: enough to repeat the
    /// run. `workers` is the pool size actually used and `backend` the backend
    /// that actually ran (`"cpu"` or `"gpu"`); `gpu_memory` is reported only
    /// for the GPU backend, which is the only one that uses it.
    pub fn settings_json(&self, workers: usize, backend: &str) -> Value {
        let (layout, fill) = match self.resizing_method {
            ResizingMethod::Fit => ("fit", None),
            ResizingMethod::Stretch => ("stretch", None),
            ResizingMethod::Fill([r, g, b]) => ("fill", Some(format!("#{r:02x}{g:02x}{b:02x}"))),
        };
        let (row, columns) = match self.dimension {
            Dimension::Row(r) => (Some(r), None),
            Dimension::Column(c) => (None, Some(c)),
        };
        fn sorted(set: &HashSet<String>) -> Vec<&str> {
            let mut ids: Vec<&str> = set.iter().map(String::as_str).collect();
            ids.sort_unstable();
            ids
        }
        json!({
            "row": row,
            "columns": columns,
            "layout": layout,
            "fill_color": fill,
            "layer_range": [self.n_layers.0, self.n_layers.1],
            "refinement_pass": self.refinement.refinement_pass,
            "window_size": self.refinement.window_size,
            "error_threshold": shortest(self.refinement.error_threshold),
            "refinement_candidate": self.refinement.refinement_candidate,
            "exact_candidates": self.refinement.exact_candidates,
            "feature_weight": shortest(self.feature_weight),
            "purity_weight": shortest(self.purity_weight),
            "feature_layers": self.feature_layers,
            "shift_range": [self.shift_range.0, self.shift_range.1],
            "perturbations": self.perturbations.map(|(t, d, r)| [t, d, r]),
            "seed": self.seed,
            "workers": workers,
            "exclude_patterns": sorted(&self.exclude_patterns),
            "exclude_blocks": sorted(&self.exclude_blocks),
            "preview": self.preview,
            "backend": backend,
            "gpu_memory": (backend == "gpu").then_some(self.gpu_memory),
        })
    }
}

/// An `f32` as the decimal it was written as: `0.7`, not the `0.699999988`
/// its widening to `f64` would print.
fn shortest(x: f32) -> Value {
    x.to_string()
        .parse::<f64>()
        .map_or(Value::Null, Value::from)
}

pub fn format_name(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Html => "html",
        OutputFormat::Schem => "schem",
        OutputFormat::Litematic => "litematic",
        OutputFormat::Json => "json",
    }
}

/// Where a value came from, which decides how a problem with it is named:
/// `--window-size`, or `window_size in 'cfg.toml'`.
#[derive(Clone, Copy)]
enum Origin {
    Flag(&'static str),
    Key(&'static str),
}

/// The CLI value if given, else the config-file value, with where it came from.
fn pick<T>(
    arg: Option<T>,
    flag: &'static str,
    file: Option<T>,
    key: &'static str,
) -> Option<(T, Origin)> {
    match (arg, file) {
        (Some(v), _) => Some((v, Origin::Flag(flag))),
        (None, Some(v)) => Some((v, Origin::Key(key))),
        (None, None) => None,
    }
}

/// `'globe, mojang'` → `["globe", "mojang"]`.
fn split_ids(list: String) -> Vec<String> {
    list.split(',')
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .collect()
}

/// Validation state: every problem found so far, and the config file path the
/// `Key` origins refer to.
#[derive(Default)]
struct Checker {
    errors: Vec<Diagnostic>,
    config_path: String,
}

impl Checker {
    fn name(&self, origin: Origin) -> String {
        match origin {
            Origin::Flag(flag) => flag.to_string(),
            Origin::Key(key) => format!("{key} in '{}'", self.config_path),
        }
    }

    fn diag(&self, kind: Kind, origin: Origin, message: String) -> Diagnostic {
        let d = Diagnostic::new(kind, message);
        match origin {
            Origin::Flag(flag) => d.option(flag),
            Origin::Key(key) => d.key(key).config(&self.config_path),
        }
    }

    /// Records `<name> must be <requirement> (got <value>)`.
    fn invalid(&mut self, origin: Origin, value: impl Display, requirement: impl Display) {
        let message = format!("{} must be {requirement} (got {value})", self.name(origin));
        let d = self.diag(Kind::InvalidValue, origin, message).value(value);
        self.errors.push(d);
    }

    fn load_config(&mut self, path: Option<&Path>) -> ConfigToml {
        let Some(path) = path else {
            return ConfigToml::default();
        };
        self.config_path = path.display().to_string();
        let shown = &self.config_path;

        let problem = if !path.exists() {
            format!("config file '{shown}' does not exist")
        } else {
            match std::fs::read_to_string(path) {
                Err(e) => format!("could not read config file '{shown}': {e}"),
                Ok(text) => match toml::from_str::<ConfigToml>(&text) {
                    Ok(config) => return config,
                    Err(e) => format!(
                        "could not parse config file '{shown}': {}",
                        e.to_string().trim_end()
                    ),
                },
            }
        };
        let d = Diagnostic::new(Kind::ConfigError, problem)
            .option("--config")
            .path(shown);
        self.errors.push(d);
        ConfigToml::default()
    }

    fn dimension(&mut self, row: Option<usize>, col: Option<usize>) -> Dimension {
        match (row, col) {
            (Some(r), None) => {
                if r < 1 {
                    self.invalid(Origin::Flag("--row"), r, "at least 1");
                }
                Dimension::Row(r)
            }
            (None, Some(c)) => {
                if c < 1 {
                    self.invalid(Origin::Flag("--columns"), c, "at least 1");
                }
                Dimension::Column(c)
            }
            (Some(r), Some(_)) => {
                self.errors.push(Diagnostic::new(
                    Kind::Conflict,
                    "only one of --row or --columns can be given",
                ));
                Dimension::Row(r)
            }
            (None, None) => {
                self.errors.push(
                    Diagnostic::new(Kind::Missing, "one of --row or --columns is required").hint(
                        "--row sets the wall height, --columns its width; the other is inferred",
                    ),
                );
                Dimension::Row(1)
            }
        }
    }

    fn resizing_method(
        &mut self,
        fit: bool,
        stretch: bool,
        fill: Option<&str>,
        config_settings: (Option<bool>, Option<bool>, Option<&str>),
    ) -> ResizingMethod {
        match (fit, stretch, fill) {
            (true, false, None) => ResizingMethod::Fit,
            (false, true, None) => ResizingMethod::Stretch,
            (false, false, Some(color)) => self.color(Origin::Flag("--fill"), color),
            (false, false, None) => match config_settings {
                (Some(true), Some(false) | None, None) => ResizingMethod::Fit,
                (Some(false) | None, Some(true), None) => ResizingMethod::Stretch,
                (Some(false) | None, Some(false) | None, Some(color)) => {
                    self.color(Origin::Key("fill"), color)
                }
                (Some(false) | None, Some(false) | None, None) => ResizingMethod::Fit,
                _ => {
                    let d = Diagnostic::new(
                        Kind::Conflict,
                        format!(
                            "only one of fit, stretch or fill can be set in '{}'",
                            self.config_path
                        ),
                    )
                    .config(&self.config_path);
                    self.errors.push(d);
                    ResizingMethod::Fit
                }
            },
            _ => {
                self.errors.push(Diagnostic::new(
                    Kind::Conflict,
                    "only one of --fit, --stretch or --fill can be given",
                ));
                ResizingMethod::Fit
            }
        }
    }

    fn color(&mut self, origin: Origin, color: &str) -> ResizingMethod {
        match parse_color(color) {
            Ok(rgb) => ResizingMethod::Fill(rgb),
            Err(problem) => {
                let message = format!(
                    "{} must be a color (got '{color}'): {problem}",
                    self.name(origin)
                );
                let d = self
                    .diag(Kind::InvalidValue, origin, message)
                    .value(color)
                    .hint("use a form like '#ff9453', '9,4,87' or 'rgb(11, 45, 14)'");
                self.errors.push(d);
                ResizingMethod::Fit
            }
        }
    }

    /// `None` when the range is invalid, so the checks that depend on it are
    /// skipped instead of reporting a knock-on error.
    fn n_layers(&mut self, args: &[usize], file: Option<Vec<usize>>) -> Option<(usize, usize)> {
        let (layers, origin) = if !args.is_empty() {
            ((args[0], args[1]), Origin::Flag("--layer-range"))
        } else if let Some(range) = file {
            if range.len() != 2 {
                self.invalid(
                    Origin::Key("layer_range"),
                    format!("{range:?}"),
                    "a list of two elements [MIN, MAX]",
                );
                return None;
            }
            ((range[0], range[1]), Origin::Key("layer_range"))
        } else {
            return Some(DEFAULT_LAYERS);
        };

        let shown = format!("{} {}", layers.0, layers.1);
        let before = self.errors.len();
        if layers.0 > layers.1 {
            self.invalid(origin, &shown, "MIN MAX with MIN no greater than MAX");
        }
        if layers.0 < 1 {
            self.invalid(origin, &shown, "MIN MAX with MIN at least 1");
        }
        (self.errors.len() == before).then_some(layers)
    }

    fn window_size(&mut self, min_layer: Option<usize>, value: Option<(usize, Origin)>) -> usize {
        let Some((k, origin)) = value else {
            return 2;
        };
        if k < 1 {
            self.invalid(origin, k, "at least 1");
        } else if let Some(min_layer) = min_layer
            && k > min_layer
        {
            self.invalid(
                origin,
                k,
                format!("at most the minimum layer count {min_layer} (MIN of the layer range)"),
            );
        }
        k
    }

    fn error_threshold(&mut self, value: Option<(f32, Origin)>) -> f32 {
        let Some((threshold, origin)) = value else {
            return 0.7;
        };
        if !(0.0..=1.0).contains(&threshold) {
            self.invalid(origin, threshold, "between 0.0 and 1.0");
        }
        threshold
    }

    /// The feature weight is a ratio, not a bounded fraction: `0` disables the
    /// term and `1` weighs the idealised banner exactly as heavily as the real
    /// target, so there is a sensible reason to go above 1 and no defensible
    /// place to cap it. Only negative and non-finite values are refused.
    fn feature_weight(&mut self, value: Option<(f32, Origin)>) -> f32 {
        let Some((weight, origin)) = value else {
            return DEFAULT_FEATURE_WEIGHT;
        };
        if !weight.is_finite() || weight < 0.0 {
            self.invalid(origin, weight, "a finite number of at least 0.0");
        }
        weight
    }

    /// Like the feature weight, a ratio with no natural cap: only negative and
    /// non-finite values are refused.
    fn purity_weight(&mut self, value: Option<(f32, Origin)>) -> f32 {
        let Some((weight, origin)) = value else {
            return DEFAULT_PURITY_WEIGHT;
        };
        if !weight.is_finite() || weight < 0.0 {
            self.invalid(origin, weight, "a finite number of at least 0.0");
        }
        weight
    }

    fn feature_layers(&mut self, value: Option<(usize, Origin)>) -> usize {
        let Some((layers, origin)) = value else {
            return DEFAULT_FEATURE_LAYERS;
        };
        if !LAYER_COUNTS.contains(&layers) {
            self.invalid(origin, layers, "2, 3 or 4");
        }
        layers
    }

    /// The shift may not exceed the banner's margins inside its block
    /// ([`MAX_SHIFT`]), so a shifted patch stays in its own block column.
    fn shift_range(&mut self, args: &[usize], file: Option<Vec<usize>>) -> (usize, usize) {
        let (range, origin) = if !args.is_empty() {
            ((args[0], args[1]), Origin::Flag("--shift-range"))
        } else if let Some(range) = file {
            if range.len() != 2 {
                self.invalid(
                    Origin::Key("shift_range"),
                    format!("{range:?}"),
                    "a list of two elements [X, Y]",
                );
                return (0, 0);
            }
            ((range[0], range[1]), Origin::Key("shift_range"))
        } else {
            return (0, 0);
        };

        let shown = format!("{} {}", range.0, range.1);
        if range.0 > MAX_SHIFT.0 {
            self.invalid(
                origin,
                &shown,
                format!("X Y with X at most {}", MAX_SHIFT.0),
            );
        }
        if range.1 > MAX_SHIFT.1 {
            self.invalid(
                origin,
                &shown,
                format!("X Y with Y at most {}", MAX_SHIFT.1),
            );
        }
        range
    }

    fn perturbations(
        &mut self,
        args: &[usize],
        file: Option<Vec<usize>>,
    ) -> Option<(usize, usize, usize)> {
        let values = if !args.is_empty() {
            args.to_vec()
        } else {
            let values = file?;
            if values.len() != 3 {
                self.invalid(
                    Origin::Key("perturbations"),
                    format!("{values:?}"),
                    "a list of three elements [TOP_N, DUPLICATES, ROUNDS]",
                );
                return None;
            }
            values
        };
        // Any 0 turns the search off.
        (!values.contains(&0)).then(|| (values[0], values[1], values[2]))
    }

    /// `--preview 0` is meaningless — a zero-pixel image — and almost
    /// certainly a typo, so it is refused rather than silently clamped to one
    /// pixel.
    fn preview(&mut self, value: Option<(usize, Origin)>) {
        if let Some((0, origin)) = value {
            self.invalid(origin, 0, "at least 1");
        }
    }

    /// The GPU memory budget must be at least 1 MiB, and large enough for the
    /// GPU backend to solve one cell with these settings. The size check uses the largest layer count the range
    /// allows, since the per-cell budgets are not known yet. It is skipped
    /// under `--backend cpu`, which never uses the budget.
    fn gpu_memory(
        &mut self,
        value: Option<(usize, Origin)>,
        backend: BackendChoice,
        n_patterns: usize,
        refinement: &RefinementConfig,
        n_layers: Option<(usize, usize)>,
    ) {
        let Some((mib, origin)) = value else {
            return;
        };
        if mib < 1 {
            self.invalid(origin, mib, "at least 1");
            return;
        }
        if backend != BackendChoice::Cpu
            && let Some((_, max_layers)) = n_layers
        {
            const MIB: usize = 1 << 20;
            let needed = crate::gpu::min_memory_budget(n_patterns, refinement, max_layers);
            if mib.saturating_mul(MIB) < needed {
                self.invalid(
                    origin,
                    mib,
                    format!(
                        "at least {} MiB for these settings (one cell and one solver slot)",
                        needed.div_ceil(MIB)
                    ),
                );
            }
        }
    }

    /// An excluded id that names nothing is a user mistake, not a silent
    /// no-op; excluding every id leaves the solver nothing to work with.
    fn exclude(
        &mut self,
        value: Option<(Vec<String>, Origin)>,
        valid: &[String],
        noun: &str,
        list: &str,
    ) -> HashSet<String> {
        let Some((ids, origin)) = value else {
            return HashSet::new();
        };
        let ids: HashSet<String> = ids.into_iter().collect();

        let mut unknown: Vec<&String> = ids.iter().filter(|id| !valid.contains(id)).collect();
        unknown.sort_unstable();
        for id in unknown {
            let d = self
                .diag(Kind::UnknownId, origin, format!("unknown {noun} id '{id}'"))
                .value(id)
                .suggestions(suggest(id, valid))
                .hint(format!("run `bannerify --list {list}` for every valid id"));
            self.errors.push(d);
        }
        if valid.iter().all(|id| ids.contains(id)) {
            let d = self.diag(
                Kind::InvalidValue,
                origin,
                format!(
                    "{} excludes every {noun}; at least one must remain",
                    self.name(origin)
                ),
            );
            self.errors.push(d);
        }
        ids
    }
}

/// Up to three valid ids close to a mistyped one, closest first. "Close" is
/// one edit per three characters, between 1 and 2, so short ids don't match
/// half the table.
fn suggest(id: &str, valid: &[String]) -> Vec<String> {
    let limit = (id.chars().count() / 3).clamp(1, 2);
    let mut close: Vec<(usize, &String)> = valid
        .iter()
        .map(|v| (edit_distance(id, v), v))
        .filter(|&(d, _)| d <= limit)
        .collect();
    close.sort();
    close.into_iter().take(3).map(|(_, v)| v.clone()).collect()
}

/// Levenshtein distance over chars.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

/// Parse `#rrggbb`, `rrggbb`, `r,g,b` or `rgb(r, g, b)`.
pub fn parse_color(s: &str) -> Result<[u8; 3], String> {
    if let Some(hex) = s
        .strip_prefix('#')
        .or_else(|| (!s.contains(',')).then_some(s))
    {
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("a hex color needs exactly 6 hex digits".to_string());
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
        return Err(format!("expected 3 components, found {}", parts.len()));
    }

    Ok([
        parse_component(parts[0])?,
        parse_component(parts[1])?,
        parse_component(parts[2])?,
    ])
}

fn parse_component(p: &str) -> Result<u8, String> {
    p.parse::<u8>()
        .map_err(|_| format!("component '{p}' is not a whole number from 0 to 255"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestions_rank_close_ids_and_drop_distant_ones() {
        let valid: Vec<String> = ["globe", "flow", "flower", "mojang", "creeper"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(suggest("glob", &valid), ["globe"]);
        assert_eq!(suggest("flowr", &valid), ["flow", "flower"]);
        assert!(suggest("zzzzzz", &valid).is_empty());
    }
}
