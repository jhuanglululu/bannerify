use colored::Colorize;

use crate::color::NUM_COLORS;

pub struct OptimalOption {
    pub layers: (usize, usize),
}

pub struct GreedyOption {
    pub layers: (usize, usize),
    pub colors: (usize, usize),
}

pub enum ComplexityOptions {
    Optimal(OptimalOption),
    Greedy(GreedyOption),
}

impl clap::FromArgMatches for ComplexityOptions {
    fn from_arg_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let layers_vec: Vec<usize> = matches.get_many("layer-range").unwrap().cloned().collect();
        let colors_opt: Option<Vec<usize>> = matches
            .get_many("color-greedy-range")
            .map(|v| v.cloned().collect());

        let layers = (layers_vec[0], layers_vec[1]);

        let mut errors = Vec::new();

        if layers.0 > layers.1 {
            errors.push(format!(
                "'{}' can not be greater than '{}' in '{}'",
                "MIN".yellow(),
                "MAX".yellow(),
                "--layers MIN MAX".yellow()
            ));
        }

        let out = if let Some(colors) = colors_opt {
            let min = colors[0];
            let max = colors[1];

            if min < 1 {
                errors.push(format!(
                    "'{}' can not be less than '{}' in '{}'",
                    "MIN".yellow(),
                    "1".yellow(),
                    "--colors MIN MAX".yellow()
                ));
            }

            if max > NUM_COLORS {
                errors.push(format!(
                    "'{}' can not be greater than '{}' in '{}'",
                    "MAX".yellow(),
                    NUM_COLORS.to_string().yellow(),
                    "--colors MIN MAX".yellow()
                ));
            }

            if max < min {
                errors.push(format!(
                    "'{}' can not be greater than '{}' in '{}'",
                    "MIN".yellow(),
                    "MAX".yellow(),
                    "--colors MIN MAX".yellow()
                ));
            }
            Self::Greedy(GreedyOption {
                layers,
                colors: (min, max),
            })
        } else {
            Self::Optimal(OptimalOption { layers })
        };

        if !errors.is_empty() {
            return Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                errors.join("\n       "),
            ));
        }

        Ok(out)
    }

    fn update_from_arg_matches(&mut self, matches: &clap::ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(matches)?;
        Ok(())
    }
}

impl clap::Args for ComplexityOptions {
    fn augment_args(cmd: clap::Command) -> clap::Command {
        cmd.arg(
            clap::Arg::new("layer-range")
                .short('L')
                .long("layer-range")
                .num_args(2)
                .value_parser(clap::value_parser!(usize))
                .default_values(["4", "6"])
                .help("Layer Range: MIN MAX")
                .help_heading("Generation"),
        )
        .arg(
            clap::Arg::new("color-greedy-range")
                .short('C')
                .long("color-greedy-range")
                .num_args(2)
                .value_parser(clap::value_parser!(usize))
                .help("Color greedy search candidates: MIN MAX")
                .help_heading("Generation"),
        )
    }

    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        Self::augment_args(cmd)
    }
}
