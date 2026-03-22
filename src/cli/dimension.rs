use colored::Colorize;

/// banner dimension, the row is one less than the height of the blocks
pub enum Dimension {
    /// Number of banner rows
    Row(usize),

    /// Number of banner columns
    Column(usize),
}

impl clap::FromArgMatches for Dimension {
    fn from_arg_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        let rows = matches.get_one::<usize>("rows");
        let columns = matches.get_one::<usize>("columns");

        match (rows, columns) {
            (Some(_), Some(_)) => Err(clap::Error::raw(
                clap::error::ErrorKind::ArgumentConflict,
                format!(
                    "cannot use '{}' and '{}' together",
                    "--rows".yellow(),
                    "--columns".yellow()
                ),
            )),
            (None, None) => Err(clap::Error::raw(
                clap::error::ErrorKind::MissingRequiredArgument,
                format!(
                    "either '{}' or '{}' is required",
                    "--rows".yellow(),
                    "--columns".yellow()
                ),
            )),
            (Some(r), None) => {
                if *r < 2 {
                    Err(clap::Error::raw(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        format!(
                            "{} can not be less than 2. a banner is two blocks tall",
                            "--rows".yellow(),
                        ),
                    ))
                } else {
                    Ok(Self::Row(*r - 1))
                }
            }
            (None, Some(c)) => {
                if *c < 1 {
                    Err(clap::Error::raw(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        format!(
                            "{} can not be less than 1. a banner is two blocks wide",
                            "--columns".yellow(),
                        ),
                    ))
                } else {
                    Ok(Self::Column(*c))
                }
            }
        }
    }

    fn update_from_arg_matches(&mut self, matches: &clap::ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(matches)?;
        Ok(())
    }
}

impl clap::Args for Dimension {
    fn augment_args(cmd: clap::Command) -> clap::Command {
        cmd.arg(
            clap::Arg::new("rows")
                .short('r')
                .long("rows")
                .value_parser(clap::value_parser!(usize))
                .help("Height of output in blocks"),
        )
        .arg(
            clap::Arg::new("columns")
                .short('c')
                .long("columns")
                .value_parser(clap::value_parser!(usize))
                .help("Width of output in blocks"),
        )
    }

    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        Self::augment_args(cmd)
    }
}
