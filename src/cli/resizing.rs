use crate::cli::color::parse_color;

pub enum ResizingMethod {
    Fit,
    Fill([u8; 3]),
    Stretch,
}

impl clap::FromArgMatches for ResizingMethod {
    fn from_arg_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        if matches.get_flag("fit") {
            Ok(Self::Fit)
        } else if let Some(s) = matches.get_one::<String>("fill") {
            match parse_color(s) {
                Ok(color) => Ok(Self::Fill(color)),
                Err(e) => Err(clap::Error::raw(clap::error::ErrorKind::InvalidValue, e)),
            }
        } else if matches.get_flag("stretch") {
            Ok(Self::Stretch)
        } else {
            Ok(Self::Fit) // default
        }
    }

    fn update_from_arg_matches(&mut self, matches: &clap::ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(matches)?;
        Ok(())
    }
}

impl clap::Args for ResizingMethod {
    fn augment_args(cmd: clap::Command) -> clap::Command {
        cmd.arg(
            clap::Arg::new("fit")
                .long("fit")
                .action(clap::ArgAction::SetTrue)
                .help_heading("Layout")
                .help("Fit image, preserving aspect ratio [default]"),
        )
        .arg(
            clap::Arg::new("fill")
                .long("fill")
                .value_name("COLOR")
                .help_heading("Layout")
                .help("Fill empty space with the given color (e.g. '#ff9453', 'rgb(114, 5, 14)', '9,4,87')"),
        )
        .arg(
            clap::Arg::new("stretch")
                .long("stretch")
                .action(clap::ArgAction::SetTrue)
                .help_heading("Layout")
                .help("Stretch image to fill empty space"),
        )
        .group(
            clap::ArgGroup::new("resizing")
                .args(["fit", "fill", "stretch"])
                .multiple(false),
        )
    }

    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        Self::augment_args(cmd)
    }
}
