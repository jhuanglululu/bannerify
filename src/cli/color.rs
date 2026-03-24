use std::sync::LazyLock;

use colored::Colorize;

static VALID_COLOR_STR: LazyLock<String> = LazyLock::new(|| {
    format!(
        "\n       valid color format includes: '{}', '{}' and '{}'",
        "#ff9453".yellow(),
        "9,4,87".yellow(),
        "rgb(11, 45, 14)".yellow()
    )
});

pub fn parse_color(s: &str) -> Result<[u8; 3], String> {
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

    let parse_component = |p: &str| {
        p.parse::<u8>().map_err(|_| {
            format!(
                "invalid color component '{}' in '{}'. {}",
                p.yellow(),
                s.yellow(),
                *VALID_COLOR_STR
            )
        })
    };

    Ok([
        parse_component(parts[0])?,
        parse_component(parts[1])?,
        parse_component(parts[2])?,
    ])
}
