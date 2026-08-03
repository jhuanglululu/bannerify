//! Cross-check the streamed resampler against Pillow's `Image.LANCZOS`.
//!
//! Pillow is optional: the test looks for a `python3` with PIL, falls back to
//! `uv run --with pillow python`, and skips (with a printed note) if neither is
//! available. Run with `--nocapture` to see which path was taken.
//!
//! The reference is Pillow's **`F` (float32) mode** resize, not its `u8` one.
//! Pillow's 8-bit path rounds *and clips* the horizontal intermediate to
//! `0..=255` before the vertical pass, so lanczos ringing that overshoots the
//! range is thrown away there; this pipeline keeps it in `f32` (as does
//! Pillow's own float path). On a high-contrast test image the two 8-bit
//! results differ by up to 18/255 for that reason alone — a deliberate
//! deviation, not an error. Against the float path the only difference left is
//! accumulation precision (Pillow: `double`; here: `f32` + fma), hence the
//! tolerance below in 0..255 units.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use bannerify::resample::{Options, PlanarU8, resize_to_planar_f32};

/// Max allowed absolute difference per sample, in 0..255 units.
const MAX_ABS: f32 = 0.005;
/// Max allowed mean absolute difference, in 0..255 units.
const MAX_MEAN: f64 = 0.0005;

/// A `python` invocation known to have PIL, if any.
fn python() -> Option<Vec<String>> {
    let probe = |cmd: &[&str]| -> bool {
        Command::new(cmd[0])
            .args(&cmd[1..])
            .args(["-c", "import PIL"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    for cmd in [
        vec!["python3"],
        vec!["uv", "run", "--with", "pillow", "python"],
    ] {
        if probe(&cmd) {
            return Some(cmd.iter().map(|s| (*s).to_string()).collect());
        }
    }
    None
}

/// Deterministic test plane: gradient plus a high-frequency checker, which is
/// where resampling differences show up first.
fn sample(width: usize, height: usize, seed: usize) -> Vec<u8> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            let base = (x * 160 / width.max(1) + y * 80 / height.max(1)) as u8;
            let checker = if (x / (2 + seed) + y / 3).is_multiple_of(2) {
                90
            } else {
                0
            };
            base.saturating_add(checker)
        })
        .collect()
}

#[test]
fn matches_pil_lanczos() {
    let Some(py) = python() else {
        println!("SKIPPED: no python3 with PIL and no working `uv run --with pillow`");
        return;
    };
    println!("using python: {}", py.join(" "));

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tmpdir");

    // upscale, downscale, non-uniform (up in y, down in x).
    for (seed, &(sw, sh, dw, dh)) in [(64, 48, 200, 150), (300, 200, 71, 53), (91, 37, 33, 160)]
        .iter()
        .enumerate()
    {
        let plane = sample(sw, sh, seed);
        let src_path = dir.join(format!("pil_src_{sw}x{sh}_{seed}.raw"));
        let ref_path = dir.join(format!("pil_ref_{sw}x{sh}_{dw}x{dh}.f32"));
        std::fs::write(&src_path, &plane).expect("write src");

        let script = format!(
            "from PIL import Image\n\
             d = open(r'{src}', 'rb').read()\n\
             im = Image.frombytes('L', ({sw}, {sh}), d).convert('F')\n\
             im = im.resize(({dw}, {dh}), Image.LANCZOS)\n\
             open(r'{dst}', 'wb').write(im.tobytes())\n",
            src = src_path.display(),
            dst = ref_path.display(),
        );
        let ok = Command::new(&py[0])
            .args(&py[1..])
            .args(["-c", &script])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "PIL resize failed");

        let raw = std::fs::read(&ref_path).expect("read PIL output");
        assert_eq!(raw.len(), dw * dh * 4, "unexpected PIL output size");
        let want: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let src = PlanarU8 {
            width: sw,
            height: sh,
            planes: vec![plane],
        };
        let got = resize_to_planar_f32(&src, dw, dh, Options::default());

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (a, b) in got[0].iter().zip(&want) {
            let d = (a - b).abs();
            max_diff = max_diff.max(d);
            sum_diff += f64::from(d);
        }
        let mean_diff = sum_diff / (dw * dh) as f64;
        println!("{sw}x{sh} -> {dw}x{dh}: max {max_diff:.5}, mean {mean_diff:.6}");
        assert!(
            max_diff <= MAX_ABS && mean_diff < MAX_MEAN,
            "{sw}x{sh} -> {dw}x{dh}: max abs diff {max_diff}, mean {mean_diff}"
        );
    }
}
