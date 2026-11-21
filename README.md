# bannerify

**English** | [繁體中文](README.zh-TW.md)

Approximate an image with a wall of Minecraft banners.

Give it an image and a wall size; it picks a base color and a stack of dyed
patterns for every banner, matches a background block for every wall
position, and writes **one self-contained HTML page** with an
original-vs-result comparison slider, a per-banner crafting guide, a
materials list, and downloadable `.schem` / `.litematic` schematics.

![The Great Wave off Kanagawa as a 100-row banner wall](docs/example.png)

*"The Great Wave off Kanagawa" as a 100-row banner wall (15,000 banners).*

## Build

Building needs a C linker and Cargo, the Rust build tool. Install the
linker first:

- **Windows:** the MSVC Build Tools, a download of several GB. Install them
  before Rust; if Rust comes first, its installer may install Visual Studio
  Community by itself.

  ```sh
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

- **Debian and Ubuntu, including WSL:** `sudo apt install build-essential`
- **macOS:** `xcode-select --install`

Then
[install Rust and Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).
On Windows, open a new terminal afterwards: terminals that were already open
do not have `cargo` on their `PATH`.

```sh
cargo install --git https://github.com/jhuanglululu/bannerify
```

This downloads the source, builds a release binary and copies it to
`~/.cargo/bin/bannerify`; you do not need to clone the repository or have
`git` installed. The Rust installer puts `~/.cargo/bin` on your `PATH`, so
`bannerify` works from any folder. All assets (banner patterns, block
textures, GPU shaders) are embedded — the binary is standalone.

On an Intel or AMD CPU (Windows, Linux or an Intel Mac), use this command
instead:

```sh
cargo install --git https://github.com/jhuanglululu/bannerify --config "build.rustflags=['-C','target-feature=+avx2,+fma']"
```

It builds the CPU side with AVX2+FMA (any Intel/AMD CPU from ~2013
onwards), which the CPU side uses even when the solve runs on the GPU. The
first command leaves these out and builds slower scalar code on x86. A
build with AVX2+FMA exits with code `5` on a CPU that lacks them; install
with the first command there.

The GPU backend is always compiled in; there is no feature flag for it and
no GPU SDK is needed to build. At run time it uses the system's GPU driver,
and falls back to the CPU when no GPU can be opened (see
[GPU backend](#gpu-backend)).

## Install with AI

If you use an AI coding agent (Claude Code, Codex, Cursor and the like), it
can do the build for you. Read the instruction below, then paste it into
the agent:

```text
Install bannerify from https://github.com/jhuanglululu/bannerify.

1. Follow the Build section of its README. Do not clone the repository.
2. Ask me before installing anything other than bannerify. If a command
   needs sudo or administrator rights, give it to me to run.
3. Download
   https://raw.githubusercontent.com/jhuanglululu/bannerify/main/docs/example.png
   and test with it, then delete `example.png` and `test.html`:
   `bannerify example.png test.html --row 10 --debug`
4. Change nothing else on my machine.

Then report where bannerify was installed, the test's `backend:` line and
run time, everything you installed or changed, and any error in full.
```

## Usage

```sh
bannerify input.jpg output.html --row 20
```

Size the wall with `--row N` **or** `--columns N` (blocks); the other axis
is inferred from the image's aspect ratio. The image is resized internally —
never scale it yourself.

Open `output.html` in a browser: drag the slider to compare against the
original, type a row/column into the crafting guide to get that banner's
pattern steps and a copy-pasteable `/give` command, and download the
schematics from the buttons at the top.

### Output formats

Use `--format schem` or `--format litematic` to write a standalone schematic
without generating HTML or its preview images:

```sh
bannerify input.jpg output.schem --row 20 --format schem
bannerify input.jpg output.litematic --row 20 --format litematic
```

`--format html` is the default and includes both schematic downloads in the
report. The output path is used exactly as supplied; its extension does not
select the format. You can also set `format = "schem"` (or `"html"` /
`"litematic"` / `"json"`) in a TOML config passed with `--config`; the CLI
flag takes precedence.

`--format json` writes the solved wall as data for other programs: every
banner (row-major, 0-based from the top-left) with its base color, pattern
layers and `/give` command, the background block grid (one row taller than
the banners; banner `(r, c)` hangs on block `(r, c)`), and the materials. Ids
are unprefixed Minecraft ids.

`--render wall.png` can additionally save a PNG in any mode. `--preview` only
affects HTML and has no effect on other output.

### Quality knobs

The defaults are a good speed/quality balance. For a nicer result:

```sh
bannerify input.jpg output.html --row 40 -p 3 4 2
```

- `-p, --perturbations TOP_N DUPLICATES ROUNDS` — random-restart search on
  top of the base solver. Big quality gain, costs roughly
  `TOP_N × DUPLICATES × ROUNDS` extra solves per banner. `3 4 2` is a
  solid setting.
- `-x, --exact-candidates N` — how many candidates each refinement step
  scores exactly, by perceptual (OKLab) distance, instead of by the cheap
  approximation (default `20`). Raise it for a slightly closer match,
  lower it (or `0`) for speed.
- `-F, --feature-weight λ` — how hard the solver is pulled towards clean
  flat colors and crisp edges (default `0.5`). Every banner is also fitted
  by an idealised two-layer banner in pure dye colors, and λ weighs that
  fit against the raw pixels. Raise it when logos and flat areas come out
  muddy, lower it (or `0`) for photographs.
- `-G, --gradient-weight μ` — removes pattern texture from areas the image
  shows as flat (default `0`, off). Every pair of neighboring pixels whose
  difference does not match the image's is charged the mismatch, times μ.
  It keeps the colors the solver would otherwise pick. It works best with
  the feature weight off: try `-F 0 -G 0.5` on logos and flags. It does not
  sharpen edges, and above `1` a pattern edge that sits a pixel away from
  the image's edge is charged enough that the banner may lose the edge.
- `-U, --purity-weight ν` — how strongly the solver prefers pure dye
  colors over mixes of them (default `0`, off). Every pixel is charged its
  distance to the nearest of the 16 dye colors, times ν, so a banner drawn
  in clean dye colors beats one whose blends average to the right shade.
  Around `0.25` with `-F 0` it cleans flat areas; from `0.5` up, a color
  that lies between two dyes moves to one of them, which can be the wrong
  hue. It penalizes photographic shades, so expect a higher error on
  photographs.
- `-S, --shift-range X Y` — lets each banner be solved against the image
  moved by up to `X` pixels sideways (at most `2`) and `Y` up or down (at
  most `4`), whichever position a clean banner fits best (default `0 0`,
  off). `1 1` straightens horizontal and vertical edges that fall just
  beside a pattern edge; larger ranges turn diagonals and curves into steps
  and take longer, because every position is searched for every banner. The
  banner is still drawn at its real position, and the reported error is
  measured against the unmoved image, so the reported error rises.
- `-s, --seed N` — perturbation is deterministic per seed; try a few seeds
  and keep the best result.
- `-L, --layer-range MIN MAX` — how many pattern layers a banner may use
  (default `4 6`). More layers = closer match, more crafting steps.

### Layout

When the image's aspect ratio doesn't match the wall's:

- `--fit` *(default)* — scale up until the wall is covered; edges crop.
- `--stretch` — distort to fit exactly.
- `--fill COLOR` — letterbox with a color (`'#ff9453'`, `'9,4,87'`,
  `'rgb(114, 5, 14)'`).

### Choosing patterns and blocks

```sh
bannerify input.jpg output.html --row 20 -P globe,mojang -B default,-falling,-beacon
```

- `-P, --exclude-patterns` — banner pattern ids the solver may not use
  (e.g. patterns you can't craft).
- `-B, --blocks` — the blocks the background may be built from (default
  `default`, every block). It is a comma-separated list of block ids and
  block sets, applied left to right; a leading `-` removes a block or a set
  instead of adding it. The list starts empty, so begin with a set:
  - `default,-falling,-beacon` — everything except the blocks that fall
    (sand, gravel, concrete powder) and the beacon.
  - `wools,concretes,-black_wool` — two color families, minus one block.

The built-in sets are `default`, `wools`, `concretes`, `concrete_powders`,
`terracottas`, `glazed_terracottas`, `logs`, `planks`, `ores`, `falling` and
`expensive`. A config file can define more under `[block_sets]`, written
the same way, and they can then be named on the command line:

```toml
[block_sets]
quiet = ["wools", "concretes", "-black_wool", "smooth_stone"]
```

```sh
bannerify input.jpg output.html --row 20 -f config.toml -B quiet,-red_wool
```

An id that is neither a set nor a block is an error, unless a later `-id`
removes it again. A custom set cannot take the name of a block or of a
built-in set.

`bannerify --list patterns` (or `blocks`, `sets`, `dyes`, `backends`) prints
every valid id; `--list sets --json` also prints the blocks of each set.

### Output extras

- `--preview PIXELS` — largest dimension of the images embedded in the
  HTML (default: the banner wall's size). Smaller = smaller HTML file.
- `--render PATH` — also write the full-resolution wall render to a PNG on
  disk (not embedded).

### GPU backend

The banner solve runs on the GPU when one can be used: greedy fill,
refinement and perturbation run there, while resizing, block matching, the
feature map and the exports stay on the CPU. It works on NVIDIA, AMD, Intel
and Apple GPUs through [wgpu](https://wgpu.rs) (Vulkan on Linux, Vulkan or
DX12 on Windows, Metal on macOS). Linux needs the Vulkan loader
(`libvulkan.so.1`, package `libvulkan1` on Debian and Ubuntu) next to the
GPU driver.

Under WSL the solve runs on the CPU: Ubuntu's Mesa packages have no Vulkan
driver for the GPU that WSL exposes. To use the GPU on a Windows machine,
build and run bannerify on Windows itself.

Whole-run times in seconds for a 2048x1152 image, on a Core Ultra 5 225
with an RTX 3090 and on an M1 Pro:

| rows | settings | 225 CPU | RTX 3090 | M1 Pro CPU | M1 Pro GPU |
|---|---|---|---|---|---|
| 20 | defaults | 0.30 | 0.29 | 0.56 | 0.21 |
| 50 | defaults | 1.47 | 0.62 | 2.98 | 0.89 |
| 100 | defaults | 5.63 | 1.55 | 11.7 | 2.87 |
| 20 | `-p 5 2 2 -R 3 -F 0.1 -k 3` | 3.14 | 0.56 | 6.63 | 1.02 |
| 50 | `-p 5 2 2 -R 3 -F 0.1 -k 3` | 17.8 | 1.61 | 36.8 | 4.87 |
| 100 | `-p 5 2 2 -R 3 -F 0.1 -k 3` | 66.9 | 4.71 | 138 | 17.2 |

The GPU only loses on a very small default wall (10 rows: 0.25 s against
0.12 s on the RTX 3090 machine), where opening the device takes longer than
the solve.

The GPU matches the CPU's quality, not its exact banners. GPU arithmetic
rounds differently in the last digit, and where two candidate banners are
nearly tied the GPU may pick the other one: on a 50-row wall 12-24% of the
banners differed from the CPU's while the mean error differed by less than
0.1%. One device gives the same result every run for a given seed.

- `--backend auto` *(default)* — use the GPU whenever one can be opened,
  whatever the wall size or settings; otherwise solve on the CPU, with a
  warning and a `backend: cpu (GPU unavailable: ...)` line that give the
  reason.
- `--backend gpu` — use the GPU; if it cannot be used, stop with exit code
  `6`. Software renderers are never used.
- `--backend cpu` — use the CPU solver on every worker thread.
- `--gpu-memory MIB` — the memory budget for the GPU solver's buffers
  (default `2048`). A smaller budget splits the work into more batches; a
  value too small for the other settings is rejected.

`bannerify --list backends` prints the backends this machine can run.
`--debug` reports which backend ran, the GPU's name, the number of
GPU batches and the peak device memory; with the GPU backend the solve times
are wall time, with the CPU backend they are CPU time summed over the
workers.

### Config file

Every option can live in a TOML file; command-line flags win:

```sh
bannerify input.jpg output.html --row 20 -f config.toml
```

See [`config.toml`](config.toml) in this repo for a commented example of
every key.

### Scripting and LLM use

Log lines (`info:`, `warning:`, `error:`) go to stderr; stdout carries only
results. `-q, --quiet` keeps only warnings and errors.

`--json` prints exactly one JSON object on stdout instead — on success the
output file, wall size, effective settings, materials and an error score
(plus timings with `--debug`); on failure every problem found, each with a
`kind`, a `message`, the flag or config key at fault, and suggestions where
they apply:

```sh
$ bannerify input.jpg out.html --row 0 -P glob --json
{"ok":false,"schema_version":1,"errors":[
  {"kind":"invalid_value","message":"--row must be at least 1 (got 0)","option":"--row","value":"0"},
  {"kind":"unknown_id","message":"unknown pattern id 'glob'","option":"--exclude-patterns","value":"glob",
   "suggestions":["globe"],"hint":"run `bannerify --list patterns` for every valid id"}],"warnings":[]}
```

Exit codes: `0` success, `2` invalid arguments or config file, `3` input
image missing or unreadable, `4` output not writable, `5` CPU lacks
AVX2+FMA, `6` GPU backend unavailable (`--backend gpu`), `1` internal
error.

### Everything else

```sh
bannerify --help
```

`--debug` prints per-stage timings and peak memory if you're curious what
the time goes to. `-w N` caps the worker threads (default: all cores).
