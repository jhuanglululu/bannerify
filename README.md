# bannerify

Approximate an image with a wall of Minecraft banners.

Give it an image and a wall size; it picks a base color and a stack of dyed
patterns for every banner, matches a background block for every wall
position, and writes **one self-contained HTML page** with an
original-vs-result comparison slider, a per-banner crafting guide, a
materials list, and downloadable `.schem` / `.litematic` schematics.

![The Great Wave off Kanagawa as a 100-row banner wall](docs/example.png)

*"The Great Wave off Kanagawa" as a 100-row banner wall (15,000 banners).*

## Build

```sh
cargo build --release
```

The binary is `target/release/bannerify`. All assets (banner patterns,
block textures) are embedded — the binary is standalone. Windows and Linux
x86 builds target AVX2+FMA by default (any Intel/AMD CPU from ~2013 onwards);
build with `--features force-scalar` for older machines.

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

### Excluding things

```sh
bannerify input.jpg output.html --row 20 -P globe,mojang -B beacon,ancient_debris
```

- `-P, --exclude-patterns` — banner pattern ids the solver may not use
  (e.g. patterns you can't craft).
- `-B, --exclude-blocks` — block ids the background matcher may not pick.

`bannerify --list patterns` (or `blocks`, `dyes`, `backends`) prints every
valid id.

### Output extras

- `--preview PIXELS` — largest dimension of the images embedded in the
  HTML (default: the banner wall's size). Smaller = smaller HTML file.
- `--render PATH` — also write the full-resolution wall render to a PNG on
  disk (not embedded).

### GPU backend

On macOS the banner solve runs on the Metal GPU: greedy fill, refinement
and perturbation run there, while resizing, block matching, the feature map
and the exports stay on the CPU. On an M1 Pro with the 40-row example wall,
the whole run takes 0.46 s instead of 1.7 s at the defaults and 4.3 s
instead of 24 s with `-p 5 2 2 -R 4 -k 4`. In every run measured on Apple
Silicon, the GPU backend gave the same output as the CPU backend, byte for
byte.

- `--backend auto` *(default)* — use the GPU when one can be used,
  otherwise the CPU. On macOS a fallback to the CPU prints a warning with
  the reason.
- `--backend gpu` — use the GPU; if it cannot be used (no Metal device, or
  not macOS), stop with exit code `6`.
- `--backend cpu` — use the CPU solver on every worker thread.
- `--gpu-memory MIB` — the memory budget for the GPU solver's buffers
  (default `2048`). A smaller budget splits the work into more batches; a
  value too small for the other settings is rejected.

`bannerify --list backends` prints the backends this machine can run.
`--debug` reports which backend ran, the GPU's name, the number of GPU
batches and the peak device memory; with the GPU backend the solve times
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
