# image2banner

Convert images to Minecraft banner walls. Outputs an HTML page with a preview, mcfunction commands, shulker box give commands, and a WorldEdit schematic!

## Usage

```
bannerify [options] <input> <output>
```

Exactly one of `-c` or `-r` is required. The other dimension is auto-calculated from the image aspect ratio.


## Examples

10 columns, auto rows
```bash
bannerify input.jpg output.html -c 10
```

use a config file
```bash
bannerify input.jpg output.html -c 10 -f quality.toml
```

8 rows, fill empty space with white after resizing
```bash
bannerify input.jpg output.html -r 8 --fill "#ffffff"
```

skip certain banner patterns
```bash
bannerify input.jpg output.html -c 10 --exclude-patterns creeper,skull,mojang
```

## Options

| Flag            | Description                          | Default          |
|-----------------|--------------------------------------|------------------|
| `-c, --columns` | Number of banner columns             | —                |
| `-r, --rows`    | Number of banner rows                | —                |
| `-f, --config`  | TOML config file for solver settings | —                |
| `-w, --workers` | Parallel worker processes            | auto (all cores) |

`-c, --columns`, `-r, --rows`, and `-f, --config` can not be passed in from config file.

---

### Resizing method for scaling

| Resizing Methods | Description                           | Default |
|------------------|---------------------------------------|---------|
| `--fit`          | Fit image, preserving aspect ratio    | ✓       |
| `--stretch`      | Stretch image to fill empty space     | -       |
| `--fill`         | Fill empty space with the given color | -       |

---

### Generation settings

| Generation               | Description                                      | Default |
|--------------------------|--------------------------------------------------|---------|
| `-P, --exclude-patterns` | Pattern ids to exclude (comma-separated)         | -       |
| `-B, --exclude-blocks`   | Block ids to exclude (comma-separated)           | -       |
| `-L, --layer-range`      | Layer Range: [MIN MAX]                           | 4, 6    |
| `-p, --perturbations`    | Perturbation search: [TOP_N, DUPLICATES, ROUNDS] | todo    |
| `-l, --lab-refine`       | Enable CIELAB refinement pass                    | todo    |

Keeping these options as default produces good enough result

---

### Refinement options

| Refinement               | Description                                                   | Default |
|--------------------------|---------------------------------------------------------------|---------|
| `-R, --refinement-pass`  | Refinement pass count                                         | 2       |
| `-k, --window-size`      | Refinement window size                                        | 2       |
| `-E, --error-threshold`  | Refinement error threshold for refinement passes (0.0 to 1.0) | 0.7     |
| `-C, --refinement-candidate` | Refinement max candidate                                      | 5       |

Keeping these options as default produces good enough result

---

CLI args always override config file values.

## Config File

Use `-f config.toml` to set solver defaults. Any setting can be overridden by CLI args.

See [config.toml](config.toml) for default config file.

## Project Structure

| File           | Description                                                   |
|----------------|---------------------------------------------------------------|
| `main.py`      | CLI entry point, config loading, argument parsing             |
| `geometry.py`  | Banner dimension constants                                    |
| `color.py`     | Minecraft color definitions and precomputed arrays            |
| `patterns.py`  | Pattern loading from disk and banner rendering                |
| `solver.py`    | Core fitting algorithms (greedy, refine, perturbation)        |
| `lab.py`       | Optional CIELAB perceptual refinement pass                    |
| `filter.py`    | Orchestration: chunking, parallelization, `fit_image`         |
| `preview.py`   | Preview image generation using tiled block textures           |
| `export.py`    | HTML export: mcfunction, shulker commands, schematic download |
| `schematic.py` | Sponge Schematic v2 (`.schem`) builder for WorldEdit          |
| `blocks.py`    | 288-block palette: texture loading, per-pixel LAB matching    |

## Output

todo
