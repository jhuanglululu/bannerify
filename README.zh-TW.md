# bannerify

[English](README.md) | **繁體中文**

用一整面 Minecraft 旗幟牆來近似一張圖片。

給它一張圖片和牆的尺寸，它會為每面旗幟挑選底色和一疊染色圖案，為牆上
每個位置配對背景方塊，然後輸出**一個獨立的 HTML 頁面**，內含原圖與成果
的比較滑桿、每面旗幟的合成指南、材料清單，以及可下載的 `.schem` /
`.litematic` 藍圖檔。

![以 100 列旗幟牆呈現的《神奈川沖浪裏》](docs/example.png)

*以 100 列旗幟牆呈現的《神奈川沖浪裏》（15,000 面旗幟）。*

## 建置

建置需要 C 連結器和 Rust 的建置工具 Cargo。請先安裝連結器：

- **Windows：**MSVC Build Tools，下載量有數 GB。請在安裝 Rust 之前先
  裝好；如果先裝 Rust，它的安裝程式可能會自行安裝 Visual Studio
  Community。

  ```sh
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

- **Debian 和 Ubuntu（包含 WSL）：**`sudo apt install build-essential`
- **macOS：**`xcode-select --install`

接著
[安裝 Rust 與 Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)。
在 Windows 上，安裝後請開一個新的終端機：原本就開著的終端機，`PATH`
裡不會有 `cargo`。

```sh
cargo install --git https://github.com/jhuanglululu/bannerify
```

這會下載原始碼、建置 release 版的執行檔，並複製到
`~/.cargo/bin/bannerify`；不需要 clone 這個 repository，也不需要安裝
`git`。Rust 安裝程式會把 `~/.cargo/bin` 加進 `PATH`，所以在任何資料夾都能
執行 `bannerify`。所有素材（旗幟圖案、方塊材質、GPU 著色器）都內嵌在
執行檔中，可以單獨使用。

如果是 Intel 或 AMD 的 CPU（Windows、Linux 或 Intel Mac），請改用這個
指令：

```sh
cargo install --git https://github.com/jhuanglululu/bannerify --config "build.rustflags=['-C','target-feature=+avx2,+fma']"
```

它會以 AVX2+FMA 建置 CPU 端（約 2013 年之後的 Intel/AMD CPU 都支援）；
即使求解在 GPU 上執行，CPU 端仍然會用到這些指令集。第一個指令不含這些
旗標，在 x86 上會建置出較慢的純量版本。以 AVX2+FMA 建置的版本在不支援的
CPU 上會以結束代碼 `5` 結束；這時請改用第一個指令安裝。

GPU 後端一律會編譯進去，沒有對應的 feature 旗標，建置時也不需要任何 GPU
SDK。執行時它使用系統的 GPU 驅動程式，無法開啟 GPU 時會改用 CPU（見
[GPU 後端](#gpu-後端)）。

## 用 AI 安裝

如果你使用 AI 程式代理（Claude Code、Codex、Cursor 等），可以讓它替你
建置。請先讀完下面的指示，再貼給代理：

```text
請從 https://github.com/jhuanglululu/bannerify 安裝 bannerify。

1. 照 README 的「建置」一節操作。不要 clone 這個 repository。
2. 除了 bannerify 之外，安裝任何東西之前請先問我。需要 sudo 或系統
   管理員權限的指令，請交給我自己執行。
3. 下載
   https://raw.githubusercontent.com/jhuanglululu/bannerify/main/docs/example.png
   並用它測試，然後刪除 `example.png` 和 `test.html`：
   `bannerify example.png test.html --row 10 --debug`
4. 不要更動我電腦上的其他任何東西。

完成後請回報：bannerify 安裝在哪裡、測試輸出的 `backend:` 那一行和執行
時間、你安裝或更動的所有東西，以及任何錯誤的完整訊息。
```

## 使用方式

```sh
bannerify input.jpg output.html --row 20
```

用 `--row N` **或** `--columns N`（單位為方塊）指定牆的尺寸，另一個方向
會依圖片的長寬比推算。圖片會在內部縮放，不需要自己先縮圖。

用瀏覽器開啟 `output.html`：拖曳滑桿與原圖比較；在合成指南輸入列與欄，
可以看到那面旗幟的圖案步驟和可直接複製貼上的 `/give` 指令；頁面上方的
按鈕可以下載藍圖檔。

### 輸出格式

使用 `--format schem` 或 `--format litematic` 可以只輸出藍圖檔，不產生
HTML 和其中的預覽圖：

```sh
bannerify input.jpg output.schem --row 20 --format schem
bannerify input.jpg output.litematic --row 20 --format litematic
```

預設為 `--format html`，報告中包含兩種藍圖檔的下載。輸出路徑會完全照你
給的使用，副檔名不會決定格式。也可以在 `--config` 指定的 TOML 設定檔中
寫 `format = "schem"`（或 `"html"` / `"litematic"` / `"json"`）；命令列
旗標優先。

`--format json` 會把解出的牆輸出成資料，供其他程式使用：每面旗幟（以列
為主的順序，從左上角起算、索引從 0 開始）的底色、圖案層和 `/give`
指令，背景方塊網格（比旗幟多一列；旗幟 `(r, c)` 掛在方塊 `(r, c)` 上），
以及材料清單。ID 為不含前綴的 Minecraft ID。

`--render wall.png` 在任何模式下都可以額外存一張 PNG。`--preview` 只影響
HTML，對其他輸出沒有作用。

### 品質設定

預設值在速度和品質之間取得不錯的平衡。想要更好的結果：

```sh
bannerify input.jpg output.html --row 40 -p 3 4 2
```

- `-p, --perturbations TOP_N DUPLICATES ROUNDS`：在基本求解器之上做隨機
  重啟搜尋。品質提升很大，每面旗幟大約要多解
  `TOP_N × DUPLICATES × ROUNDS` 次。`3 4 2` 是穩定好用的設定。
- `-x, --exact-candidates N`：每個精修步驟中，有多少候選改用感知
  （OKLab）距離精確評分，而不是用便宜的近似值（預設 `20`）。調高可以
  稍微更接近原圖，調低（或設為 `0`）可以加快速度。
- `-F, --feature-weight λ`：求解器被拉向乾淨純色和銳利邊緣的強度（預設
  `0.5`）。每面旗幟也會用一面只含純染料色的理想化兩層旗幟來擬合，λ
  決定這個擬合相對於原始像素的權重。標誌和純色區域看起來混濁時調高，
  照片則調低（或設為 `0`）。
- `-U, --purity-weight ν`：求解器偏好純染料色、不偏好染料混色的程度
  （預設 `0`，關閉）。每個像素都會依它到 16 種染料色中最接近者的距離乘以
  ν 計分，所以用乾淨染料色畫成的旗幟，會勝過靠混色平均出正確色調的旗幟。
  它會懲罰照片中的漸層色調，所以照片的誤差會變高。
- `-s, --seed N`：同一個種子的擾動結果是固定的；可以多試幾個種子，保留
  最好的結果。
- `-L, --layer-range MIN MAX`：一面旗幟可以使用的圖案層數（預設
  `4 6`）。層數越多越接近原圖，合成步驟也越多。

### 版面配置

圖片的長寬比和牆不一致時：

- `--fit`*（預設）*：放大到蓋滿整面牆，邊緣會被裁切。
- `--stretch`：變形以完全符合牆的尺寸。
- `--fill COLOR`：用指定顏色填滿上下或左右的空白（`'#ff9453'`、
  `'9,4,87'`、`'rgb(114, 5, 14)'`）。

### 排除項目

```sh
bannerify input.jpg output.html --row 20 -P globe,mojang -B beacon,ancient_debris
```

- `-P, --exclude-patterns`：求解器不可使用的旗幟圖案 ID（例如你無法合成
  的圖案）。
- `-B, --exclude-blocks`：背景配對不可選用的方塊 ID。

`bannerify --list patterns`（或 `blocks`、`dyes`、`backends`）會列出所有
有效的 ID。

### 其他輸出選項

- `--preview PIXELS`：內嵌在 HTML 中的圖片的最大邊長（預設：旗幟牆的
  尺寸）。越小，HTML 檔案越小。
- `--render PATH`：另外把完整解析度的牆面算圖寫成磁碟上的 PNG（不會
  內嵌）。

### GPU 後端

有可用的 GPU 時，旗幟求解會在 GPU 上執行：貪婪填充、精修和擾動在 GPU
上執行，縮放、方塊配對、特徵圖和匯出則留在 CPU。它透過
[wgpu](https://wgpu.rs) 支援 NVIDIA、AMD、Intel 和 Apple 的 GPU（Linux
用 Vulkan，Windows 用 Vulkan 或 DX12，macOS 用 Metal）。Linux 除了 GPU
驅動程式之外，還需要 Vulkan loader（`libvulkan.so.1`，在 Debian 和
Ubuntu 上是 `libvulkan1` 套件）。

在 WSL 裡求解會在 CPU 上執行：Ubuntu 的 Mesa 套件沒有能驅動 WSL 所提供
GPU 的 Vulkan 驅動程式。要在 Windows 電腦上使用 GPU，請直接在 Windows
上建置和執行 bannerify。

以下是 2048x1152 圖片的完整執行時間（秒），測試機器為搭配 RTX 3090 的
Core Ultra 5 225，以及 M1 Pro：

| 列數 | 設定 | 225 CPU | RTX 3090 | M1 Pro CPU | M1 Pro GPU |
|---|---|---|---|---|---|
| 20 | 預設 | 0.30 | 0.29 | 0.56 | 0.21 |
| 50 | 預設 | 1.47 | 0.62 | 2.98 | 0.89 |
| 100 | 預設 | 5.63 | 1.55 | 11.7 | 2.87 |
| 20 | `-p 5 2 2 -R 3 -F 0.1 -k 3` | 3.14 | 0.56 | 6.63 | 1.02 |
| 50 | `-p 5 2 2 -R 3 -F 0.1 -k 3` | 17.8 | 1.61 | 36.8 | 4.87 |
| 100 | `-p 5 2 2 -R 3 -F 0.1 -k 3` | 66.9 | 4.71 | 138 | 17.2 |

GPU 只有在使用預設值的極小牆面上比較慢（10 列：在 RTX 3090 的機器上是
0.25 秒對 0.12 秒），因為開啟裝置花的時間比求解還長。

GPU 的品質和 CPU 相同，但選出的旗幟不會完全一樣。GPU 運算在最後一位數
的捨入方式不同，兩面候選旗幟幾乎同分時，GPU 可能選到另一面：在 50 列的
牆上，有 12-24% 的旗幟和 CPU 的結果不同，而平均誤差的差距不到 0.1%。
同一個裝置用同一個種子，每次執行的結果都相同。

- `--backend auto`*（預設）*：只要能開啟 GPU 就使用，不論牆的尺寸或
  設定；否則在 CPU 上求解，並在警告和
  `backend: cpu (GPU unavailable: ...)` 那一行說明原因。
- `--backend gpu`：使用 GPU；無法使用時以結束代碼 `6` 停止。絕不會使用
  軟體算繪器。
- `--backend cpu`：在所有工作執行緒上使用 CPU 求解器。
- `--gpu-memory MIB`：GPU 求解器緩衝區的記憶體預算（預設 `2048`）。預算
  較小時，工作會分成更多批次；數值小到無法配合其他設定時會被拒絕。

`bannerify --list backends` 會列出這台機器可以執行的後端。`--debug` 會
回報實際使用的後端、GPU 名稱、GPU 批次數和裝置記憶體用量峰值；使用 GPU
後端時，求解時間是實際經過的時間，使用 CPU 後端時則是所有工作執行緒的
CPU 時間總和。

### 設定檔

所有選項都可以寫在 TOML 檔案中；命令列旗標優先：

```sh
bannerify input.jpg output.html --row 20 -f config.toml
```

本儲存庫中的 [`config.toml`](config.toml) 是附有註解的範例，涵蓋所有
設定鍵。

### 腳本與 LLM 使用

記錄訊息（`info:`、`warning:`、`error:`）輸出到 stderr；stdout 只輸出
結果。`-q, --quiet` 只保留警告和錯誤。

`--json` 會改為在 stdout 輸出恰好一個 JSON 物件。成功時包含輸出檔案、
牆的尺寸、實際生效的設定、材料和誤差分數（加上 `--debug` 時還有各階段
時間）；失敗時列出找到的每個問題，各自帶有 `kind`、`message`、出錯的
旗標或設定鍵，以及適用時的建議：

```sh
$ bannerify input.jpg out.html --row 0 -P glob --json
{"ok":false,"schema_version":1,"errors":[
  {"kind":"invalid_value","message":"--row must be at least 1 (got 0)","option":"--row","value":"0"},
  {"kind":"unknown_id","message":"unknown pattern id 'glob'","option":"--exclude-patterns","value":"glob",
   "suggestions":["globe"],"hint":"run `bannerify --list patterns` for every valid id"}],"warnings":[]}
```

結束代碼：`0` 成功，`2` 引數或設定檔無效，`3` 輸入圖片不存在或無法
讀取，`4` 無法寫入輸出，`5` CPU 不支援 AVX2+FMA，`6` GPU 後端無法使用
（`--backend gpu`），`1` 內部錯誤。

### 其他

```sh
bannerify --help
```

想知道時間花在哪裡的話，`--debug` 會印出各階段的時間和記憶體用量峰值。
`-w N` 限制工作執行緒數量（預設：所有核心）。
