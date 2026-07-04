# See-through 個人用セットアップメモ

このリポジトリ (https://github.com/shitagaki-lab/see-through.git) を
`C:\dev\github\wit-maker\see-through` にクローンして構築した際の手順と、
実際にハマったポイントの記録。公式READMEの日本語要約 + このマシン固有の対応。

## 環境

- OS: Windows 11
- GPU: NVIDIA GeForce RTX 3060 Ti (VRAM 8GB)
- conda は未導入 → venv で代用
- Python 3.12 を新規インストール(winget)、3.10/3.11は既存だが README指定の3.12を使用

## セットアップ手順(実際に実行した内容)

### 1. Python 3.12 のインストール

```powershell
winget install -e --id Python.Python.3.12 --scope user
```

### 2. venv 作成

```powershell
cd C:\dev\github\wit-maker\see-through
py -3.12 -m venv .venv
```

以降、`.\.venv\Scripts\python.exe` を使う(activateしなくてもOK)。

### 3. PyTorch (CUDA 12.8) インストール

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128
```

ダウンロードは torch 単体で約3.5GBあるので時間がかかる。

### 4. 依存関係インストール

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`-e ./common` と `-e ./annotators` も含めて editable install される。

### 5. assets シンボリックリンク

Windowsでシンボリックリンクを作るには管理者権限が必要。
管理者権限なしでも作れる「ジャンクション」で代用した。

```
mklink /J assets common\assets
```

### 6. 低VRAM(8GB)向け: NF4量子化パイプライン用の追加パッケージ

このGPUは8GBなので、公式READMEの「8 GB GPUs」向け手順(NF4量子化)を採用。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-inference-bnb.txt
```

## 動作確認

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# => 2.8.0+cu128 True
```

## 使い方

以降のコマンドはすべて `.\.venv\Scripts\python.exe` を使う想定。
必ずリポジトリのルート (`C:\dev\github\wit-maker\see-through`) から実行すること
(PowerShellを開いた直後は `C:\WINDOWS\system32` 等になっているので、
最初に必ず `cd C:\dev\github\wit-maker\see-through` を実行する)。
画像パスに日本語やスペースが含まれる場合は `"..."` で囲むこと。

```powershell
cd C:\dev\github\wit-maker\see-through
.\.venv\Scripts\python.exe inference\scripts\inference_psd_quantized.py `
  --srcp "C:\Users\wiiiiii\Pictures\zonos\アバター\透過-全身.png" `
  --save_to_psd `
  --cpu_offload `
  --group_offload `
  --resolution 768 `
  --resolution_depth 512
```

### 1. レイヤー分解のメインパイプライン

標準版(通常VRAM向け、bf16、約12〜16GB VRAM):

```powershell
.\.venv\Scripts\python.exe inference\scripts\inference_psd.py `
  --srcp assets\test_image.png `
  --save_to_psd
```

フォルダを指定して複数枚まとめて処理することも可能:

```powershell
.\.venv\Scripts\python.exe inference\scripts\inference_psd.py `
  --srcp path\to\image_folder\ `
  --save_to_psd
```

出力先はデフォルトで `workspace\layerdiff_output\` 以下。各画像ごとに
レイヤー分割済みの `.psd` ファイルと、中間生成物(深度マップ、セグメンテーションマスク)が保存される。

### 2. 低VRAM(8GB)向け: NF4量子化パイプライン ← **このマシンの標準構成**

**動作確認済みの設定 (RTX 3060 Ti 8GB)**:

```powershell
.\.venv\Scripts\python.exe inference\scripts\inference_psd_quantized.py `
  --srcp assets\test_image.png `
  --save_to_psd `
  --cpu_offload `
  --group_offload `
  --resolution 768 `
  --resolution_depth 512
```

実測値 (透過PNG 1枚):

| 指標 | 値 |
|---|---|
| peak VRAM | 4.19 GB / 8 GB |
| LayerDiff | ~207s |
| Marigold | ~10s |
| PSD書き出し | ~1s |
| 合計 | **~240s (約4分)** |

初回実行時、HuggingFaceから LayerDiff3D (NF4量子化版) モデルが自動ダウンロードされる
(数GB、数分かかる)。2回目以降はキャッシュされるので速い。

> **解像度について**: 512px では head レイヤーが空になる(モデルの学習分布外)。
> **実用最低解像度は 768px。**

主なオプション:

| オプション | 意味 |
|---|---|
| `--srcp` | 入力画像(または画像フォルダ)のパス |
| `--save_dir` | 出力先(デフォルト `workspace/layerdiff_output`) |
| `--resolution` | LayerDiff処理解像度(デフォルト1280、**768推奨**) |
| `--resolution_depth` | Marigold深度推定の解像度(デフォルト768、**512推奨**) |
| `--quant_mode` | `nf4`(デフォルト) か `none`(bf16、切り分け用) |
| `--cpu_offload` / `--no_cpu_offload` | モデルCPUオフロード(**`--cpu_offload` 必須**) |
| `--group_offload` / `--no_group_offload` | ブロック単位オフロード(**`--group_offload` 必須**) |
| `--tblr_split` | 手袋・目など一部パーツを左右に分割 |
| `--seed` | 乱数シード |

### 3. block swapパイプライン(bf16のまま8GB狙い、NF4より高速な場合あり)

```powershell
.\.venv\Scripts\python.exe inference\scripts\inference_psd_blockswap.py `
  --srcp assets\test_image.png `
  --save_to_psd
```

### 4. PSD出力後の追加分割(深度ベース / 左右分割)

`inference_psd.py` (または `_quantized` / `_blockswap`) で書き出したPSDを
さらに細かく分割したい場合:

```powershell
# 深度に基づいてパーツを分割(例: 手袋レイヤーを前後で分割)
.\.venv\Scripts\python.exe inference\scripts\heuristic_partseg.py seg_wdepth `
  --srcp workspace\layerdiff_output\test_image.psd `
  --target_tags handwear

# 左右で分割
.\.venv\Scripts\python.exe inference\scripts\heuristic_partseg.py seg_wlr `
  --srcp workspace\layerdiff_output\test_image_wdepth.psd `
  --target_tags handwear-1
```

### 5. UI(データ確認・アノテーション用)

`ui/README.md` を参照。起動には `workspace/datasets/` フォルダ(サンプルデータ)が
リポジトリルートに必要。mmdet階層(`requirements-inference-mmdet.txt`)を
入れていないと起動に失敗する可能性があるので注意(このマシンでは未インストール)。

## 既知の問題

### blockswap版

```
UNetFrameConditionModel.from_pretrained() → Windows access violation
```

`inference_psd_blockswap.py` は Windows でロード段階でクラッシュする。
未調査。NF4 + cpu_offload 構成で代替可能なので当面保留。

## セットアップ完了状態のまとめ

- [x] リポジトリのクローン
- [x] Python 3.12 + venv 作成
- [x] PyTorch (cu128) インストール、CUDA認識確認済み
- [x] requirements.txt / bitsandbytes インストール
- [x] assets ジャンクション作成
- [x] モデルの自動ダウンロード成功
- [x] **透過PNG 1枚の PSD 書き出し完走確認 (NF4 + cpu_offload + group_offload, 768px, ~4分)**
