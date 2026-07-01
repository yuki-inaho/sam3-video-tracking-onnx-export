# LLMオンボーディングサマリー

> このドキュメントは、新任LLMエージェントが本プロジェクトに参加する際の初期資料です。記載内容はリポジトリ内の `README.md`、`justfile`、`pyproject.toml`、テスト資産、作業記録に基づきます。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** `sam3-video-tracking-onnx-export`。SAM3 video tracking のONNX export / ONNX Runtime推論 / Gradio UI。
- **最終成果物:** SAM3のvideo trackingに必要なONNXモジュール群、等価ソース生成、検証用notebook、bbox指定からmask overlayを生成するGradio UI。
- **ビジネス背景・価値:** PyTorch実装に依存せず、ONNX RuntimeでSAM3 video trackingを再現・検証できる実験基盤を作る。GTX 1070のような古いGPUでもCUDAExecutionProvider経由で動作確認できる構成を維持する。
- **現時点の進捗サマリ:** `sam3/` はGit submodule前提。`just build-all` でONNX export一式を生成する設計。`notebooks/sam3_onnx_video_demo.ipynb` がONNX推論例。Gradio UIは `just image-encoder-tracker-fp16` 後に `just webgui` で起動する。GTX 1070では `onnxruntime-gpu==1.18.0` とCUDA11系runtime libs、`image_encoder_tracker_fp16.onnx` を使う。

## 2. クリティカルな要求・制約

> 「壊してはいけない」品質・仕様ラインを箇条書きで列挙します。

- ONNX RuntimeのWeb UI経路はTensorRTを使わない。`CUDAExecutionProvider` を先頭にし、ORTが必要とする場合のみ `CPUExecutionProvider` fallbackを許可する。
- GTX 1070 / 8GB VRAMではfp32の `image_encoder_tracker.onnx` はcuDNN Frontend失敗またはVRAM OOMになりうる。`outputs/onnx/image_encoder_tracker_fp16.onnx` を生成して優先使用する。
- `sam3/` は上流SAM3のsubmoduleとして扱う。上流コードを直接整形・改変しない。
- `.env`、作業者名、絶対ホームパス、個人環境固有のパスをnotebook・README・コード・出力例に混入させない。
- `outputs/`、`temp/`、ONNX成果物、Gradio session出力は生成物として扱う。コミット対象にする前に意図を確認する。
- notebookは `nbqa` とruffで整形・lintする方針。通常のruff対象からは `notebooks` と `sam3` が除外されている。

## 3. 参照すべき合意済み資料

> 新任エージェントが必ず確認すべき一次資料の一覧です。パスと役割を記載します。

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | `README.md` | prerequisites、最小コマンド、notebook、Gradio UI起動手順。 |
| 要件定義書 | `justfile` | export、oracle、inference、test、quality、Web UI関連ターゲットの一次情報。 |
| WBS / 進捗 | `temp/workdoc_May20-2026_sam3_onnx_video_export.md` | ONNX export / video tracking作業の作業書と記録。 |
| テスト資産 | `tests/` | env smoke、ONNX等価性、video e2e、Gradio helper tests。 |
| notebook | `notebooks/sam3_onnx_video_demo.ipynb` | ONNX推論のnotebook例。個人パスを出さないこと。 |
| Web UI entrypoint | `tools/webgui.py`, `src/sam3_onnx_equiv/gradio_app.py` | Gradio起動とbbox指定tracking処理。 |
| GTX 1070対応 | `tools/convert_image_encoder_tracker_fp16.py` | tracker image encoderのfp16変換。 |
| 既知課題リスト | ONNX Runtime issue #23301 | cuDNN Frontend Conv失敗の参考。`onnxruntime-gpu==1.19.0`回避案は本環境ではsegfaultしたため、最終的に1.18.0 + CUDA11 runtime libsを採用。 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- `justfile` に沿ったONNX export、format、lint、test、notebook整形の追試。
- `src/sam3_onnx_equiv/` と `tools/` の小さな修正、Gradio UI helperのテスト追加。
- `README.md`、`docs/ONBOARDING.md`、作業書への事実ベースの追記。
- `temp/` 配下の入力サンプルを使った再現確認。ただし生成物のコミット可否は確認する。

### 任せないタスク（例）

- `sam3/` submoduleの上流コードを無断で直接編集すること。
- `.env`、checkpoint、ONNX成果物、個人データ、絶対ホームパスを不用意にコミットすること。
- TensorRT前提への変更。ユーザー要求はCUDAExecutionProviderでの実行。
- 未検証の性能値、精度値、対応GPU範囲を断定すること。

## 5. インタラクション方針

- **回答スタイル:** 日本語で簡潔に、変更内容・検証結果・残リスクを分けて報告する。
- **回答手順:** まず現状確認、次に実装または修正、最後に実行したコマンドと結果を示す。
- **禁止事項・注意:** 未確認事項を断定しない。ユーザー作業や未コミット差分を勝手に戻さない。長時間動くWeb UIや変換処理はプロセス状態を確認する。
- **秘匿情報の扱い:** `.env`、checkpoint、個人データパス、ユーザー名、ホスト名をドキュメントやnotebook出力に残さない。例示は相対パスにする。

## 6. 試行タスク（オンボーディング演習）

> 小さな検証タスクを2〜3件記載してください。理解度を確認するために実施します。

1. `just --dry-run build-all` と `just --dry-run webgui` を実行し、どのファイル・ディレクトリが入力/出力になるか説明する。
2. `just format-check`、`just lint`、`uv run python -m pytest tests/test_env_smoke.py tests/test_gradio_app.py -q` を実行し、結果を報告する。
3. `outputs/onnx/image_encoder_tracker_fp16.onnx` が無い状態で `just image-encoder-tracker-fp16` を実行し、生成後にGradio UIがどのモデルを優先するか `src/sam3_onnx_equiv/gradio_app.py` から説明する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 実行済みコマンドは `確認済み` として書く。推測や未実行の手順は `未検証` と明記する。
- **TBDの扱い:** 未確認の責任者、データ入手元、checkpoint配布条件、GPU互換範囲は `未確認` とし、推測で補完しない。
- **レビュー/承認フロー:** 未確認。コミット・pushはユーザー指示がある場合のみ行う。
- **その他の運用ルール:** `rtk` 指示がある環境ではshell commandに `rtk` を付ける。生成物や大容量ファイルは `.gitignore` と `git status` を確認して扱う。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** `sam3/`、`src/sam3_onnx_equiv/`、`tools/`、`tests/`、`notebooks/`、`outputs/onnx/`、`outputs/reference/constants/`、`temp/`。
- **代表的なコマンド:** `uv sync --extra dev --group dev`、`just build-all`、`just image-encoder-tracker-fp16`、`just webgui`、`just lint`、`just format-check`、`just format-notebooks`、`just lint-notebooks`。
- **依存ライブラリ:** Python `>=3.10,<3.13`、uv、just、PyTorch `2.7.0` cu126、ONNX、ONNX Runtime `1.18.0`、ONNX Runtime GPU `1.18.0`、Gradio、`gradio-image-annotation`、CUDA11 runtime libs for Web UI。
- **確認済み:** 2026-07-01時点で、GTX 1070環境にて `image_encoder_tracker_fp16.onnx` を使った2フレームGradio相当処理がCUDAExecutionProvider経由で完走し、mask/overlay/zipを生成した。
- **連絡先/責任者:** 未確認。GitHub remoteは `README.md` のclone例を参照。

> ※テンプレートは必要に応じて拡張・縮退して構いません。記入済みのドキュメントはバージョン管理してください。
