# 開発オンボーディング

更新: 2026-10-01。対象は `main`。この文書は、新しい開発者や自動化エージェントが、SAM 3 / SAM 3.1 / EfficientSAM3 EV-M の CPU 経路を再現するための入口です。コマンドは、特記がない限りリポジトリルートで実行してください。

## 1. 最初に行うこと

1. `README.md` と本書を読む。
2. `git status --short --branch` で既存変更を確認する。
3. 下のモデル表で経路を選ぶ。SAM 3 / SAM3.1 は公式 submodule、EV-M は固定 source cache を使用する。
4. 選んだ環境を同期する。EV-M は `just efficient-sync`、SAM 3 / SAM3.1 は `just sync`。
5. 使用するモデルの利用条件を確認し、重みを取得する。公式 gated model はアクセス承認が必要。
6. 軽い契約テストから実行する。SAM 3 / SAM3.1 は下記の CPU 固定変数を付ける。
7. ONNX を生成してから、数値比較、動画 E2E、品質ゲートへ進む。
8. commit 前に、生成物と秘密情報が staged diff に含まれていないことを確認する。

利用できるコマンドは次で確認できます。

```bash
just --list
```

## 2. プロジェクト概要

このリポジトリは、公式 SAM 3 と SAM 3.1 Object Multiplex の動画追跡を、ONNX Runtime の CPUExecutionProvider で実行・検証するための実装です。

| モデル | 環境・取得経路 | 画像列の意味 | 詳細 |
| --- | --- | --- | --- |
| SAM 3 | root uv project / `sam3/` submodule / `models/sam3.pt` | memory-bank tracking | 本書7.1 |
| SAM3.1 | root uv project / `sam31/` submodule / `models/sam3.1_multiplex.pt` | Object Multiplex tracking | [ANNOTATION](ANNOTATION.md) |
| EfficientSAM3 EV-M | CPU `efficientsam3/.venv` / pinned cache / `models/efficientsam3_ev_m.pt` | フレームごとの独立検出 | [EFFICIENTSAM3](EFFICIENTSAM3.md) |

EV-M は EfficientViT **b1** + MobileCLIP **S0**、context **16**、入力 **1008×1008**。
公開 checkpoint は detector のみで、memory tracker を持ちません。query ID は追跡 ID ではありません。
native C++/GGUF 版は [sam3.cpp の ONBOARDING](https://github.com/yuki-inaho/sam3.cpp/blob/develop/docs/ONBOARDING.md) を参照します。

### SAM 3

```text
公式 SAM 3 source + checkpoint
  -> CPU向け生成source
  -> image / decoder / memory ONNX群
  -> Python memory bank
  -> PyTorch oracleとの動画比較
```

### SAM 3.1 Object Multiplex

```text
公式 SAM 3.1 source + multiplex checkpoint
  -> CPU向け生成source
  -> TriHead image encoder
  -> multiplex mask decoder
  -> multiplex memory encoder
  -> bounded memory-attention ONNX群
  -> 公式bucket・slot・時間状態
  -> 公式PyTorchとの複数物体動画比較
```

SAM 3.1 の初回対話プロンプトと bucket / slot / 時間状態は、公式 PyTorch / Python 実装が管理します。画像特徴と動画伝播の重いテンソル境界を ONNX Runtime で実行します。

## 3. 主要なファイルと責務

| パス | 用途 |
| :--- | :--- |
| `README.md` | 利用者向けの最短セットアップと実行例 |
| `justfile` | source生成、export、oracle、テスト、品質ゲート |
| `pyproject.toml` / `uv.lock` | Python依存関係と品質設定 |
| `sam3/` | 固定された公式SAM 3 source。直接編集しない |
| `sam31/` | 固定された公式SAM 3.1 source。直接編集しない |
| `src/sam3_onnx_equiv/` | source変換、モデル読込、ONNX wrapper、動画orchestrator |
| `src/sam3_onnx_equiv/annotation.py` | ポイント推論、COCO RLE、overlay、画像・動画伝播 |
| `src/sam3_onnx_equiv/annotation_server.py` | ローカルアノテーション API と成果物保存 |
| `src/sam3_onnx_equiv/export/` | ONNX境界とexport実装 |
| `tools/` | source生成、export、oracle、推論CLI |
| `web/annotation/` | 画像・動画アノテーション UI |
| `docs/ANNOTATION.md` | アノテーションの操作、出力形式、制限 |
| `tests/` | 契約、checker、数値比較、動画E2E |
| `efficientsam3/` | EV-M の CPU uv project、固定取得、strict load、export、torch非依存runtime、tests |
| `docs/efficientsam3-validation/` | 公開checkpointのinventory、ロード検査、ORT/PyTorch E2E証跡 |
| `notebooks/` | SAM 3 ONNX動画デモ |
| `models/` | 公式checkpoint。Git管理外 |
| `outputs/` | 生成source、ONNX、oracle、推論結果。Git管理外 |
| `logs/` | 実行ログ。Git管理外 |
| `temp/` | 一時調査・作業記録。Git管理外 |

公式 submodule に CPU 固有変更を直接加えないでください。`sam3_source_patcher.py` と `sam31_source_patcher.py` が、別ディレクトリにレビュー可能な生成コピーを作ります。

## 4. 前提条件

- Python `>=3.10,<3.13`
- EV-M の独立 project は Python 3.11 または3.12
- `git`
- `uv`
- `just`
- Hugging Face アカウントと、使用する公式モデルへのアクセス権
- 十分なディスク容量とメモリ

checkpoint、Python環境、両版のONNXを含めると、20〜25 GB程度の空き容量を見込んでください。CPU export と E2E は大きなモデルをロードするため、重い処理を並列実行しないでください。

## 5. セットアップ

### 5.1 clone と submodule

```bash
git clone --recursive <repository-url>
cd <repository-directory>
git submodule update --init --recursive
```

既存checkoutでは、作業前に次を確認します。

```bash
git status --short --branch
git submodule status
```

### 5.2 Python環境

以下は SAM 3 / SAM3.1 用です。root の lock は CUDA 対応torch wheelを含みます。
CPU専用 EV-M を使う場合は、このroot環境を同期せず、7.5の独立projectを使います。

```bash
just sync
uv run python -c "import torch, onnx, onnxruntime; print(torch.__version__); print(torch.cuda.is_available()); print(onnxruntime.get_available_providers())"
```

lockfileを変更せず厳密に同期したい場合は、次を使います。

```bash
uv sync --locked --extra dev --group dev
```

### 5.3 CPU固定

共有GPUを使用しない実行では、各コマンドに次を付けます。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 <command>
```

CPU thread数も制限する場合は、同じコマンドに次を追加します。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
```

PyTorch が CUDA 対応wheelでも、`torch.cuda.is_available()` が `False`、ONNX session が `CPUExecutionProvider` であればCPU経路です。公式コードのautocast decoratorに由来する「CUDAを無効化した」というwarningは表示される場合があります。

## 6. 公式checkpointの取得

SAM 3 / SAM 3.1 は gated model です。Hugging Face上で各モデルの利用条件を承認してから、次を実行します。

```bash
uv run hf auth login
mkdir -p models
uv run hf download facebook/sam3 sam3.pt --local-dir models
uv run hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir models
```

配置契約は次のとおりです。

| モデル | checkpoint |
| :--- | :--- |
| SAM 3 | `models/sam3.pt` |
| SAM 3.1 | `models/sam3.1_multiplex.pt` |

認証トークンをコマンド引数、Markdown、ログ、commit messageへ書かないでください。トークン値を表示するコマンドも実行しません。`models/` はGit管理外です。

## 7. 標準ワークフロー

### 7.1 SAM 3

SAM 3 の動画E2Eには、ONNXとPyTorch oracleの両方が必要です。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just build-all
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just oracle
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just e2e
```

主な生成物:

- `outputs/sam3_equiv_source/`
- `outputs/onnx/`
- `outputs/reference/baseline_detector.npz`
- `outputs/reference/video_oracle_all.npz`

### 7.2 SAM 3.1

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just build-sam31
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just sam31-e2e
```

`just build-sam31` は、CPU向けsource生成と7個のONNX exportを順番に実行します。分割して調査するときは次を使います。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just sam31-source
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just export-sam31-all
```

主な生成物:

- `outputs/sam31_cpu_source/`
- `outputs/onnx_sam31/trihead_image_encoder.onnx`
- `outputs/onnx_sam31/multiplex_mask_decoder.onnx`
- `outputs/onnx_sam31/multiplex_memory_encoder.onnx`
- `outputs/onnx_sam31/multiplex_memory_attention*.onnx`

### 7.3 SAM 3.1 動画CLI

入力はファイル名順の画像フォルダです。時系列が明確になるよう、ゼロ埋めした名前を使ってください。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python tools/run_sam31_video.py \
  --video-dir <frames> \
  --point 1:0.25:0.35 \
  --point 2:0.70:0.65 \
  --output outputs/reference/sam31_video.npz
```

`--point` は `object_id:x:y` で、座標は画像幅・高さに対する `[0, 1]` の比率です。`--max-frames` で短いsmoke runにできます。

### 7.4 SAM 3.1 画像・動画アノテーション

`just build-sam31` 完了後、ローカル Web UI は次で起動できます。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just annotate
```

既定の `http://127.0.0.1:8765` を開き、ポイントからマスクを作成します。動画では先頭フレームのオブジェクト ID を保ったまま全フレームへ伝播します。ブラウザを閉じてもサーバーが動いていれば、再度開いたときに現在のセッションを復元します。

画面を使わない画像・動画 CLI、負例点、COCO JSON、mask PNG、overlay、preview MP4 の詳細は [`docs/ANNOTATION.md`](ANNOTATION.md) を参照してください。軽量テストは次で実行します。

```bash
just test-annotation
```

### 7.5 EfficientSAM3 EV-M：取得 → ONNX → annotation

公式 submodule の初期化はこの経路には不要です。`fetch` が公開 source / checkpoint を固定取得します。

```sh
just efficient-sync
just efficient-test
just efficient-fetch
just efficient-export
just efficient-run --images outputs/efficientsam3/source/sam3/assets/dog_person.jpeg \
  --text dog --threads 4 --output outputs/efficientsam3/annotation
```

画像列は `--images frame0.png frame1.png frame2.png` の明示順で処理します。
出力は `annotations.json` と `frame_000000/mask_*.png` 等。frame index、元画像寸法、score、pixel box、時間を保存します。
実取得checkpointは799 tensors、実測97,435,030 parameters。上流README表の89.2Mとは異なります。
不足・余分なキー、shape、dtype、非有限値を拒否し、strictロード後に vision / text / grounding の3 graphをexportします。

`runtime.py` はtorchをimportせず、ORT CPUのみで推論します。同じpromptのtext出力はcacheします。
画像はRGB → Pillow bilinear → CHW float32 → `pixel/127.5-1`。参照比較にも同じ前処理を渡します。
manifest はvariant、context、解像度、checkpoint SHA、sequence modeを照合します。

モデルなし検査は `just efficient-test`、source・重み・ONNX配置後の実モデル6frame比較は次です。

```sh
just efficient-e2e
```

2026-10-01 の [実モデルreport](efficientsam3-validation/onnx-e2e.json) は全6frame mask IoU **1.0**。
finite、同じ選択query、非空mask、IoU >=0.90 を要求します。通常testのskipを実モデル成功として数えません。

## 8. 検証の進め方

高負荷テストを繰り返す前に、軽い検査から進めます。

### 8.1 sourceと環境

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python -m pytest \
  tests/test_env_smoke.py \
  tests/test_source_patcher.py -q
```

### 8.2 SAM 3.1契約とbucket

公式重みと生成sourceの配置後に実行します。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python -m pytest \
  tests/test_sam31_contract.py \
  tests/test_sam31_multiplex_state.py -q
```

### 8.3 export checkerと数値比較

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python -m pytest tests/test_sam31_export.py -q
```

このテストは全7 ONNXを `onnx.checker` に通し、公式PyTorchとONNX Runtime CPU出力を比較します。

### 8.4 全体ゲート

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just test
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just e2e
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just sam31-e2e
just quality
git diff --check
```

`just test` と動画E2EはCPUでは時間がかかります。実装変更に応じた対象テストを先に通し、最終確認で全体を実行してください。

EV-Mだけの変更は `just efficient-test` と対象ファイルのruff検査を入口にし、export/runtime変更時に `just efficient-e2e` を実行します。
新しい環境で全体testを実行すると、checkpointや生成source未配置のテストがskip・失敗する場合があります。必要物を配置して再実行し、結果を分けて記録してください。

### 8.5 性能計測とブラウザ確認

model SHA、入力順・寸法、前処理、provider、thread数、反復回数を揃えて比較します。
`annotations.json` のstage時間と、load・I/Oを含むプロセス全体時間は別に記録します。
公開EV-MのORT約1.35〜1.51秒/frameはCPU16threadsの過去の観測で、環境をまたぐ速度保証ではありません。

SAM3.1 UI変更時はサーバーを起動し、headless Playwrightで画像/画像列upload、点追加、伝播、フレーム移動、reload、COCO downloadを確認します。
console/network errorとmaskの見た目も確認し、browser結果とmodel-free HTTP testsを区別します。

## 9. SAM 3.1 の実装契約と制約

- Object Multiplexは1 bucketあたり16 object slotです。
- ONNX動画モードは、1つ前のmask-memory frameと最大2つのobject-pointer frameを使います。
- memory-attentionは、実際のmemory/pointer長に対応する4つの固定shape graphから選択します。
- 公式PyTorchとの比較では、両経路に同じ固定windowを設定します。
- 公式modelの既定windowはより長いため、長尺動画では出力が異なる場合があります。
- 初回のpoint/mask prompt処理と状態管理は公式PyTorch/Python、画像特徴と伝播はONNXです。
- ONNX sessionは `CPUExecutionProvider` を明示します。providerの暗黙fallbackを前提にしません。
- SAM 3 とSAM 3.1のsource、checkpoint、ONNX出力先を混在させません。

## 10. 変更時の基本手順

1. 作業開始前に `git status --short --branch` を確認する。
2. 公式submoduleではなく、patcher・wrapper・tool・testを変更する。
3. 挙動変更は、対象契約を表す失敗テストを先に用意する。
4. 最小テストを通してから、export parityと動画E2Eへ進む。
5. `just quality` と `git diff --check` を通す。
6. staged fileを限定し、秘密情報と大容量artifactがないことを確認する。

環境変数でパスを変更できます。

| 用途 | 変数 |
| :--- | :--- |
| SAM 3 source | `SAM3_SRC` |
| SAM 3 checkpoint | `SAM3_CHECKPOINT` または `CHECKPOINT` |
| SAM 3 ONNX | `SAM3_ONNX_DIR` または `ONNX_DIR` |
| SAM 3.1 source | `SAM31_SRC` |
| SAM 3.1生成source | `SAM31_CPU_SOURCE` |
| SAM 3.1 checkpoint | `SAM31_CHECKPOINT` |
| SAM 3.1 ONNX | `SAM31_ONNX_DIR` |

値は端末の環境やGit管理外の設定から渡し、個人の絶対パスをtracked fileへ書かないでください。

## 11. 公開・秘密情報の境界

Gitへ含めるもの:

- source code
- tests
- `justfile`、`pyproject.toml`、`uv.lock`
- 秘密情報を除いた再現手順

Gitへ含めないもの:

- Hugging Face token、cookie、credential
- `.env`、shell設定、認証設定
- 個人名、メールアドレス、内部hostname、端末固有の絶対パス
- model checkpoint、ONNX、NPZ、画像・動画annotation payload
- `models/`、`outputs/`、`logs/`、`temp/`
- セッション transcript や生の作業ログ

commit前に、必ず明示したファイルだけをstageし、差分を目視します。

```bash
git status --short
git diff --cached --stat
git diff --cached
```

追加行に秘密らしい値や絶対パスがないことも検査します。

```bash
git diff --cached --no-ext-diff --unified=0 | \
  rg '^\+' | rg -v '^\+\+\+' | \
  rg -q '(/home/|/Users/|BEGIN [A-Z ]*PRIVATE KEY|HF_TOKEN=|hf_[A-Za-z0-9]{20,})'
```

最後の `rg` が終了コード0ならヒットがあります。内容を表示・共有せず、該当ファイルをunstageして安全な相対パスやプレースホルダへ置き換えてください。
source archiveとモデルは別に梱包し、tar owner名・環境cacheを除外します。展開後のファイル数、SHA、PNGとJSONの対応も確認します。
作業終了時の `uv cache prune` は `uv run` の子として起動せず、端末から直接実行します。

## 12. トラブルシューティング

### submodule内のファイルがない

```bash
git submodule update --init --recursive
git submodule status
```

行頭が `-` のsubmoduleは未初期化です。

### Hugging Faceから401 / 403になる

モデルの利用条件が承認済みか確認し、`uv run hf auth login` をやり直します。トークンをissue、チャット、ログへ貼らないでください。

### checkpointが見つからない

`models/sam3.pt` と `models/sam3.1_multiplex.pt` の名前を確認します。別の場所を使う場合は、対応する環境変数を明示します。

### ONNXが見つからない

SAM 3 は `just build-all`、SAM 3.1 は `just build-sam31` を先に実行します。SAM 3 E2Eでは `just oracle` も必要です。
EV-M は `just efficient-fetch` → `just efficient-export`。別ディレクトリのgraphは `just efficient-run --models <directory>` で指定します。

### EV-MのSHA / manifest / strict loadが失敗する

固定revisionのEV-M Stage3 checkpointとsourceを使っているか確認します。variant・context・shapeの検査を緩めて別モデルを通さないでください。

### CPU実行でプロセスが終了する

大きなexport、全テスト、E2Eを同時に実行していないか確認します。空きメモリとディスク容量を確認し、対象exportや対象テストを1つずつ実行します。

### `ty` がwarningを表示する

公式SAM 3の動的module属性やONNX Runtimeの型情報に由来するwarningがあります。`just quality` の終了コードを確認し、新しいerrorをwarning設定へ追加して隠さないでください。

## 13. オンボーディング完了チェックリスト

- [ ] `README.md` と本書を読んだ
- [ ] モデル・対応範囲・画像列の意味を選んだ
- [ ] 作業ツリーと対象sourceの状態を確認した
- [ ] 選んだ環境の `just sync` または `just efficient-sync` が成功した
- [ ] CPU経路を確認した
- [ ] モデルの利用条件と必要なアクセス承認を確認した
- [ ] checkpointを規定パスへ取得した
- [ ] 対象版のsource生成とONNX exportが成功した
- [ ] checker / parity testが成功した
- [ ] 対象版の追跡または独立検出6frame E2Eが成功した
- [ ] 対象品質ゲートと `git diff --check` が成功した
- [ ] checkpoint、生成物、ログ、秘密情報をstageしていない

## 14. 参照と更新履歴

関連リポジトリ FlashVSR / ZipMap のONBOARDINGを参考に、入口・責務・環境・実行・検証・トラブル対応の構成を採用しました。
内容は本repoの `justfile`、CLI、uv lock、検証JSONに照合しています。

- [公式SAM3](https://github.com/facebookresearch/sam3): model演算とObject Multiplexの参照
- [EfficientSAM3固定source](https://github.com/SimonZeng7108/efficientsam3/tree/bd0936c788fed8d51fa799437f05abd97b401b06): EV-M builder
- [native C++/GGUF版](https://github.com/yuki-inaho/sam3.cpp/blob/develop/docs/ONBOARDING.md): 同じ公開EV-Mの別backend

2026-10-01: EV-Mを最初のモデル選択へ統合し、CPU独立環境、strict load、入力順、実モデルE2E、公開・性能の契約を整理。
