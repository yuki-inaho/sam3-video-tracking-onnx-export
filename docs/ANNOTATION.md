# SAM 3.1 画像・動画アノテーション

この機能は、画像または動画に正例・負例のポイントを置き、SAM 3.1 Object
Multiplex でオブジェクトのマスクを作成します。動画では、指定したオブジェクト ID
を保ったまま読み込んだ全フレームへマスクを伝播できます。ブラウザ UI と、画面を
使わない `image` / `video` コマンドの両方を利用できます。

入力メディアとモデルはローカルで処理されます。Web サーバーの既定の待受先は
`127.0.0.1` です。エクスポートされる JSON には入力ファイルのベース名と成果物内の
相対パスだけが入り、入力元の絶対パスや認証情報は入りません。

## 実行アーキテクチャ

推論は、公式 SAM 3.1 とこのリポジトリの ONNX CPU 実装を組み合わせています。

- `sam31/` の公式ソースと `models/sam3.1_multiplex.pt` の公式 checkpoint を使います。
- `outputs/sam31_cpu_source/` は、公式ソースを直接変更せずに作る CPU 対応コピーです。
- 画像特徴抽出、メモリアテンション、Multiplex マスクデコーダー、メモリエンコーダーを
  `outputs/onnx_sam31/` の ONNX モデルで実行します。
- ポイントプロンプトの処理、16 slot の bucket 管理、オブジェクト ID と動画状態の管理は
  公式 PyTorch/Python 実装が担当します。これは単一 ONNX ファイルだけで完結する構成では
  ありません。
- 動画伝播の直前に公式 preflight を実行し、対話フレームの一時出力を統合して
  mask memory を生成します。
- ONNX Runtime は `CPUExecutionProvider` を使います。モデルと ONNX session は最初の
  メディアを開いた時に遅延ロードされます。

ONNX 成果物は次の 7 ファイルです。

```text
outputs/onnx_sam31/
├── trihead_image_encoder.onnx
├── multiplex_mask_decoder.onnx
├── multiplex_memory_encoder.onnx
├── multiplex_memory_attention.onnx
├── multiplex_memory_attention_m1_p2.onnx
├── multiplex_memory_attention_m2_p1.onnx
└── multiplex_memory_attention_m2_p2.onnx
```

メモリ窓は、直前のマスクメモリ最大 1 フレームとオブジェクトポインター最大 2
フレームに固定されています。4 種類のメモリアテンション ONNX は、実際に利用できる
メモリとポインターの組み合わせに対応します。この条件は公式モデルの既定の長い時間窓
とは異なるため、長い動画では公式既定設定と結果が異なる場合があります。

## 準備

必要なものは Python `>=3.10,<3.13`、`uv`、Git submodule、Hugging Face 上の
`facebook/sam3.1` の利用許諾、公式 checkpoint です。Hugging Face への認証はローカルで
済ませ、認証情報をコマンド行、リポジトリ内のファイル、ログ、成果物へ直接書き込まないで
ください。

```bash
git submodule update --init --recursive
uv sync --extra dev --group dev
mkdir -p models
uv run hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir models
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just build-sam31
```

実行前に次の成果物が存在することを確認します。

```text
models/sam3.1_multiplex.pt
outputs/sam31_cpu_source/
outputs/onnx_sam31/
```

`just build-sam31` は CPU 対応ソースの生成と 7 個の ONNX の export を行います。

## ブラウザで操作する

ローカルサーバーを起動します。

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python tools/run_sam31_annotation.py serve
```

表示された `http://127.0.0.1:8765` をブラウザで開き、次の順で操作します。

1. 「画像・動画を選択」からメディアを読み込みます。動画は「最大フレーム数」までを
   先頭から連続してデコードします。
2. オブジェクト ID とラベルを指定します。
3. キャンバスをクリックして正例点を追加します。`Shift` を押しながらクリックするか、
   右クリックすると負例点を追加できます。
4. 「このフレームをセグメント」を押してマスクを確認します。同じ ID でもう一度実行
   すると、そのオブジェクトのポイントとマスクを更新します。
5. 複数の対象には「新しいオブジェクト」を使い、別の ID と少なくとも 1 個の正例点を
   指定します。
6. 動画では「動画全体へ伝播」を押します。タイムラインで各フレームの結果を確認します。
7. 「COCO JSON を書き出す」で JSON をダウンロードします。

伝播後にいずれかのフレームを再セグメントすると、旧い伝播結果は無効化されます。
Web UI は古いマスクを消して書き出しを一時無効にするため、「動画全体へ伝播」を
もう一度実行してください。サーバーが動作中であれば、ブラウザを再読み込みしても
現在のメディアとマスクを復元できます。

画像は Pillow が読める JPEG、PNG、BMP、TIFF、WebP など、動画はローカルの OpenCV が
読める形式を利用できます。アップロード上限は 512 MiB です。作業用コピーと JPEG
フレームは既定で `outputs/annotation_workspace/` に置かれ、次のメディアを開くと置き換え
られます。機微な入力を扱った後は、この作業ディレクトリも管理対象として扱ってください。
同じ workspace を使うサーバーの二重起動は、先行セッションを保護するため拒否されます。

待受先はローカルマシン内に限定されます。port、CPU thread 数、動画の上限は変更できます。

```bash
uv run python tools/run_sam31_annotation.py serve \
  --host 127.0.0.1 --port 8765 --threads 8 --max-video-frames 60
```

## 画面を使わずに実行する

### 画像

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python tools/run_sam31_annotation.py image \
  --input path/to/input.jpg \
  --point 1:0.48:0.52 \
  --point 1:0.68:0.52:0 \
  --label 1:target \
  --output outputs/annotations/example-image
```

### 動画

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python tools/run_sam31_annotation.py video \
  --input path/to/input.mp4 \
  --point 1:0.28:0.44 \
  --point 1:0.10:0.44:0 \
  --point 2:0.72:0.58 \
  --label 1:person \
  --label 2:bag \
  --max-frames 60 \
  --output outputs/annotations/example-video
```

`--point` の形式は `OBJECT_ID:X:Y[:LABEL]` です。

- `OBJECT_ID` は `1` から `16` の整数です。同じ ID の点は 1 つのオブジェクトへまとめ
  られます。
- `X` と `Y` は画像幅・高さに対する正規化座標で、どちらも `0.0` から `1.0` です。
  左上が `(0, 0)`、右下が `(1, 1)` です。
- `LABEL` を省略するか `1` にすると正例点です。`0` は背景として除外する負例点です。
  例: `--point 1:0.68:0.52:0`。
- `--point` と `--label OBJECT_ID:NAME` は繰り返し指定できます。各オブジェクトには
  少なくとも 1 個の正例点が必要です。
- headless コマンドのポイントはすべてフレーム 0 に与えられます。`video` はその後、
  読み込んだ全フレームへ自動で伝播します。

`image` の `--max-frames` の既定値は 1、`video` は 6 です。`video` では指定数までを
先頭から読み込みます。長い動画は実行時間とメモリ使用量が増えるため、必要な範囲を明示
してください。出力先は実行ごとに分けることを推奨します。

## 制限

- 1 セッションでアノテーションできるオブジェクトは最大 16 個です。
- 1 回のセグメントに指定できる正例点・負例点は合計 64 個までです。
- オブジェクトのラベルは 120 文字までです。
- Web UI の動画上限は既定で 60 フレームです。`--max-video-frames`
  を変更すると、UI の最大値もサーバー設定に合わせて更新されます。
- headless `video` の既定値は 6 フレームで、`--max-frames` に 1 以上の整数を指定できます。
- 動画はフレームを間引かず、先頭から上限までを読み込みます。
- 1 フレームは最大 `4096 × 4096` pixel 相当、読み込み全体は最大 600,000,000
  pixel です。例えば 4K UHD の 60 フレームは範囲内です。
- 推論は CPU で順次実行されます。解像度、フレーム数、オブジェクト数に応じて時間と
  メモリ使用量が増えます。
- ポイントプロンプトのみを対象にしています。テキスト、box、mask 入力には対応して
  いません。

## 出力形式

ブラウザの「COCO JSON を書き出す」は `sam31-annotations.json` をダウンロードします。
headless コマンドは、確認用ファイルを含む自己完結したディレクトリを保存します。
ブラウザ版 JSON の `images[].file_name` はサーバーがデコードしたフレーム名で、
JSON 単体に画像は含まれません。入力フレーム、mask PNG、overlay、preview を
まとめて残す場合は headless コマンドを使用してください。

```text
outputs/annotations/example-video/
├── annotations.json
├── media/
│   └── input.mp4
├── frames/
│   ├── 000000.jpg
│   └── 000001.jpg
├── overlays/
│   ├── 000000.jpg
│   └── 000001.jpg
├── masks/
│   ├── object_0001/
│   │   ├── 000000.png
│   │   └── 000001.png
│   └── object_0002/
│       ├── 000000.png
│       └── 000001.png
└── preview.mp4
```

`frames/` はデコード済み入力、`overlays/` は色付きマスクと bbox の確認画像、`masks/`
はオブジェクト別の 8 bit PNG です。`preview.mp4` は 2 フレーム以上の動画でだけ作られ
ます。画像の場合も同じ構造ですが、フレームは `000000.jpg` の 1 枚で、動画 preview は
作られません。

`annotations.json` は COCO の `images`、`annotations`、`categories` を中心に、動画追跡と
再現情報を追加した形式です。構造の要点は次のとおりです。

```json
{
  "info": {"description": "SAM 3.1 image/video annotation", "version": "1.0"},
  "licenses": [],
  "sam3": {
    "version": "3.1",
    "variant": "object-multiplex",
    "runtime": "onnxruntime-cpu",
    "mask_threshold": 0.0
  },
  "media": {
    "name": "input.mp4",
    "kind": "video",
    "width": 4,
    "height": 3,
    "frame_count": 2,
    "fps": 30.0
  },
  "videos": [
    {"id": 1, "file_name": "media/input.mp4", "width": 4, "height": 3,
     "frame_count": 2, "fps": 30.0}
  ],
  "images": [
    {"id": 1, "file_name": "frames/000000.jpg", "width": 4,
     "height": 3, "frame_index": 0, "video_id": 1},
    {"id": 2, "file_name": "frames/000001.jpg", "width": 4,
     "height": 3, "frame_index": 1, "video_id": 1}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "track_id": 1,
      "iscrowd": 0,
      "score": 0.98,
      "bbox": [1, 1, 2, 2],
      "area": 4,
      "segmentation": {"size": [3, 4], "counts": [4, 2, 1, 2, 3]}
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": ""}
  ],
  "objects": [
    {"id": 1, "label": "person",
     "prompts": [{"frame_index": 0, "x": 0.28, "y": 0.44, "label": 1}]}
  ]
}
```

- `bbox` は絶対 pixel の COCO XYWH `[x, y, width, height]`、`area` は前景 pixel 数です。
- `track_id` は入力したオブジェクト ID で、動画の全フレームを通して維持されます。
- `category_id` はラベル文字列ごとに割り当てられます。同じラベルを持つ別オブジェクトは
  同じ category でも異なる `track_id` を持ちます。
- `score` は有限の数値、取得できない場合は `null` です。
- `videos` と各画像の `video_id` は動画だけに含まれます。
- headless の成果物では、`images[].file_name` と `videos[].file_name` は成果物ディレクトリ
  からの相対パスです。ブラウザからダウンロードする JSON では、デコード済みフレーム名と
  入力動画のベース名になります。

`segmentation` は非圧縮 COCO RLE です。`size` は `[height, width]`、`counts` は
column-major（Fortran order）で走査した run length の整数配列です。配列は背景 run から
始まり、背景と前景を交互に数えます。左上 pixel が前景なら最初の背景 run は `0` です。

## テスト

モデルをロードしないアノテーション単体・HTTP 境界テスト:

```bash
uv run python -m pytest \
  tests/test_sam31_annotation.py \
  tests/test_sam31_annotation_server.py -q
```

公式 checkpoint と生成済み ONNX を使う CPU 動画 E2E:

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just sam31-e2e
```

CLI 引数と全体の品質確認:

```bash
uv run python tools/run_sam31_annotation.py --help
just quality
```

## UI の参考資料

画像上への正例・負例ポイント、マスク overlay、画像と動画の操作導線は、
[`yuki-inaho/sam3.cpp` の `develop` ブランチ](https://github.com/yuki-inaho/sam3.cpp/tree/develop)
にある annotator の UX を参考にしています。この記載は操作設計への帰属を示すもので、
コードの同一性、移植、または実装互換性を主張するものではありません。本機能の推論経路は
公式 SAM 3.1 Object Multiplex と、このリポジトリの ONNX Runtime CPU 実装です。
