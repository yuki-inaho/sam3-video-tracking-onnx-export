# EfficientSAM3 EV-M（CPU / ONNX）

公開 Stage3 EV-M を、画像または明示した順序の画像列へ適用する。
この checkpoint は detector のみを含む。画像列は各フレームの独立検出であり、SAM3.1 のメモリ追跡とは別の機能。

## 固定モデル

- [upstream source](https://github.com/SimonZeng7108/efficientsam3/tree/bd0936c788fed8d51fa799437f05abd97b401b06)
- [HF checkpoint](https://huggingface.co/Simon7108528/EfficientSAM3/tree/85b05896928f974e308f889d7ccb2eefc069de98/efficientsam3_ft): `efficientsam3_efficientvit.pt`
- SHA256: `086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17`
- EfficientViT **b1**、MobileCLIP **S0**、context **16**、image **1008×1008**。
- 実ロード：799 tensors、97,435,030 parameters。公開 README の表は89.2M。ここでは取得した checkpoint の実測値を記載する。

## 取得から推論まで

リポジトリのルートで実行する。uv/just と Python3.11または3.12を使用する。
root の CUDA 環境とは別に `efficientsam3/.venv` へ CPU torch を同期する。

```sh
just efficient-sync
just efficient-fetch
just efficient-export
just efficient-run --images outputs/efficientsam3/source/sam3/assets/dog_person.jpeg --text dog --output outputs/efficientsam3/annotation
```

`fetch` は source revision とモデル SHA256 を検証する。`export` は構築したモデルの全799キー・shape・dtypeを照合し、strictロード後に3つのONNX graphとBPE tokenizer、manifestを書き出す。upstreamの部分ロードhelperは使用しない。

画像列は `--images frame0.png frame1.png frame2.png` と順序を指定する。
出力はフレーム別mask PNGと `annotations.json`。query ID は検出クエリの番号であり、フレームをまたぐ物体IDではない。
`--threads 16`、`--threshold 0.5` を変更できる。入力batch、モデル解像度、contextは固定。

## 検証

```sh
just efficient-test
just efficient-e2e
```

通常テストはモデル取得を要求しない。実モデルE2Eは明示的に有効化し、公開dog画像を移動した6フレームについてPyTorchとORTを比較する。
必須条件はfinite、同じ選択query、非空mask、mask IoU >=0.90。
参照・ORTのstage差分、時間、mask PNGは実行出力へ保存する。
実検証の [report](efficientsam3-validation/onnx-e2e.json) では6フレームすべてIoU1.0、ORT約1.35–1.51秒/フレーム（CPU16threads）。

## 実行境界

`runtime.py` はtorchをimportせず、ONNX Runtime CPUのみで推論する。
入力画像はRGB、Pillow bilinear resize、CHW float32、`pixel/127.5-1`。
E2EのPyTorch参照にも同じ前処理を入力し、preprocessing差とモデル差を区別する。
text/tokenizer処理をキャッシュするため、同じpromptの画像列でテキストエンコーダーを再実行しない。

学習時metadata、利用者の絶対パス、認証tokenはmanifest/reportへ保存しない。
モデル、ONNX、source cache、入力画像はgit管理外。ライセンスは取得先のSAM3・EfficientViT・MobileCLIPとcheckpointの条件を参照する。
