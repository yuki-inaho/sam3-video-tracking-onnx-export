# EfficientSAM3 EV-M 作業計画書 兼 記録書

**日付：** 2026年10月01日
**作業ディレクトリ・リポジトリ：** 隣接する `sam3.cpp` と `sam3-video-tracking-onnx-export` の各ルート。公開 origin は yuki-inaho の同名 GitHub リポジトリ。
**作業者：** Codex（ユーザー指定に従い単独で作業）
**正本：** `sam3.cpp/temp/workdoc_Oct01-2026_efficientsam3_ev_m.md`。ONNX リポジトリには区切りごとに同内容を同期する。
**ブランチ：** 両方 `feat/efficientsam3-ev-m`。基点は C++ develop `89e1300`、ONNX main `d0ac94f`。

## 1. 作業目的

### 1.1 ゴール要求分析

- **直観的な目的：** HF で公開された EfficientSAM3 EV-M の実重みを取得し、既存の ONNX 系と C++/ggml 系の双方で使えるようにする。変換、推論、検証を一貫した手順で再現できる状態を完成とする。
- **明示要求：** SAM3.1 完了後に着手、write/review スキルで作業書を追記、E2E・改善・リファクタリング・実行時間計測、区切りごとの commit/push、PR 作成とマージ。
- **制約：** Python は uv。ONNX の justfile を利用する。C++ は CLAUDE.md と PLAN.md の structs/free functions/C++14/段階ごとの ggml graph 方針に従う。秘密・ローカル利用者情報は公開しない。モデルは git 管理外。推論バックエンドの暗黙 fallback、欠落重みをランダム値のまま成功扱いすることは禁止。
- **範囲：** 指定表の EV-M Stage3 モデルを優先。ほかのサイズや再学習は対象外。動画の機能範囲は公開チェックポイントに追跡重みがあるか調査して確定する。フレーム独立検出とメモリ追跡を混同しない。
- **成功条件：** 実重みの全必須 tensor がロードされ、ONNX と C++ が参照実装と比較して画像および複数画像を処理する。マスク IoU >=0.90、有限値、入力依存、バックエンド実行の証拠、所要時間を保存する。出力が空の場合は別の実画像で有効な検出を得て比較する。
- **前提・リスク：** README の EV-M は EfficientViT b1。テキスト種別・context 長・tracker 有無は実重みと builder から確定する。SAM3.1 のモデルを EfficientSAM3 に代用しない。ONNX root の uv は CUDA index のため、この追加機能には独立した CPU uv プロジェクトを作り、依存衝突と不要な GPU wheel を避ける。

### 1.2 サブゴール構造

| ID | サブゴール | 成果物 | 検証 |
| --- | --- | --- | --- |
| SG1 | モデル仕様確定 | tensor inventory、固定 source/HF revision、設計 | SHA256、必須/余分キーの照合 |
| SG2 | ONNX 対応 | CPU uv、export、runtime、テスト | 実重み PyTorch/ORT 比較 |
| SG3 | C++ 対応 | GGUF converter、native graph、CLI | 重み照合、段階比較、実マスク |
| SG4 | 再現と公開 | docs、E2E、時間、zstd、PR | restore 検査、CI、merge SHA |

### 1.3 トレーサビリティ方針

| Trace | 要求 | 手順 | 証跡 |
| --- | --- | --- | --- |
| E1 | 指定 EV-M 実モデル | 1–5 | `outputs/efficientsam3/inventory.json` |
| E2 | ONNX 推論 | 6–9 | export manifest、ORT oracle report |
| E3 | native C++/ggml 推論と GGUF | 10–13 | conversion/graph/mask report |
| E4 | 画像・複数画像 E2E と時間 | 9、13–15 | `docs/efficientsam3-validation/` |
| E5 | self-consistent な案内と公開 | 16–20 | docs、SHA256、PR、work log |

## 2. 作業内容

### フェーズ1：調査（SG1/E1）

両リポジトリの規約と entry points、公開 EV-M の実 tensor、EfficientViT/MobileCLIP/decoder/segmentation の仕様を確認する。環境の準備と読み取り検査のみを行い、推論コードの変更は後のフェーズとする。

### フェーズ2：設計（SG1–3/E1–3）

checkpoint loading を strict に検証できるモデル構築方法、段階別入出力、GGUF metadata と名前の対応、動画機能範囲、既存 API との接点を記録する。未確定事項は調査結果で埋め、実装前に再レビューする。

### フェーズ3：実装（SG2–3/E2–3）

ONNX の CPU 専用 uv project/export/runtime を追加する。C++ の既存 PCS stage を再利用し、学生 encoder と重み loader を追加する方針を優先する。方式変更は設計に理由を追記する。新しい境界条件を先にテストし red/green を保存する。

### フェーズ4：検証（SG4/E4）

実重みの stage 比較、画像・連続画像 E2E、有限値/入力依存/再実行、既存機能回帰、計測を行う。品質基準未達を完了扱いにしない。

### フェーズ5：公開（SG4/E5）

docs/ONBOARDING.md と機能 guide を更新、公開情報検査、zstd 梱包復元検査、両 repo commit/push/PR/merge を実施する。実モデル・原本 Downloads は削除せず、今回作った一時 staging を片付け uv cache prune を行う。

## 3. 作業チェックリスト

### フェーズ1：調査

### 手順 1: 既存コードと固定参照の確認（E1）
- [x] 🖐 **操作**: 両 repo の規約/justfile と EfficientSAM3 source/HF revision を読む。
- [x] 🔎 **確認**: source `bd0936c788fed8d51fa799437f05abd97b401b06`、HF `85b05896928f974e308f889d7ccb2eefc069de98`、EV-M b1 を特定。
- [x] 🧪 **テスト**: 読み取り調査。既存 SAM31 CTest と ONNX pytest を回帰対象とする。
- [x] 🛠 **エラー時対処**: README と example が異なる場合は実重みの shapes を優先して次の inventory に残す。

### 手順 2: CPU uv 環境の作成（E1/E2）
- [x] 🖐 **操作**: ONNX repo の `efficientsam3/pyproject.toml` に CPU torch/torchvision と export/test 依存を定義し `uv sync --project efficientsam3` を実行する。
- [x] 🔎 **確認**: CUDA wheel に依存せず torch、ONNX、ORT、pytest を import できる。
- [x] 🧪 **テスト**: `uv run --project efficientsam3 python -c 'import torch, onnx, onnxruntime; print(torch.__version__)'`。
- [x] 🛠 **エラー時対処**: CPU index の互換 torch/vision を固定する。未確認の root uv sync を実行しない。

### 手順 3: 実重み inventory の保存（E1）
- [x] 🖐 **操作**: `weights_only=True` で checkpoint の tensor keys/shapes/count を `outputs/efficientsam3/inventory.json` に保存する。
- [x] 🔎 **確認**: model SHA256 `086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17`、サイズ468394477 bytes、学生モデルと tracker 有無が分かる。
- [x] 🧪 **テスト**: checksum 不一致なら export を拒否する仕様を設計する。pickle unrestricted load を行わない。
- [x] 🛠 **エラー時対処**: checkpoint の wrapper は dict のみ受理し metadata の private 値を出力しない。

### 手順 4: 参照モデルの必須重み検査（E1）
- [ ] 🖐 **操作**: 固定 source から参照モデルを構築し必須キー/shape/余分キーを照合する。
- [ ] 🔎 **確認**: text 種別/context、vision neck、共有 PCS、tracker 有無を確定する。
- [ ] 🧪 **テスト**: 必須キーを1つ除いた入力が失敗する。strict=False の warning だけで成功としない。
- [ ] 🛠 **エラー時対処**: 構築依存は import trace を見て CPU project に必要分だけ追加する。

### フェーズ2：設計

### 手順 5: 実装仕様の追記と再レビュー（E1/E2/E3）
- [ ] 🖐 **操作**: 本書の第8章へ入出力、GGUF mapping、動画機能範囲、具体的な追加テストを記載する。
- [ ] 🔎 **確認**: 実装順、段階別比較、失敗条件が source と inventory に一致する。
- [ ] 🧪 **テスト**: review-written-workdoc の再判定が PASS、未確定の必須仕様がない。
- [ ] 🛠 **エラー時対処**: tracker が無い場合は frame sequence 検出を明示し、メモリ追跡の要件は追加重み/設計を別項目で定義する。

### フェーズ3：実装

### 手順 6: ONNX loading 境界テスト（E2）
- [ ] 🖐 **操作**: `efficientsam3/tests/test_contract.py` に不足キー/shape/誤モデルの拒否テストを追加する。
- [ ] 🔎 **確認**: 未実装の loader に対してテストが red になる。
- [ ] 🧪 **テスト**: `uv run --project efficientsam3 pytest efficientsam3/tests/test_contract.py -q`。
- [ ] 🛠 **エラー時対処**: テスト収集失敗と期待する仕様失敗を区別する。

### 手順 7: ONNX exporter の実装（E2）
- [ ] 🖐 **操作**: `efficientsam3/export.py` に固定仕様の段階別 export と manifest を実装する。
- [ ] 🔎 **確認**: 全必須重み検査に合格したものだけを export する。
- [ ] 🧪 **テスト**: 手順6が green。実 export の ONNX checker が成功する。
- [ ] 🛠 **エラー時対処**: 未対応 operator を source の等価実数演算へ明示変換し変更を記録する。

### 手順 8: ORT runner の実装（E2）
- [ ] 🖐 **操作**: `efficientsam3/runtime.py` に image と frame sequence 推論を実装する。
- [ ] 🔎 **確認**: runtime は torch を import せず ORT CPU session で処理する。
- [ ] 🧪 **テスト**: frame順/空入力/サイズ不正/誤manifestを拒否するテストが green。
- [ ] 🛠 **エラー時対処**: stage の layout と dtype を manifest と照合する。

### 手順 9: ONNX 実重み比較（E2/E4）
- [ ] 🖐 **操作**: `efficientsam3/tests/test_real_e2e.py` を実行し oracle report を保存する。
- [ ] 🔎 **確認**: 画像と6フレームで非空の有効なマスク、参照との IoU >=0.90、finite、時間を確認。
- [ ] 🧪 **テスト**: stage差分と最終 mask 比較を両方保存。モデル未配置は明示 skip し実検証完了とは扱わない。
- [ ] 🛠 **エラー時対処**: 最初にずれる stage を特定して手順7/8を修正する。

### 手順 10: GGUF 変換境界テスト（E3）
- [ ] 🖐 **操作**: C++ repo に `tests/test_efficientsam3_conversion.py` を追加する。
- [ ] 🔎 **確認**: モデル種別/必須tensor/shape/metadata 不正の red を観測する。
- [ ] 🧪 **テスト**: CPU uv 環境を明示して pytest を実行する。
- [ ] 🛠 **エラー時対処**: synthetic fixture と trained model を明確に分けて期待値を作る。

### 手順 11: GGUF converter の実装（E3）
- [ ] 🖐 **操作**: `convert_efficientsam3_to_gguf.py` に重みと仕様の保存を実装する。
- [ ] 🔎 **確認**: runtime が必要な全 tensor を保存、metadata に variant/source revision/context を記録する。
- [ ] 🧪 **テスト**: 手順10が green、実重み tensor payload の照合成功。
- [ ] 🛠 **エラー時対処**: 意図的な BN fusion 等は式と数値比較を記録、黙って tensor を除外しない。

### 手順 12: C++ 学生 encoder と loader の実装（E3）
- [ ] 🖐 **操作**: `sam3.cpp`/`sam3.h` に EfficientSAM3 のロードと段階別 ggml graph を追加する。
- [ ] 🔎 **確認**: vision/text/PCS 必須重みを検証してから compute する。Python/ORT 推論への委譲がない。
- [ ] 🧪 **テスト**: operator/stage 参照比較と malformed GGUF 拒否、既存 build を確認する。
- [ ] 🛠 **エラー時対処**: shape/layout の小さい oracle から切り分け、graph ごとに CPU buffer を受け渡す。

### 手順 13: C++ 実モデル image/sequence E2E（E3/E4）
- [ ] 🖐 **操作**: headless CLI を使い同じ実画像/6フレームで native output を保存する。
- [ ] 🔎 **確認**: PyTorch/ORT と mask IoU >=0.90、finite、入力依存、再現性を満たす。
- [ ] 🧪 **テスト**: 外部プログラムを起動できない PATH で native 推論を実行して証明する。
- [ ] 🛠 **エラー時対処**: 共有 PCS stage と学生 stage を別々に比較して修正する。

### フェーズ4：検証

### 手順 14: 性能計測（E4）
- [ ] 🖐 **操作**: 同一入力/スレッド数で stage時間/全体時間/モデル容量を計測する。
- [ ] 🔎 **確認**: 実測値が report にあり、最適化した場合は同一結果と前後時間がある。
- [ ] 🧪 **テスト**: 修正後の数値/マスク比較を再実行する。
- [ ] 🛠 **エラー時対処**: ボトルネックを特定してから変更する。近似を無断追加しない。

### 手順 15: 回帰・品質ゲート（E4）
- [ ] 🖐 **操作**: 各 repo の既存 model-free tests、新規 tests、CMake build、対象 lint/format を実行する。
- [ ] 🔎 **確認**: SAM31 CTest `-LE real` と ONNX 新規 tests が成功し、既存 API の互換を確認する。
- [ ] 🧪 **テスト**: 実行 command と exit code を保存する。
- [ ] 🛠 **エラー時対処**: 既存環境の不足と今回の回帰を分離して記録する。

### フェーズ5：公開

### 手順 16: 導入手順の更新（E5）
- [ ] 🖐 **操作**: 両 repo の `docs/ONBOARDING.md` と EfficientSAM3 guide に download→convert/export→image/sequence→test を記載する。
- [ ] 🔎 **確認**: パス/モデル/variant が一貫し、空環境で必要な source/deps が分かる。
- [ ] 🧪 **テスト**: docs のコマンドを clean output directory で実行確認する。
- [ ] 🛠 **エラー時対処**: 固定 revision と public URL を用い、token や利用者の絶対パスを保存しない。

### 手順 17: zstd 成果物の復元検査（E5）
- [ ] 🖐 **操作**: source/artifacts を追跡対象と明示 evidence から梱包し復元検査する。
- [ ] 🔎 **確認**: zstd test/SHA256 が成功し秘密/環境/キャッシュが含まれない。
- [ ] 🧪 **テスト**: 復元後の build/CLI と manifest の照合を行う。
- [ ] 🛠 **エラー時対処**: private metadata を whitelist で除外して再梱包する。

### 手順 18: commit と push（E5）
- [ ] 🖐 **操作**: 両 feature branch の review済み差分を commit して origin へ push する。
- [ ] 🔎 **確認**: remote SHA と local HEAD が一致する。途中の docs-only commit も許可済み。
- [ ] 🧪 **テスト**: `git diff --check`、公開情報検査、branch status を確認する。
- [ ] 🛠 **エラー時対処**: conflict は変更単位で解決、force push しない。

### 手順 19: PR と merge（E5）
- [ ] 🖐 **操作**: C++ develop / ONNX main への PR を作成しレビューと必須 CI 通過後 merge する。
- [ ] 🔎 **確認**: 両 PR の merged 状態と merge SHA を記録する。
- [ ] 🧪 **テスト**: `gh pr view` と fetch した base の SHA を確認する。
- [ ] 🛠 **エラー時対処**: CI 失敗を修正して再 push、完成していない機能を完了として説明しない。

### 手順 20: 一時ファイル整理（E5）
- [ ] 🖐 **操作**: 今回作成した検証用 staging directory を削除し `uv cache prune` を実行する。
- [ ] 🔎 **確認**: source repo、実重み原本、提出済み成果物を保持した状態で不要 staging のみ整理できた。
- [ ] 🧪 **テスト**: git status と提出物 SHA が成功する。
- [ ] 🛠 **エラー時対処**: 削除対象の所有/用途が不明なら調査して今回作成分に限定する。

## 4. 作業に使用するコマンド参考情報

ONNX repo ルート（E1/E2）：
```sh
uv sync --project efficientsam3
uv run --project efficientsam3 pytest efficientsam3/tests -q
```
C++ repo ルート（E3/E4）：
```sh
cmake --build build --parallel 16
ctest --test-dir build -LE real --output-on-failure
git diff --check
```
実装 CLI の確定 command は第8章の設計ゲートで追記する。未作成 CLI を動作済みと記載しない。

## 6. 完了の定義

- [ ] D1/E1：EV-M 固定実重み、モデル構成、必須キー/shape の検査結果が保存されている。
- [ ] D2/E2：ONNX の画像/6フレームで参照 mask IoU >=0.90、finite、実行証跡がある。
- [ ] D3/E3：GGUF 変換照合、C++/ggml の画像/6フレームで同等基準、外部推論を使わない証跡がある。
- [ ] D4/E4：境界/回帰テスト成功、時間を実測し report を公開情報だけで保存した。
- [ ] D5/E5：導入 guide、zstd 復元検査、両 repo push/PR/merge、整理まで記録した。

## 7. 作業記録

**重要な注意事項：**

* 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
* 各作業項目を開始する際と完了する際の両方で記録を行うこと。
* 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
* 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
* 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
* コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
* エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| --- | --- | --- | --- | --- |
| 2026-10-01 | 09:53 JST+0900 | Codex | 前作業完了後、手順1開始 | SAM31 PR#2 merged、両 feature branch を準備、固定 source/HF metadata を取得。開始時刻は前の date 記録を参照した分単位の記録。 |
| 2026-10-01 | 09:56:50 JST+0900 | Codex | 手順1完了 | EV-M Stage3 実重み SHA 一致。README と source を調査、b1 確定。text と tracker は次に実重みで検査する。 |

## 8. 設計ゲート

未確定：text backbone/context、必須 tensor、tracker の有無、stage 入出力。手順3–5で確定してから実装する。

| 2026-10-01 | 10:02:52 JST+0900 | Codex | 手順2/3完了、手順4開始 | CPU torch 2.7.0+cpu/ORT/ONNX import成功。実重み799 tensors/97455649 elements、MobileCLIP-S0 context16、trackerなし。private trainer metadataは保存しない。参照 import は動画backend未配置で失敗したため必要依存opencvを明示追加。 |
