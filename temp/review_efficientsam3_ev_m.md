# EfficientSAM3 EV-M 作業書レビュー

**Verdict:** PASS

**Mode:** review-and-fix

**Findings**

- Minor: 手順5の仕様は現時点で調査中。実装前の再レビューを必須にした。未知のテキスト構成と tracker 有無を完成済み扱いにしていないため調査開始を妨げない。
- Minor: ONNX に独立 CPU uv project を追加するため既存 just targets をそのまま使えない。追加機能専用の just targets を設計ゲートで確定する。

**Applied Changes**

- 実重み SHA/size、source/HF revision、C++規約、両基点、正本/同期先、model-free 回帰、実マスク基準、backend fallback 禁止を明示した。
- 動画機能を checkpoint の tracker 有無で決定し、frame独立推論をメモリ追跡と称さない方針を明示した。

**Residual Findings**

- 手順4のstrict load799/799と手順5の構成/入出力/ファイル/テスト記載を確認。必須仕様の未確定事項はない。

**Coverage Notes**

- ゴール要求分析: adequate
- サブゴールと作業要素の対応: adequate
- 完了の定義: adequate
- チェックリスト原子性: adequate
- TDD/検証可能性: adequate
- エラー時対処: adequate
- トレーサビリティ: adequate

**Open Questions**

- なし（未知の構成は実重みの調査で決定できる）。

**Recommended Patch Scope**

- 手順5完了時に第8章の入出力/ファイル/実行コマンドを確定し、このレビューを更新する。

## 実装後の再レビュー（2026-10-01）

**Verdict:** PASS（D1–D5達成）

- 必須799 tensorのstrict検査と全payload照合、実ONNX/native画像・6frame比較、有限値、入力依存、frame0再現、計測、復元buildを確認した。
- nativeはgraph分離、CPU vector受け渡し、PVS/両tracker拒否、既存API dispatchを確認。拒否漏れとエラー時解放順を修正した。
- 高速化を未計測で主張しない。grouped batched案は撤回し、4/16threadの計測は各1回と明記した。
- 手順15の実checkpoint必須の既存テストは環境不足で失敗。model-free回帰36 passed/19 skippedとSAM31 CTest5 groupsを根拠とし、未実行の実重み回帰を成功に数えていない。
- work logを第7章に集約し、途中のred/import failureと未完了の公開項目を残した。各DoDの証跡はdocs/efficientsam3-validationに追跡できる。
- source archiveのtar uid/gid/利用者名を消去し、tokens/cache/元checkpoint trainer metadataを含めていない。

**Residual Findings:** mandatory CI contextsは両対象branchに無い。ローカルの実モデルゲートを使用する。PR#3/#5のmerged状態とstaging整理・pruneを確認し、手順19/20とD5を更新した。checkoutの終了処理は提出先cleanup.jsonへ記録する。


## 2026-10-01 追加完了監査・ONBOARDINGレビュー

model選択→isolated uv→固定取得→変換/export→画像/画像列→境界/実モデルtestの導線を照合。相対リンク、CLI引数、b1/S0/context16、tracker無し、実測と保証の区別、private値不在を確認。ONNXの順序/空入力/寸法境界を追加して23件成功。native CTestは共有libraryのbuild漏れを補い5/5成功。梱包tarのowner名を削除して2146file SHAを実検査。作業書第10章に追加公開と終了処理待ちを明記し、削除済みローカル提出物の再生成は依頼変更により除外。安全な追記を両repoへ反映済み。
