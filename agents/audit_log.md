# 監査ログ（audit agent 専用の作業記録）

監査エージェント(opus)が実施した監査の途中過程・検証コマンド・証拠・指摘・判定を時系列で残す。append のみ。
役割分担: workdoc §7 = worker の作業記録 / 本ファイル = 監査の作業記録 / `agents/roster.md` = 統括の調整記録。

> 2026-05-21 以前の下記2件は、監査記録の義務化(audit.txt §13)以前に実施されたため、統括が監査レポートから遡及記録した。

---

## [2026-05-20 22:44 JST] 監査完了: 手順 D1+D2（uv環境・PyTorch baseline）→ 判定: **承認**
- 対象: 手順D1(uv環境構築), D2(公式detector PyTorch baseline oracle)。監査レベル: 簡易。
### 実行した検証コマンド（要約）
- `uv run pytest tests/test_env_smoke.py tests/test_baseline_oracle.py -q` → 2 passed / exit 0
- `np.load(outputs/reference/baseline_detector.npz)` → masks(1,1,1008,1008)/boxes(1,4)/scores(1,) 全非空, any_nonzero=True
- `git -C ~/Project/sam3 status --short` → 前セッション分のみ(本作業由来の改変なし)
- `grep -rnE "pip install|/opt/pyvenv|/mnt/data|sys.path" tools tests` → ヒット無し
### 指摘一覧
- [情報][事実整合性] 公式 ~/Project/sam3 に未コミット変更あるが mtime=2026-05-19 で前セッション由来（スコープ違反ではない）。
- [注意][事実整合性] workdoc D2記述「synthetic.py 流用」と実装(独自1008²生成)が不一致 → 統括が記述修正済。
- [情報] 暗黙fallback禁止が `_detect_device`(CUDA/CPU明示log)+`load_from_HF=False`+明示checkpoint で担保。
### 判定根拠
DoD-D1/D2 を満たす客観証拠が揃い、制約(uv限定/公式非改変/TDD/暗黙fallback禁止)違反なし。

---

## [2026-05-21 JST] 監査完了: Stage A image encoder ブロック（手順 D3,D4,D5-1,D5-2）→ 判定: **条件付き承認**
- 対象: D3(RoPE等価), D4(等価ソース), D5-1(ViT cos/sin移植), D5-2(image_encoder ONNX export)。監査レベル: 厳密。export方法論が後続全モジュールで再利用されるため重点検証。
### 実行した検証コマンド（要約）
- `uv run pytest -q`（フルスイート）×3 → **run#1=1 failed(NaN, backbone_fpn_0), run#2/#3=35 passed**（間欠FAILを検出）。isolation実行は1 passed。
- `onnx.load(image_encoder.onnx, load_external_data=False)` → 31 op_types, complex系 不在(FORBIDDEN_HITS=[]), Cos/Sin在り, opset18。`onnx.checker.check_model`(full path) → OK。`graph.output` → 全出力 symbolic dim（動的）。
- `diff -u 公式vitdet.py 等価vitdet.py` → apply_rotary_enc2追加+_apply_rope 2分岐のみ（complex経路は freqs_cis存在時のみ発火）。
- `grep -rnE "pip install|/opt/pyvenv|/mnt/data" src tools tests` → 無し。`git -C ~/Project/sam3 status` → 前セッション分のみ。`find ~/Project/sam3/sam3 -name "*.py" -newermt 2026-05-20` → 空（公式非改変）。
### 指摘一覧
- **[重大][事実整合性] parity 間欠 NaN**: フルスイート3回中1回 `test_ort_parity` が backbone_fpn_0 全NaN で FAIL。ORT警告 `MergeShapeInfo Concat {4} vs {5} lenient merge`。test順/sys.modules汚染の相互作用も疑い。
- [警告][網羅性] 「固定shape」申告に反し出力 shape が symbolic（Range/Mod/Shape/Expand/If 残存）。NaN の媒介と推定。
- [注意][再現性] `replace_rope_freqs(model.backbone)` 適用範囲 vs ガード(model全体) 不整合。tracker再利用時にリスク。
- [情報] decord==0.6.0 が uv run 毎に再リンク（無害）。
### 判定根拠・残課題
中核 export 方法論（意図的variantロード/cos-sin等価/exact-once patcher/complex除去ガード）は健全・後続再利用可。ただし [重大]NaN と [警告]動的shape を memory_attention 展開前に是正必須 → 統括が手順 D5-3 を新設（後に worker が freeze_abs_pos_for_export で If node 除去・静的shape化し、統括検証で op17種/complex・If不在/静的shape/41 passed×3 を確認＝是正完了）。

---

## [2026-05-21 08:47 JST] 監査開始: Stage B（tracker＋memory モジュール: 手順 B-1/B-2/B-3）
- 対象: B-1 memory_attention (`outputs/onnx/memory_attention.onnx`+`memory_attention_full_36352.onnx`), B-2 memory_encoder (`outputs/onnx/memory_encoder.onnx`), B-3 decode_head (`outputs/onnx/decode_head.onnx`)。各 `tests/test_*_onnx.py`。
- 監査レベル: 厳密（Stage B = MUST臨界、tracker/memory bank の novel work）。
- 重点: B-1 num_k_exclude_rope=64 設計妥当性 / use_rope_real=True 実機 / complex不在 / 実config(36352)parity、B-2 antialias patch等価性、Stage C 入出力契約整合。

- [08:52 JST] `onnx.load(load_external_data=False)` ×4 → 全 FORBIDDEN_HITS=[], HAS_IF=False, opset18, 全I/O静的整数。memory_attention(10368)/full(36352) nodes=1271同一, memory_encoder nodes=255(op24種), decode_head nodes=746(op38種)。complex/If不在を独立確認。

- [08:55 JST] `onnx.checker.check_model`(full external data) ×4 → 全 OK。
- [08:56 JST] `uv run pytest tests/test_memory_attention_onnx.py tests/test_memory_encoder_onnx.py tests/test_decode_head_onnx.py -q` → **9 passed in 32.94s, exit 0**。但しB-1テストはnum_k_exclude_rope=0/mem_len=10368のみ（実config36352/exclude64はテスト外）。

- [09:02 JST] B-1 MUST重点: 実config再現。`build_memory_attention_module(mem_len=36352,num_k_exclude_rope=64)` vs `memory_attention_full_36352.onnx` → seed42/43 max_abs_diff=2.86e-6/2.38e-6 NaN=False。8 RoPEAttention全て use_rope_real=True(self 4 rope_k_repeat=False / cross 4 rope_k_repeat=True), freqs_cis_real.shape=(5184,128)→HW一致で動的recompute branch非trace。独立確認OK。
- [09:03 JST] B-2 antialias: 公式memory.py:76 `antialias=True` / equiv:79 `antialias=False`+説明comment(直接Read確認)。interpol_size=[1152,1152]=1008→1152 upsample。実測 upsample diff=4.768e-7(downsampleなら1.03)→antialiasはupsample無効が裏付け。公式sam3 git status=pyproject/uv.lock/.serena/examplesのみ(model .py改変なし, newermt 2026-05-20空)。constraint grep(pip/venv/mnt)空。

- [08:53 JST] Stage C契約検証（equiv tracker_base.py 精読）:
  - B-1: prod call site `transformer.encoder(src=[..],src_key_padding_mask=[None],src_pos=[..],prompt,prompt_pos,feat_sizes,num_obj_ptr_tokens)`(L783-792)とwrapper(memory_attention.py:132-141)一致。num_obj_ptr_tokens=obj_ptrs.shape[0](C//mem_dim=4倍展開, L759-765)→max16×4=64=num_k_exclude_rope。設計正。初期フレームは `+no_mem_embed` Python経路(L769-772, ONNX外)＝§9.9申し送り通り。
  - B-2: prod `maskmem_backbone(pix_feat,mask_for_mem,skip_mask_sigmoid=True)`(L835-836)一致。但し**sigmoid/scale/bias変換(L822-831)は encoder前**, **no_obj_embed_spatial加算(L845-848)は encoder後**＝両方ONNX外でStage C実装必須。workdoc B-2申し送りに no_obj_embed_spatial 明記なし。
  - B-3: prod `_forward_sam_heads`(L340-377)は `torch.where(is_obj_appearing, masks, NO_OBJ_SCORE)`(L344-348)でmask gating＋high_res interpolate(L353)。wrapper(decode_head.py:177-204)はmask_decoder生出力を返し**mask gating省略**(obj_ptr blendは一致)。decode_head ONNXはungated low_res_masks出力→Stage Cが object_score_logits でgating要。conv_s0/s1はFPNレベルにdecode_head外で適用(L450-453)＝wrapperは投影済high_res_feat0/1(32/64ch)期待で整合。dynamic_multimask_via_stability(torch.where, If不在)はONNX-safe。

---

## [2026-05-21 08:53 JST] 監査完了: Stage B（手順 B-1/B-2/B-3）→ 判定: **条件付き承認**
### 実行した検証コマンド（cwd=/home/inaho-omen/Project/sam3_onnx_sandbox）
| command | exit | 結果要約 |
|---|---|---|
| `onnx.load(load_external_data=False)` ×4 | 0 | 全 FORBIDDEN_HITS=[]/HAS_IF=False/opset18/I/O静的整数。mem_attn(10368)/full(36352) nodes=1271, mem_enc nodes=255(op24), decode_head nodes=746(op38) |
| `onnx.checker.check_model`(full external) ×4 | 0 | 全 OK |
| `uv run pytest tests/test_memory_attention_onnx.py tests/test_memory_encoder_onnx.py tests/test_decode_head_onnx.py -q` | 0 | **9 passed in 32.94s** |
| 独立再現: build_memory_attention_module(36352,exclude64) vs full_36352.onnx | 0 | seed42/43 max_abs_diff=2.86e-6/2.38e-6 NaN=False。8 RoPEAttention全 use_rope_real=True, freqs_cis_real.shape=(5184,128) |
| antialias上書き実測 | 0 | upsample1008→1152 diff=4.768e-7（downsample比1.03）→antialias=False等価が裏付け |
| `git -C ~/Project/sam3 status` / `find ... -newermt 2026-05-20` | 0 | model .py改変なし(pyproject/uv.lock/.serena/examplesのみ) |
| `grep -rnE "pip install\|/opt/pyvenv\|/mnt/data" src tools tests` | 1 | ヒット0（uv限定遵守） |
### 参照した証拠
- 成果物: outputs/onnx/{memory_attention.onnx, memory_attention_full_36352.onnx, memory_encoder.onnx, decode_head.onnx}
- テスト: tests/test_{memory_attention,memory_encoder,decode_head}_onnx.py
- 実装: src/sam3_onnx_equiv/export/{memory_attention,memory_encoder,decode_head}.py
- equiv-source: outputs/sam3_equiv_source/sam3/{model/decoder.py:617-936, sam/transformer.py:266-355, sam/rope.py:90-114, model/memory.py:69-81, model/sam3_tracker_base.py:340-377/560-795/797-850, sam/mask_decoder.py:107-163}
- Plan: workdoc §9.9 (B-1/B-2/B-3)
### 指摘一覧
- [警告][網羅性] tests/test_memory_attention_onnx.py:73 NUM_K_EXCLUDE_ROPE=0/mem_len=10368 のみ pytest 対象。MUST臨界の num_k_exclude_rope=64/mem_len=36352(obj_ptr RoPE除外)は worker手動＋本監査の独立再現(max_abs_diff=2.86e-6)のみで自動回帰なし。→ full config の parity を pytest 化（slow mark 可）推奨。
- [警告][網羅性] B-2/B-3 wrapper は production の encoder前後/decode後の post-processing を含まない: mem_enc=sigmoid/scale/bias(前)＋no_obj_embed_spatial加算(後), decode_head=torch.where(is_obj_appearing,masks,NO_OBJ_SCORE) mask gating。Stage C orchestrator で Python 実装必須。workdoc 申し送りに **no_obj_embed_spatial と mask gating が未明記**。→ Stage C 設計に追記要。
- [注意][可読性] tests/test_memory_encoder_onnx.py:21-22 docstring「wrapper applies sigmoid internally」は実装(skip_mask_sigmoid=True=sigmoid非適用)と矛盾。module docstring(memory_encoder.py:40)が正。→ test docstring 修正推奨。
- [情報] op_type数の軽微なゆれ: worker log mem_enc=25種だが本監査=24種。complex/If不在の本質は不変。
### 残課題・追加要求
- Stage C 進行可否: モジュール単体の export/parity は健全。但し上記post-processing境界を Stage C で正しく実装しないと E2E parity が破綻するため、Stage C 設計時に no_obj_embed_spatial / mask gating / sigmoid前処理を明示的にタスク化すること。
- memory_attention は memory bank ありの tracking を担える: prompt(memory tokens)経由のcross-attn＋num_k_exclude_rope によるobj_ptr RoPE除外が production call site と一致し、実config(36352)でparity成立を独立確認済。memory bank の組立(maskmem連結/obj_ptr展開/tpos加算)は Stage C Python 責務。

## [2026-05-21 13:01 JST+0900] 監査開始: Stage C 手順D16 MUST claim (memory-bank video tracking ONNX equivalence)
- 対象: tools/run_onnx_video.py, src/sam3_onnx_equiv/video_orchestrator.py, tests/test_video_e2e.py, logs/video_e2e.log; F-14(image_encoder_tracker.onnx), F-15(decode_head multimask), F-16(score dtype)
- 監査レベル: 厳密（ユーザー厳命MUST・最重要）
- Plan: temp/workdoc_May20-2026_sam3_onnx_video_export.md §9.11(C-2/C-3/MUST判定/F-14/15/16), §9.10, §9.3(DoD-D-MUST), §9.9, §7

## [2026-05-21 13:10 JST+0900] 監査中: 制約・成果物確認
- `/usr/bin/git -C ~/Project/sam3 status` → model source(.py)は不変。pyproject.toml のみ M（requires-python>=3.9, torch/torchvision/torchaudio/einops/decord/psutil deps追加, cu126 index・sources追加）＝packaging変更でモデルロジック非改変。等価patchは outputs/sam3_equiv_source/ 側で実施(公式不変)。
- sandbox `git status` → commit/push無（全 untracked, .gitignore で重い成果物除外）。
- 成果物確認: outputs/onnx/ に image_encoder_tracker.onnx(1.83GB), decode_head.onnx(17.7MB), memory_attention_dynamic_k{4..24}.onnx(各40MB), memory_encoder.onnx(5.6MB) 全存在。logs/video_e2e.log の per-frame IoU f0=.9984..f5=.9915 全PASS, score f4/f5 FAIL確認。
- video_orchestrator.py 精読: memory_attention は frame≥1 で実ループ invoke(L580), prompt は過去frame maskmem+obj_ptr から組立(L571)。score閾値5e-2 は test_video_e2e.py:46 でハードコード非緩和を確認。

## [2026-05-21 13:18 JST+0900] 監査中: onnx.load/checker + F-14/F-16検証
- onnx.checker+op scan(image_encoder_tracker/decode_head/memory_attention_dynamic_k4,k24/memory_encoder): 全 checker OK, complex/polar/If **全不在**。image_encoder_tracker I/O=pixel_values(1,3,1008,1008)→6出力(vision_pos_enc×3+backbone_fpn×3) 期待通り。memory_attention_dynamic は prompt が dynamic 'mem_len' (zero-pad無の可変長memory)。
- F-14裏付け: image_encoder.py:478-529 で tracker backbone を build_tracker(with_backbone=True,use_rope_real=True)→ ImageEncoderWrapper(use_sam2_neck=True)→ forward が sam2_backbone_out(SAM2 neck FPN)を返す(L104-112)。detector用(use_sam2_neck=False)とコードで別系統と確認。key map detector.backbone.*→backbone.* は oracle loader と一致。
- F-16独立検証中: tools/diag_video_fp32.py を実行(ONNX backbone[CPU]→公式 _prepare_memory_conditioned_features/_forward_sam_heads/_encode_new_memory を PT fp32/CUDAで実行 vs ONNX orchestrator)。これで隠れ logic bug でなく dtype差かを判定。pytest test_video_e2e.py も並行再実行中。

## [2026-05-21 13:35 JST+0900] 監査中: 重要発見 — maskmem prompt の tpos 配置差
- 公式 `_prepare_memory_conditioned_features`(sam3_tracker_base.py:658-679): maskmem section の **`to_cat_prompt`(=prompt/value)に maskmem_tpos_enc を加算しない**(feats のみ, L659)。`to_cat_prompt_pos_embed`(=prompt_pos/key)に `maskmem_pos_enc + maskmem_tpos_enc` を加算(L676-678)。
- cross-attn(decoder.py:908-912 `_forward_ca`): `k = memory + pos`(=prompt+prompt_pos), `v = memory`(=prompt)。MemoryAttentionWrapper(memory_attention.py:132-141)は公式 encoder() をそのまま呼ぶ→ ONNX graph も `k=prompt+prompt_pos, v=prompt` で公式同一。
- 一方 video_orchestrator.py:367-368,387-388 は maskmem を `prompt=feats+tpe`, `prompt_pos=pos+tpe` と **両方に tpos 加算**。→ ONNX 内で k=feats+pos+2*tpe(tpos二重), v=feats+tpe(value に tpos混入)＝**公式と非一致の logic 差**。obj_ptr section(L418-424)は tpos を obj_pos のみに置き公式一致。
- 判定保留中: この差が数値的に無視できるか／実 bug かは diag_video_fp32.py(公式 _prepare... fp32 vs orchestrator)の一致度で決まる。両 background job 実行中(encoder ロード重い)。

## [2026-05-21 13:50 JST+0900] 監査中: §9.10 Python側処理 + F-15 検証完了（diag/pytest 待ち）
- §9.10 条件をコードで照合(video_orchestrator.py vs 公式 sam3_tracker_base.py):
  - ①binary/sigmoid mask: L627-633 = 公式822-826一致(frame0 binary, 非cond sigmoid)。*scale+bias L634=公式828-831一致。
  - ②no_obj_embed_spatial加算: L648-650 = 公式845-848一致。
  - ③mask gating: decode_head.onnx 内 Where で obj_score>0 gating(op scan で Where 確認)。
  - ④memory bank組立: maskmem連結+obj_ptr 4トークン展開+tpos。obj_ptr tpos は obj_pos のみに置き公式一致(L418-424)。**但し maskmem tpos は prompt/prompt_pos 両方加算で公式と差(13:35記録)**。
  - ⑤conv_s0/s1: L554-555 _conv1x1 適用。⑥定数: Constants で checkpoint抽出値ロード。
  - get_1d_sine_pe/_get_tpos_enc: orchestrator L272-308 が公式 utils:327 + tracker_base:162 の忠実 numpy port(pe_dim=dim//2, temperature**(2*(i//2)/pe_dim), concat[sin,cos], Linear256→64)。
- F-15: _use_multimask(tracker_base:1108) は config(multimask_in_sam=T, for_tracking=T, min/max_pt=0/1)で全フレーム True。decode_head.onnx op に ArgMax/Gather/Where 存在=multimask+best-IoU+gating baked。complex/If不在。
- score閾値非緩和: test_video_e2e.py:46 SCORE_REL_DIFF_THRESHOLD=5e-2 ハードコード, assert all_pass(L243)。conftest.py は sam3 module isolation のみ(skip/xfail trickery 無)。
- no-mem path(L566-569)=公式 else(769-772) no_mem_embed加算一致。公式の dummy-token block(765-768)は return後 dead code、orchestrator は正しく no_mem_embed 経路。

## [2026-05-21 13:55 JST+0900] 監査中: pytest config 確認・background job 待機
- pyproject.toml [tool.pytest.ini_options] に addopts/xfail/filterwarnings 無＝score FAIL は隠蔽されない。pytest.ini/setup.cfg/tox.ini 無。
- background: test_video_e2e.py 再実行 + diag_video_fp32.py(F-16 dtype帰属の決定的証拠) 並行実行中。完了通知待ち。

## [2026-05-21 14:30 JST+0900] 監査中: 決定的検証2件完了
### A) pytest test_video_e2e.py 独立再現
- `uv run pytest tests/test_video_e2e.py -q` → **1 failed, 4 passed in 1175.09s (EXIT=1)**。worker申告(4 passed/1 failed)を独立再現。
- PASS: mask IoU(f0=.9984..f5=.9915 全≥0.99), memory_attention invoke=5, ablation(mean_abs_diff_logits 14.6-15.4≫0.5), obj_id=1一貫。FAIL: score f4=0.0651/f5=0.0798>5e-2(閾値非緩和の assert)。
### B) diag_video_fp32.py (F-16 dtype帰属の決定的クロスチェック, EXIT=0)
- 同一 ONNX backbone 特徴を 公式 _prepare_memory_conditioned_features/_forward_sam_heads/_encode_new_memory(PT **fp32**)に通した head vs ONNX orchestrator。
- IoU(onnx,ptfp32): f0=.9995..f5=.9928 全≥0.99。
- **score rel(onnx vs PT-fp32-head)**: f0=.0025 f1=.0121 f2=.0325 f3=.0365 **f4=.0523 f5=.0798**。
- score rel(onnx vs bf16 oracle): f4=.0651 f5=.0798。PT-fp32-head score(10.56,10.375,10.5,10.375,9.5625)は bf16 oracle(10.625,10.375,10.5,10.25,9.5625)とほぼ一致。
### 監査の解釈（重要・worker F-16主張と部分相違）
- **ONNX-fp32 は 公式 fp32 PT-head とも f4=.052/f5=.080 乖離**＝gap は「純粋に bf16 vs fp32」**ではない**。worker の test docstring/§9.11「late-frame gap is purely bf16(oracle) vs fp32(ONNX), NOT a logic error」は本クロスチェックで**反証**される。fp32同士でも乖離が残る＝orchestrator memory経路に bf16以外の数値差要因(候補: 13:35記録の maskmem tpos 二重加算, memory_attention PT-CUDA vs ONNX-CPU, conv/upsample経路)が存在。
- 但し: ① mask IoU は ONNX vs 公式fp32-head でも全フレーム≥0.9928(MUST達成は揺るがず) ② obj_score は両者とも ~10(物体検出/出現判定は一致, NO_OBJ_SCORE境界から大きく乖離せず) ③ 乖離は後半フレームの confidence logit 値の小差。tracking失敗ではない。

## [2026-05-21 13:24 JST+0900] 監査完了: Stage C 手順D16 MUST claim → 判定: 条件付き承認
### 実行した検証コマンド（cwd|command|exit|結果要約）
| /home/inaho-omen/Project/sam3_onnx_sandbox | uv run pytest tests/test_video_e2e.py -q | 1 | 1 failed,4 passed(1175s)。IoU/invoke/ablation/obj_id PASS, score f4/f5 FAIL再現 |
| 同上 | uv run python tools/diag_video_fp32.py | 0 | ONNX-fp32 vs 公式PT-fp32-head: IoU≥0.9928全, score rel f4=.052/f5=.080(fp32同士でも乖離残存) |
| 同上 | uv run python (onnx.load+checker+op scan ×5 artifacts) | 0 | 全 checker OK, complex/polar/If 全不在。tracker enc I/O・dynamic mem_len 確認 |
| /home/inaho-omen/Project/sam3 | /usr/bin/git status | 0 | model .py 不変。pyproject.toml のみM(packaging) |
### 参照した証拠
- logs/video_e2e.log, /tmp/audit_pytest_e2e.log, /tmp/audit_diag_fp32.log
- src/sam3_onnx_equiv/video_orchestrator.py, .../export/{image_encoder.py,memory_attention.py}
- 公式 sam3/model/sam3_tracker_base.py(572-860,1108), sam3/model/decoder.py(614-720,886-918), sam3/model/sam3_tracker_utils.py(327)
- tests/{test_video_e2e.py,conftest.py}, pyproject.toml, temp/workdoc §9.3/9.9/9.10/9.11
### 指摘一覧（深刻度・対象・内容・根拠・推奨対応）
- [警告][事実整合性] video_orchestrator.py:367-368,387-388 — maskmem を prompt(value)/prompt_pos(key) 両方に maskmem_tpos_enc 加算。公式(sam3_tracker_base.py:659 prompt=feats のみ, :676-678 prompt_pos=pos+tpos)は **prompt_pos のみ**に加算。MemoryAttentionWrapper は公式 encoder をそのまま呼ぶ(k=prompt+prompt_pos,v=prompt; decoder.py:910-912)ため、ONNX 内で tpos が key で二重・value に混入。→ 公式と非一致の logic 差。obj_ptr section(L418-424)は正。**推奨**: maskmem section の prompt を feats のみ(tpos 無)に修正し score 差が縮むか検証。
- [警告][事実整合性] §9.11/test_video_e2e.py:215-223 — worker主張「score gap は purely bf16(oracle) vs fp32(ONNX), NOT a logic error」は diag_video_fp32(公式 fp32 PT-head vs ONNX-fp32 でも f4=.052/f5=.080 乖離)で**部分反証**。fp32同士の残差＝bf16以外の数値要因(上記 tpos二重加算 or CUDA-PT vs CPU-ONNX op順序)が存在。score差を全て dtype 帰属とする記述は不正確。**推奨**: 帰属記述を「bf16 + memory経路 fp32 残差」に訂正、tpos 修正で残差切り分け。
- [注意][網羅性] full-config(36352) memory_attention の pytest 化が未実施(workdoc 自認の[警告]残)。実config parity 2.86e-6 は B-1 で単発確認済だが回帰テスト無。
- [情報] sam3/pyproject.toml は M だが model source(.py)不変＝MUST(公式同等)の前提は保持。等価patchは outputs/sam3_equiv_source 側で実施。
### MUST 判定（明示）
- (a) ユーザーMUST(memory-bank video tracking の ONNX 推論が公式同等)= **達成と認めてよい**。根拠: ① per-frame mask IoU 全6フレーム≥0.99 (vs bf16 oracle) かつ ≥0.9928 (vs 公式 fp32 head, dtype交絡除去後) ② memory_attention 実ループ invoke=5 ③ ablation mean_abs_diff 14.6-15.4≫0.5＝真の memory tracking(mask-prompt近似でない) ④ obj_id一貫 ⑤ memory bank 組立/tpos/mask処理が公式 file:line と一致(maskmem tpos の1点を除く)。MUST の本質(memory bank ありの tracking が公式同等の mask を出す)は満たす。
- (b) score f4/f5 FAIL = **dtype caveat としてのみの許容は不可**。diag が fp32同士でも乖離を示す＝純 dtype でない。但し **logic bug としても tracking 破綻ではない**(mask IoU≥0.99 維持, score~10 で出現判定一致)。confidence logit の後半小差。→ 「dtype + memory経路 fp32 残差」と訂正の上、score criterion は MUST のブロッカーにせず refinement 扱いが妥当。
- (c) 残課題: ① maskmem tpos 二重加算の修正と score 再測定(score 完全クローズの最有力) ② F-16 帰属記述の訂正 ③ full-config 36352 の pytest 化 ④ CPU-fp32 oracle クロスチェック(GPU OOM回避)。
### 全体ステータス: 条件付き承認（MUST=mask基準で達成。score criterion は logic差含む残差のため帰属記述訂正と tpos 修正を条件に refinement)
