# justfile — SAM3 ONNX export / oracle / inference / test / quality-gate
#
# All Python commands run via `uv run` (no bare python).
# Portability: set SAM3_SRC to the upstream SAM3 repo root on your machine.
# Every other path is relative to the repository root (justfile location).
#
# Usage:
#   just sync           # install all deps (including dev + dev-group)
#   just equiv-source   # generate equivalent SAM3 source (required before export)
#   just export-all     # run all ONNX export steps in dependency order
#   just oracle         # generate PyTorch oracle artefacts (slow, GPU recommended)
#   just run-video      # run ONNX video orchestrator and compare with oracle
#   just annotate       # start the local SAM 3.1 image/video annotator
#   just test-annotation # run annotation unit and HTTP tests
#   just test           # run full test suite
#   just e2e            # run MUST test (mask IoU >= 0.90)
#   just quality        # run all quality gates (format-check + lint + typecheck + complexity)

# ---------------------------------------------------------------------------
# Configurable variables — override via environment or `just --set VAR value`
# ---------------------------------------------------------------------------

# Upstream SAM3 repository root (source for equiv-source generation).
# Override with SAM3_SRC in .env when the official checkout is elsewhere.
SAM3_SRC := env_var_or_default("SAM3_SRC", "sam3")

# Equivalent-source output directory (relative to repo root).
EQUIV_SOURCE := env_var_or_default("EQUIV_SOURCE", "outputs/sam3_equiv_source")

# SAM3 checkpoint file (relative to repo root).
CHECKPOINT := env_var_or_default("CHECKPOINT", "models/sam3.pt")

# ONNX output directory (relative to repo root).
ONNX_DIR := env_var_or_default("ONNX_DIR", "outputs/onnx")

# Pinned SAM 3.1 Object Multiplex source and separate generated artifacts.
SAM31_SRC := env_var_or_default("SAM31_SRC", "sam31")
SAM31_CPU_SOURCE := env_var_or_default("SAM31_CPU_SOURCE", "outputs/sam31_cpu_source")
SAM31_CHECKPOINT := env_var_or_default("SAM31_CHECKPOINT", "models/sam3.1_multiplex.pt")
SAM31_ONNX_DIR := env_var_or_default("SAM31_ONNX_DIR", "outputs/onnx_sam31")

# ---------------------------------------------------------------------------
# Default: list all targets
# ---------------------------------------------------------------------------

_default:
    @just --list

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Install all dependencies: project extras (dev) + dependency-group (dev).
sync:
    uv sync --extra dev --group dev

# ---------------------------------------------------------------------------
# Equivalent-source generation (MUST run before any export)
# ---------------------------------------------------------------------------

# Generate SAM3 equivalent source copy with explicit real-valued RoPE wiring.
# Reads from SAM3_SRC; writes to EQUIV_SOURCE.
equiv-source:
    @echo "SAM3_SRC={{ SAM3_SRC }}"
    @echo "EQUIV_SOURCE={{ EQUIV_SOURCE }}"
    uv run python tools/create_equivalent_sam3_source.py \
        --source-root "{{ SAM3_SRC }}" \
        --output-root "{{ EQUIV_SOURCE }}"

# Generate a separate CPU/ONNX source copy from the official SAM 3.1 submodule.
sam31-source:
    uv run python tools/create_sam31_cpu_source.py \
        --source-root "{{ SAM31_SRC }}" \
        --output-root "{{ SAM31_CPU_SOURCE }}"

# ---------------------------------------------------------------------------
# ONNX export — individual targets
# ---------------------------------------------------------------------------

# Export detector image encoder (add_sam2_neck=False).
export-image-encoder:
    uv run python tools/export_image_encoder.py \
        --equiv-source "{{ EQUIV_SOURCE }}" \
        --checkpoint "{{ CHECKPOINT }}" \
        --output "{{ ONNX_DIR }}/image_encoder.onnx"

# Export tracker image encoder (SAM2 neck; uses repo-relative defaults in the tool).
export-image-encoder-tracker:
    uv run python tools/export_image_encoder_tracker.py

# Export memory_attention with fixed mem_len (2-frame default).
export-memory-attention:
    uv run python tools/export_memory_attention.py \
        --output "{{ ONNX_DIR }}/memory_attention.onnx"

# Export memory_attention with dynamic maskmem dimension (per num_k_exclude_rope).
export-memory-attention-dynamic:
    uv run python tools/export_memory_attention_dynamic.py

# Export memory encoder (SimpleMaskEncoder).
export-memory-encoder:
    uv run python tools/export_memory_encoder.py \
        --equiv-source "{{ EQUIV_SOURCE }}" \
        --checkpoint "{{ CHECKPOINT }}" \
        --output "{{ ONNX_DIR }}/memory_encoder.onnx"

# Export decode head (prompt_encoder + mask_decoder + obj_ptr_proj).
export-decode-head:
    uv run python tools/export_decode_head.py \
        --equiv-source "{{ EQUIV_SOURCE }}" \
        --checkpoint "{{ CHECKPOINT }}" \
        --output "{{ ONNX_DIR }}/decode_head.onnx"

# ---------------------------------------------------------------------------
# ONNX export — combined
# ---------------------------------------------------------------------------

# Run all ONNX exports in dependency order.
# equiv-source must already exist (run `just equiv-source` first).
export-all: export-image-encoder export-image-encoder-tracker export-memory-attention export-memory-attention-dynamic export-memory-encoder export-decode-head

# equiv-source + export-all as a single convenience target.
build-all: equiv-source export-all

# Export SAM 3.1 Object Multiplex tensor modules after sam31-source.
export-sam31-all:
    SAM31_CPU_SOURCE="{{ SAM31_CPU_SOURCE }}" \
    SAM31_CHECKPOINT="{{ SAM31_CHECKPOINT }}" \
    SAM31_ONNX_DIR="{{ SAM31_ONNX_DIR }}" \
    uv run python tools/export_sam31.py

# Generate the CPU source copy and export all SAM 3.1 ONNX modules.
build-sam31: sam31-source export-sam31-all

# ---------------------------------------------------------------------------
# Oracle generation (PyTorch reference; slow — GPU strongly recommended)
# ---------------------------------------------------------------------------

# Generate PyTorch detector oracle (outputs/reference/baseline_detector.npz).
oracle-detector:
    uv run python tools/run_pytorch_detector.py

# Generate PyTorch video tracking oracle (outputs/reference/video_oracle_all.npz).
oracle-video:
    uv run python tools/run_pytorch_video.py

# Generate both oracles.
oracle: oracle-detector oracle-video

# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

MAX_FRAMES := env_var_or_default("MAX_FRAMES", "")
EMULATE_BF16 := env_var_or_default("EMULATE_BF16", "")

# Run ONNX video orchestrator and compare with oracle (MAX_FRAMES / EMULATE_BF16 optional).
run-video:
    uv run python tools/run_onnx_video.py \
        {{ if MAX_FRAMES != "" { "--max-frames " + MAX_FRAMES } else { "" } }} \
        {{ if EMULATE_BF16 == "true" { "--emulate-bf16" } else { "" } }}

# Start the local SAM 3.1 image/video annotation UI.
annotate:
    uv run python tools/run_sam31_annotation.py serve

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Run the full test suite.
test:
    uv run python -m pytest -q

# Run model-free annotation data and HTTP boundary tests.
test-annotation:
    uv run python -m pytest \
        tests/test_sam31_annotation.py \
        tests/test_sam31_annotation_server.py -q

# Run the MUST e2e test (memory-bank video tracking, mask IoU >= 0.90).
e2e:
    uv run python -m pytest tests/test_video_e2e.py -q

# Run the two-object, six-frame SAM 3.1 ORT vs official PyTorch comparison.
sam31-e2e:
    SAM31_SRC="{{ SAM31_SRC }}" \
    SAM31_CPU_SOURCE="{{ SAM31_CPU_SOURCE }}" \
    SAM31_CHECKPOINT="{{ SAM31_CHECKPOINT }}" \
    SAM31_ONNX_DIR="{{ SAM31_ONNX_DIR }}" \
    uv run python -m pytest tests/test_sam31_video_e2e.py -q -s

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

# Check formatting (non-destructive; exits non-zero if reformatting is needed).
format-check:
    uv run python -m ruff format --check .

# Apply auto-formatting.
format:
    uv run python -m ruff format .

# Lint (ruff check).
lint:
    uv run python -m ruff check .

# Static type check (ty).
typecheck:
    uv run python -m ty check src tools tests

# Cyclomatic complexity gate (fail on grade C or worse).
complexity:
    uv run python -m radon cc src -n C

# Check Jupyter notebook formatting with ruff via nbqa.
format-notebooks-check:
    uv run nbqa "ruff format --check" notebooks/sam3_onnx_video_demo.ipynb

# Format Jupyter notebooks with ruff via nbqa.
format-notebooks:
    uv run nbqa "ruff format" notebooks/sam3_onnx_video_demo.ipynb

# Lint Jupyter notebooks with ruff via nbqa.
# E402: notebook cells import after top-level code by design.
# E501: long f-strings in print cells are acceptable in demo notebooks.
lint-notebooks:
    uv run nbqa "ruff check --ignore E402,E501" notebooks/sam3_onnx_video_demo.ipynb

# Run all quality gates (Python + notebook format/lint + typecheck + complexity).
quality: format-check lint format-notebooks-check lint-notebooks typecheck complexity
