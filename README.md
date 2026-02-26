# ROCm-FA3 (MI300X): FlashAttention on AMD with CK or Triton

This fork is focused on one thing: getting FlashAttention working reliably on AMD ROCm systems (especially MI300X), with a clear path for either backend:

- **CK backend** (Composable Kernel, default ROCm backend)
- **Triton backend** (recommended when targeting the FA3-style Triton path on ROCm)

> Tested in this repo on **AMD Instinct MI300X VF (`gfx942`)** with **ROCm 6.2**.

---

## Backend choice

| Backend      | Best for                                             | Notes                                                      |
| ------------ | ---------------------------------------------------- | ---------------------------------------------------------- |
| CK (default) | Stable ROCm path, simple install                     | Uses ROCm composable-kernel backend path.                  |
| Triton       | FA3-style interface path on ROCm, flexibility/tuning | Requires pinned Triton (`3.5.1`) and AMD Triton env flags. |

---

## Prerequisites

- Linux host with ROCm installed (ROCm **6.2** recommended for this repo setup)
- AMD GPU supported by ROCm (MI300X/MI300 class recommended)
- `git`
- `uv` (Python env + package management)

If `uv` is missing:

```bash
python3 -m pip install --user --break-system-packages uv
```

---

## 1) Common environment setup (uv)

```bash
uv venv
source .venv/bin/activate
```

Install shared build tools:

```bash
uv pip install packaging ninja wheel numpy
```

Install ROCm PyTorch (required before building):

```bash
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
```

Clone source:

```bash
git clone https://github.com/RESMP-DEV/ROCm-FA3.git
cd ROCm-FA3
```

---

## 2A) Build with CK backend (default ROCm path)

Do **not** set `FLASH_ATTENTION_TRITON_AMD_ENABLE`.

```bash
export MAX_JOBS=4
uv pip install --no-build-isolation .
```

---

## 2B) Build with Triton backend (FA3-focused ROCm path)

Install pinned Triton:

```bash
uv pip install triton==3.5.1
```

Enable AMD Triton backend and limit build parallelism:

```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export MAX_JOBS=4
```

Build/install:

```bash
uv pip install --no-build-isolation .
```

Optional tuning:

```bash
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"
```

---

## 3) Validate FA3 execution

This repo includes two validation scripts:

- `test_fa3.py` — minimal forward-pass validation
- `prove_fa3_mi300x.py` — prints backend + device proof and runs forward pass

Run:

```bash
python test_fa3.py
python prove_fa3_mi300x.py
```

Expected proof signals from `prove_fa3_mi300x.py`:

- `torch version: ...+rocm6.2`
- `device name: AMD Instinct MI300X VF`
- `device gfx arch: gfx942...`
- `flash-attn backend symbol: flash_attn.flash_attn_triton_amd.interface_v2` (for Triton path)
- `FA3 forward success ...`

---

## 3B) Benchmark FA2 vs FA3 for MLP Transformer training

To compare end-to-end training step performance (forward + backward + optimizer step)
on a small Transformer with MLP blocks, run:

```bash
python benchmarks/benchmark_fa2_vs_fa3_mlp_transformer.py
```

By default, this uses the `mi300x` preset (larger sequence/layer counts) to produce
a stronger, more realistic MI300X signal out of the box.

The script runs two isolated subprocesses so backend selection is fair and explicit:

- `FA2` via `FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE`
- `FA3` via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`

It reports:

- median step time (ms)
- steps/sec and tokens/sec
- peak GPU memory
- FA3 speedup over FA2

Available presets:

- `mi300x` (default): `batch=4, seqlen=4096, d_model=1024, heads=16, layers=8, warmup=20, timed=80, dtype=bfloat16`
- `default`: lighter baseline (`batch=4, seqlen=2048, layers=6, warmup=10, timed=50`)

Example with custom model/sequence settings:

```bash
python benchmarks/benchmark_fa2_vs_fa3_mlp_transformer.py \
  --batch-size 4 --seq-len 4096 --d-model 1024 --n-heads 16 --n-layers 8 \
  --warmup-steps 20 --timed-steps 80 --dtype bfloat16
```

### Integration (automation / CI / dashboards)

Use `--output-json` to emit machine-readable results:

```bash
python benchmarks/benchmark_fa2_vs_fa3_mlp_transformer.py \
  --preset mi300x \
  --output-json fa2_vs_fa3_mi300x.json
```

The JSON contains:

- resolved benchmark config
- raw FA2/FA3 metrics (step time, throughput, memory, loss, backend symbol)
- summary ratios (FA3 speedup and memory delta)

You can archive this JSON per run for trend analysis and regression alerts.

---

## 4) Quick troubleshooting

- **`ModuleNotFoundError: torch` during build**  
  Install ROCm torch first (see prerequisites section).

- **`ModuleNotFoundError: wheel` during build**  
  `uv pip install wheel`

- **Build OOM / machine freezes during compile**  
  Reduce parallel jobs, e.g. `export MAX_JOBS=2`.

- **Triton backend not selected**  
  Ensure `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` is exported **before** install and before runtime.

---

## Notes

- This repository is based on `Dao-AILab/flash-attention` and is narrowed toward ROCm + FA3 execution workflows.
- For original papers and broader project details, see upstream:  
  https://github.com/Dao-AILab/flash-attention
