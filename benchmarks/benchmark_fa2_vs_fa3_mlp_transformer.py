#!/usr/bin/env python3
"""Benchmark FA2 vs FA3 for training a small MLP Transformer.

This script runs the same training workload twice in separate subprocesses:
- FA2 path: CK/CUDA backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE`)
- FA3 path: Triton AMD backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)

Using subprocesses is important because backend selection in
`flash_attn.flash_attn_interface` happens at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


PRESETS = {
    "default": {
        "batch_size": 4,
        "seq_len": 2048,
        "vocab_size": 32000,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "warmup_steps": 10,
        "timed_steps": 50,
        "lr": 1e-4,
        "dtype": "bfloat16",
    },
    "mi300x": {
        "batch_size": 4,
        "seq_len": 4096,
        "vocab_size": 32000,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "warmup_steps": 20,
        "timed_steps": 80,
        "lr": 1e-4,
        "dtype": "bfloat16",
    },
}


@dataclass
class BenchResult:
    backend: str
    backend_symbol: str
    device_name: str
    dtype: str
    step_time_ms_median: float
    step_time_ms_mean: float
    steps_per_second: float
    tokens_per_second: float
    peak_memory_gb: float
    final_loss: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FA2 vs FA3 on a toy MLP Transformer training loop."
    )

    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="mi300x",
        help="Benchmark preset. 'mi300x' is tuned for stronger MI300X signal by default.",
    )

    # Keep these override-friendly by defaulting to None, then apply preset below.
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--timed-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default=None,
        help="Activation dtype used with autocast during training.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write machine-readable benchmark output JSON.",
    )

    # Internal worker-only flags.
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend", choices=["fa2", "fa3"], help=argparse.SUPPRESS)

    args = parser.parse_args()

    preset_values = PRESETS[args.preset]
    for key, value in preset_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def _torch_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _run_worker(args: argparse.Namespace) -> BenchResult:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the active environment. "
            "Install ROCm/CUDA torch first (see README prerequisites), then rerun this benchmark."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    amp_dtype = _torch_dtype(args.dtype)

    # Import here so backend env has already been set by the subprocess launcher.
    import flash_attn.flash_attn_interface as fai
    from flash_attn import flash_attn_qkvpacked_func

    class FlashMHA(nn.Module):
        def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.dropout = dropout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seqlen, _ = x.shape
            qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
            out = flash_attn_qkvpacked_func(
                qkv, dropout_p=self.dropout, causal=True)
            out = out.reshape(bsz, seqlen, -1)
            return self.proj(out)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
            super().__init__()
            hidden = int(d_model * mlp_ratio)
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = FlashMHA(d_model, n_heads, dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, d_model, bias=False),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyMLPTransformer(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            mlp_ratio: float,
            dropout: float,
            seq_len: int,
        ) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            x = self.tok_emb(tokens) + self.pos_emb
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    model = TinyMLPTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        seq_len=args.seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tokens = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )

    total_steps = args.warmup_steps + args.timed_steps
    step_times_s: list[float] = []
    final_loss = float("nan")

    torch.cuda.reset_peak_memory_stats(device)

    for step in range(total_steps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = model(tokens, targets)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        final_loss = float(loss.detach().item())
        if step >= args.warmup_steps:
            step_times_s.append(dt)

    median_s = statistics.median(step_times_s)
    mean_s = statistics.fmean(step_times_s)
    tokens_per_step = args.batch_size * args.seq_len

    return BenchResult(
        backend=args.backend,
        backend_symbol=getattr(
            fai.flash_attn_gpu, "__name__", str(fai.flash_attn_gpu)),
        device_name=torch.cuda.get_device_name(torch.cuda.current_device()),
        dtype=args.dtype,
        step_time_ms_median=median_s * 1000.0,
        step_time_ms_mean=mean_s * 1000.0,
        steps_per_second=1.0 / median_s,
        tokens_per_second=tokens_per_step / median_s,
        peak_memory_gb=torch.cuda.max_memory_allocated(device) / (1024**3),
        final_loss=final_loss,
    )


def _launch_mode(args: argparse.Namespace, backend: str) -> BenchResult:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--backend",
        backend,
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--vocab-size",
        str(args.vocab_size),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--n-layers",
        str(args.n_layers),
        "--mlp-ratio",
        str(args.mlp_ratio),
        "--dropout",
        str(args.dropout),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
    ]

    env = os.environ.copy()
    if backend == "fa3":
        env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    else:
        env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "FALSE"

    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        message = (
            f"{backend} worker failed with exit code {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
        raise RuntimeError(message)

    marker = "RESULT_JSON="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
            return BenchResult(**payload)

    raise RuntimeError(
        f"No worker result found in output for backend={backend}.\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}"
    )


def _print_report(fa2: BenchResult, fa3: BenchResult, args: argparse.Namespace) -> None:
    speedup = fa2.step_time_ms_median / fa3.step_time_ms_median
    toks_speedup = fa3.tokens_per_second / fa2.tokens_per_second
    mem_delta_pct = 100.0 * \
        (fa3.peak_memory_gb - fa2.peak_memory_gb) / max(fa2.peak_memory_gb, 1e-9)

    print("\\n=== FA2 vs FA3: MLP Transformer Training Benchmark ===")
    print(
        "Config: "
        f"batch={args.batch_size}, seqlen={args.seq_len}, d_model={args.d_model}, "
        f"heads={args.n_heads}, layers={args.n_layers}, mlp_ratio={args.mlp_ratio}, "
        f"dtype={args.dtype}, warmup={args.warmup_steps}, timed={args.timed_steps}"
    )
    print()
    print(f"Device: {fa2.device_name}")
    print()
    print("Mode | Backend symbol | Median step (ms) | Steps/s | Tokens/s | Peak mem (GiB) | Final loss")
    print("--- | --- | ---: | ---: | ---: | ---: | ---:")
    print(
        f"FA2 | `{fa2.backend_symbol}` | {fa2.step_time_ms_median:.2f} | "
        f"{fa2.steps_per_second:.3f} | {fa2.tokens_per_second:,.0f} | "
        f"{fa2.peak_memory_gb:.3f} | {fa2.final_loss:.4f}"
    )
    print(
        f"FA3 | `{fa3.backend_symbol}` | {fa3.step_time_ms_median:.2f} | "
        f"{fa3.steps_per_second:.3f} | {fa3.tokens_per_second:,.0f} | "
        f"{fa3.peak_memory_gb:.3f} | {fa3.final_loss:.4f}"
    )
    print()
    print(f"FA3 speedup over FA2 (median step-time): {speedup:.3f}x")
    print(f"FA3 throughput gain (tokens/s): {toks_speedup:.3f}x")
    print(f"FA3 peak-memory delta vs FA2: {mem_delta_pct:+.2f}%")


def _build_output_payload(
    fa2: BenchResult, fa3: BenchResult, args: argparse.Namespace
) -> dict:
    return {
        "preset": args.preset,
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "mlp_ratio": args.mlp_ratio,
            "dropout": args.dropout,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "lr": args.lr,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "results": {
            "fa2": asdict(fa2),
            "fa3": asdict(fa3),
        },
        "summary": {
            "fa3_speedup_vs_fa2_step_time": fa2.step_time_ms_median / fa3.step_time_ms_median,
            "fa3_throughput_gain_vs_fa2_tokens_per_sec": fa3.tokens_per_second / fa2.tokens_per_second,
            "fa3_peak_memory_delta_pct_vs_fa2": (
                100.0 * (fa3.peak_memory_gb - fa2.peak_memory_gb) /
                max(fa2.peak_memory_gb, 1e-9)
            ),
        },
    }


def main() -> None:
    args = _parse_args()

    if args.worker:
        result = _run_worker(args)
        print("RESULT_JSON=" + json.dumps(result.__dict__))
        return

    fa2 = _launch_mode(args, "fa2")
    fa3 = _launch_mode(args, "fa3")
    _print_report(fa2, fa3, args)

    if args.output_json:
        payload = _build_output_payload(fa2, fa3, args)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote benchmark JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
