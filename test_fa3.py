import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device="cuda")

try:
    output = flash_attn_func(q, k, v, causal=True)
    print("Flash Attention 3 forward pass executed successfully via Triton backend.")
except Exception as e:
    print("Execution failed:", e)
