import os

# Ensure AMD Triton backend path is selected at import time.
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

import torch
import flash_attn.flash_attn_interface as fai
from flash_attn import flash_attn_func

print("FLASH_ATTENTION_TRITON_AMD_ENABLE=", os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE"))
print("torch version:", torch.__version__)
print("torch HIP version:", torch.version.hip)
print("torch CUDA version field:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("No ROCm/CUDA device available to run FA3 test.")

device_index = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device_index)
print("device index:", device_index)
print("device name:", torch.cuda.get_device_name(device_index))
print("device gfx arch:", getattr(props, "gcnArchName", "<unavailable>"))
print("visible memory (GiB):", round(props.total_memory / (1024**3), 2))
print("flash-attn backend symbol:", getattr(fai.flash_attn_gpu, "__name__", str(fai.flash_attn_gpu)))

q = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, 128, 64, 64, dtype=torch.bfloat16, device="cuda")

try:
    out = flash_attn_func(q, k, v, causal=True)
    print("FA3 forward success; output dtype:", out.dtype, "shape:", tuple(out.shape))
except Exception as e:
    print("FA3 execution failed:", repr(e))
    raise
