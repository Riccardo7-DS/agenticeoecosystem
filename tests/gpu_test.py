import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)
print("Torch version:", torch.__version__)
print("Tensor Core FP16 GEMM test:", end=" ")

a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
try:
    c = torch.matmul(a, b)
    print("SUCCESS")
except RuntimeError as e:
    print("FAIL", e)