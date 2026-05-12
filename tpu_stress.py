
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import time

def stress():
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Large tensors to stress the TPU
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    
    print("Starting stress loop...")
    for i in range(100):
        c = torch.matmul(a, b)
        if i % 10 == 0:
            print(f"Iteration {i}")
            # Ensure the computation is synced
            xm.mark_step()
    
    print("Done.")

if __name__ == "__main__":
    stress()
