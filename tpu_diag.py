
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
try:
    import tpu_info
    from tpu_info import device, metrics
    has_tpu_info = True
except ImportError:
    has_tpu_info = False

def diag():
    print(f"PID: {os.getpid()}")
    print(f"Torch version: {torch.__version__}")
    
    device = xm.xla_device()
    print(f"XLA Device: {device}")
    
    if has_tpu_info:
        try:
            chip_type, num_chips = device.get_local_chips()
            print(f"TPU Chip: {chip_type}, count: {num_chips}")
            # We know get_chip_usage failed earlier due to connection refused
            # Let's see if it works after XLA initialization
            usage = metrics.get_chip_usage(chip_type)
            print(f"TPU Usage: {usage}")
        except Exception as e:
            print(f"TPU Info error: {e}")
    else:
        print("tpu_info not installed")

if __name__ == "__main__":
    diag()
