import time
import argparse
from datetime import datetime, timezone
import torch

def log(msg): print(f"{datetime.now(timezone.utc).isoformat()} | {msg}", flush=True)

def memcpy_stress(seconds: int, mb: int):
    log(f"PHASE=memcpy start seconds={seconds} mb={mb}")
    size = mb * 1024 * 1024 // 4  # float32
    a = torch.empty(size, device="cuda", dtype=torch.float32)
    b = torch.empty(size, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    t_end = time.time() + seconds
    it = 0
    while time.time() < t_end:
        b.copy_(a)
        a.copy_(b)
        it += 1
    torch.cuda.synchronize()
    log(f"PHASE=memcpy end iters={it}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--idle", type=int, default=10)
    ap.add_argument("--memcpy", type=int, default=25)
    ap.add_argument("--mb", type=int, default=4096)  # 4GB copies (adjust if needed)
    args=ap.parse_args()

    log(f"GPU={torch.cuda.get_device_name(0)}")
    log(f"PHASE=idle start seconds={args.idle}")
    time.sleep(args.idle)
    log("PHASE=idle end")
    memcpy_stress(args.memcpy, args.mb)
    log(f"PHASE=idle start seconds={args.idle}")
    time.sleep(args.idle)
    log("PHASE=idle end")
    log("DONE")

if __name__=="__main__":
    assert torch.cuda.is_available()
    main()
