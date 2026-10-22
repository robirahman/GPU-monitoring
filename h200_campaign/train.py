#!/usr/bin/env python3
"""Instrumented training on real data, ordinary or with a useful-learning evasion.

Families: mlp / resnet (CIFAR-10), bert (SST-2), lora (causal LM on WikiText-2).
Modes:
  ordinary     standard training
  diluted      forward pass on every batch, backward + optimizer update on 1 of --dilution
  interleave   --train-sec of training, then --other-sec of filler (cuFFT or forward-only inference)
  shape        duty-cycle pacing: each --period-ms, train for --duty of it, sleep the rest;
               --power-cap W additionally caps the board via nvidia-smi -pl (restored on exit)

Stops after --updates optimizer updates or --duration seconds of training phase,
whichever comes first. Validation loss/accuracy on a held-out split every
--eval-every updates and at the end. Writes events (EventLog) and a metrics
JSON (--metrics) with useful samples/tokens, updates, timings, data fingerprints,
model revision and seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h200_campaign.events import EventLog  # noqa: E402

DATA = Path(os.environ.get("H200_DATA", Path.home() / "campaign" / "data"))


def sha(t: torch.Tensor | np.ndarray | bytes) -> str:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().contiguous().numpy()
    if isinstance(t, np.ndarray):
        t = t.tobytes()
    return hashlib.sha256(t).hexdigest()[:16]


# --------------------------------------------------------------------------- tasks
class Task:
    """batch(i) -> inputs; loss(model, batch) -> loss; evaluate(model) -> metrics."""
    unit = "samples"


class CifarTask(Task):
    def __init__(self, arch, bs, device, seed):
        from datasets import load_dataset
        self.n_cls = 100 if arch == "effnet" else 10
        ds = load_dataset("uoft-cs/cifar100" if self.n_cls == 100 else "uoft-cs/cifar10")
        lab = "fine_label" if self.n_cls == 100 else "label"
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

        def to(split):
            imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in split["img"]])
            x = (torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255 - mean) / std
            return x.to(device), torch.tensor(split[lab]).to(device), imgs
        self.xtr, self.ytr, raw_tr = to(ds["train"])
        self.xte, self.yte, raw_te = to(ds["test"])
        self.fingerprint = {"train": f"{ds['train']._fingerprint}/sha={sha(raw_tr)}/n={len(raw_tr)}",
                            "test": f"{ds['test']._fingerprint}/sha={sha(raw_te)}/n={len(raw_te)}"}
        if arch == "mlp":
            self.model = nn.Sequential(nn.Flatten(), nn.Linear(3072, 2048), nn.ReLU(), nn.Linear(2048, 1024),
                                       nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10))
            self.lr = 0.05
        elif arch == "effnet":
            from transformers import AutoModelForImageClassification
            m = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0", num_labels=100,
                                                                ignore_mismatched_sizes=True)
            self.hf_revision = getattr(m.config, "_commit_hash", None)
            self.model = m
            self.lr = 0.05
        else:
            import torchvision.models as tvm
            m = tvm.resnet18(num_classes=10)
            m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
            self.model = m
            self.lr = 0.1
        self.model = self.model.to(device).to(memory_format=torch.channels_last if arch != "mlp" else torch.contiguous_format)
        self.bs, self.g = bs, torch.Generator(device=device).manual_seed(seed)
        self.revision = getattr(self, "hf_revision", None) or f"torchvision-{arch}"
        self.augment = arch != "mlp"
        self.resize = 224 if arch == "effnet" else None
        self.hf = arch == "effnet"

    def _fwd(self, x):
        if self.resize:
            x = F.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
        o = self.model(pixel_values=x) if self.hf else self.model(x)
        return o.logits if self.hf else o

    def optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    def batch(self):
        idx = torch.randint(0, self.xtr.shape[0], (self.bs,), device=self.xtr.device, generator=self.g)
        x = self.xtr[idx]
        if self.augment:
            flip = torch.rand(self.bs, device=x.device, generator=self.g) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        return x, self.ytr[idx]

    def loss(self, b):
        x, y = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.cross_entropy(self._fwd(x), y)

    def size(self, b):
        return b[0].shape[0]

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, loss, n = 0, 0.0, 0
        step = 250 if self.resize else 1000
        for i in range(0, self.xte.shape[0], step):
            x, y = self.xte[i:i + step], self.yte[i:i + step]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self._fwd(x)
            loss += F.cross_entropy(o.float(), y, reduction="sum").item()
            correct += (o.argmax(1) == y).sum().item()
            n += y.numel()
        self.model.train()
        return {"val_loss": loss / n, "val_acc": correct / n, "val_n": n}


class BertTask(Task):
    def __init__(self, bs, device, seed, max_len=128):
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = "google-bert/bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(name)
        ds = load_dataset("nyu-mll/glue", "sst2")
        enc = lambda s: tok(s["sentence"], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        tr, va = ds["train"], ds["validation"]
        a, b = enc(tr[:]), enc(va[:])
        self.tr = {k: v.to(device) for k, v in a.items()} | {"labels": torch.tensor(tr["label"], device=device)}
        self.va = {k: v.to(device) for k, v in b.items()} | {"labels": torch.tensor(va["label"], device=device)}
        self.fingerprint = {"train": tr._fingerprint, "validation": va._fingerprint}
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
        self.revision = getattr(self.model.config, "_commit_hash", None) or name
        self.bs, self.g, self.lr = bs, torch.Generator(device=device).manual_seed(seed), 2e-5

    def optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

    def batch(self):
        n = self.tr["labels"].shape[0]
        idx = torch.randint(0, n, (self.bs,), device=self.tr["labels"].device, generator=self.g)
        return {k: v[idx] for k, v in self.tr.items()}

    def loss(self, b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**b).loss

    def size(self, b):
        return b["labels"].shape[0]

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct, loss, n = 0, 0.0, 0
        N = self.va["labels"].shape[0]
        for i in range(0, N, 256):
            b = {k: v[i:i + 256] for k, v in self.va.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.model(**b)
            loss += o.loss.float().item() * b["labels"].numel()
            correct += (o.logits.argmax(-1) == b["labels"]).sum().item()
            n += b["labels"].numel()
        self.model.train()
        return {"val_loss": loss / n, "val_acc": correct / n, "val_n": n}


class LoraTask(Task):
    unit = "tokens"

    def __init__(self, model_name, bs, seq, device, seed, rank=16, dataset="wikitext", eval_tokens=65536):
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        if dataset == "wikitext":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
            splits = {"train": ds["train"], "validation": ds["validation"]}
        else:   # TinyStories: first 200k training stories (~40M tokens) and the validation split
            ds = load_dataset("roneneldan/TinyStories")
            splits = {"train": ds["train"].select(range(200_000)), "validation": ds["validation"]}
        eos = tok.eos_token or ""

        def ids(split):
            text = (eos + "\n\n").join(t for t in splits[split]["text"] if t.strip())
            return torch.tensor(tok(text, return_tensors=None)["input_ids"], dtype=torch.long)
        self.tr, self.va = ids("train").to(device), ids("validation").to(device)
        self.fingerprint = {"train": splits["train"]._fingerprint + f"/tokens={self.tr.numel()}",
                            "validation": splits["validation"]._fingerprint + f"/tokens={self.va.numel()}",
                            "dataset": dataset}
        self.eval_tokens = eval_tokens
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        self.revision = getattr(base.config, "_commit_hash", None) or model_name
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        # every linear layer except the output head, identically for every LLM lineage
        tm = sorted({n.split(".")[-1] for n, mod in base.named_modules() if isinstance(mod, nn.Linear)}
                    - {"lm_head", "embed_out"})
        self.model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                                     target_modules=tm, task_type="CAUSAL_LM"))
        self.bs, self.seq, self.lr = bs, seq, 2e-4
        self.g = torch.Generator(device=device).manual_seed(seed)

    def optimizer(self):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def batch(self):
        st = torch.randint(0, self.tr.numel() - self.seq - 1, (self.bs,), device=self.tr.device, generator=self.g)
        x = torch.stack([self.tr[s:s + self.seq] for s in st.tolist()])
        return x

    def loss(self, x):
        return self.model(input_ids=x, labels=x).loss

    def size(self, x):
        return x.numel()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        loss, n = 0.0, 0
        v = self.va[:self.eval_tokens]
        for i in range(0, v.numel() - self.seq, self.seq * 8):
            chunk = v[i:i + self.seq * 8]
            x = chunk[: (chunk.numel() // self.seq) * self.seq].view(-1, self.seq)
            loss += self.model(input_ids=x, labels=x).loss.float().item() * x.numel()
            n += x.numel()
        self.model.train()
        return {"val_loss": loss / n, "val_ppl": float(np.exp(loss / n)), "val_n": n}

    def trainable_state(self):
        return {k: v for k, v in self.model.state_dict().items() if "lora_" in k}


# --------------------------------------------------------------------------- filler
class Filler:
    """Non-learning GPU work for interleaving: cuFFT, or forward-only inference of the same model."""

    def __init__(self, kind, task, device):
        self.kind, self.task = kind, task
        if kind == "fft":
            self.x = torch.randn(16, 1024, 1024, dtype=torch.complex64, device=device)

    @torch.no_grad()
    def step(self):
        if self.kind == "fft":
            torch.fft.ifft2(torch.fft.fft2(self.x))
        else:
            self.task.loss(self.task.batch())


# --------------------------------------------------------------------------- main
def set_power_cap(watts):
    out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(watts)], check=True, capture_output=True)
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["mlp", "resnet", "bert", "lora", "effnet"])
    ap.add_argument("--dataset", default="wikitext", choices=["wikitext", "tinystories"], help="lora corpus")
    ap.add_argument("--eval-tokens", type=int, default=65536)
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                    help="cosine: linear warmup then cosine decay over --updates (needs a finite update budget)")
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--ckpt-every-sec", type=float, default=0.0, help="save trainable weights this often (file I/O)")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="lora base model")
    ap.add_argument("--mode", default="ordinary", choices=["ordinary", "diluted", "interleave", "shape", "cover_only"])
    ap.add_argument("--updates", type=int, default=10**9)
    ap.add_argument("--duration", type=float, default=300.0, help="wall seconds of the training loop")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=10**9, help="updates between evaluations")
    ap.add_argument("--eval-every-sec", type=float, default=30.0, help="wall seconds between evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dilution", type=int, default=5)
    ap.add_argument("--train-sec", type=float, default=5.0)
    ap.add_argument("--other-sec", type=float, default=15.0)
    ap.add_argument("--other", default="fft", choices=["fft", "infer"])
    ap.add_argument("--duty", type=float, default=0.3)
    ap.add_argument("--period-ms", type=float, default=200.0)
    ap.add_argument("--power-cap", type=int, default=0)
    ap.add_argument("--warmup-updates", type=int, default=5)
    ap.add_argument("--start-at-mono", type=float, default=None,
                    help="after loading and the initial eval, wait until this CLOCK_MONOTONIC second to start training")
    ap.add_argument("--gpu-marks", type=int, default=1, help="CUDA-event timestamp every update (0=off)")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--events", default=None)
    args = ap.parse_args()

    log = EventLog(args.events, stream="workload", gpu=False)
    log.emit("process_start", argv=sys.argv, family=args.family, mode=args.mode)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    bs = args.batch_size or {"mlp": 512, "resnet": 256, "bert": 32, "lora": 8, "effnet": 128}[args.family]
    log.emit("load_start")
    if args.family in ("mlp", "resnet", "effnet"):
        task = CifarTask(args.family, bs, dev, args.seed)
    elif args.family == "bert":
        task = BertTask(bs, dev, args.seed)
    else:
        task = LoraTask(args.model, bs, args.seq_len, dev, args.seed, dataset=args.dataset, eval_tokens=args.eval_tokens)
    opt = task.optimizer()
    sched = None
    if args.lr_schedule == "cosine":
        assert args.updates < 10**8, "cosine schedule needs --updates"
        warm = max(1, int(args.warmup_frac * args.updates))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 *
                                                  (1 + math.cos(math.pi * min(u, args.updates) / args.updates)))
    torch.cuda.synchronize()
    log.emit("load_end", batch_size=bs, unit=task.unit, revision=task.revision, data=task.fingerprint)

    gl = EventLog(args.events, stream="workload_gpu", gpu=True) if args.gpu_marks else None
    mark = (lambda ev, **kv: gl.emit(ev, gpu_mark=True, _depth=2, **kv)) if gl else (lambda ev, **kv: log.emit(ev, _depth=2, **kv))
    filler = Filler(args.other, task, dev) if args.mode in ("interleave", "cover_only") else None
    ckpt_dir = Path(args.ckpt_dir or Path(args.metrics).parent / "ckpt")
    last_ckpt, ckpt_bytes = None, 0
    prev_cap = set_power_cap(args.power_cap) if args.power_cap else None
    if prev_cap:
        log.emit("power_cap_set", watts=args.power_cap, previous=prev_cap)

    evals, useful, fwd_only, updates, micro = [], 0, 0, 0, 0
    phase_train = True
    try:
        evals.append({"updates": 0, "t": 0.0, **task.evaluate()})
        log.emit("eval", **evals[-1])
        task.model.train()
        if args.start_at_mono is not None:
            torch.cuda.synchronize()
            lateness = time.monotonic() - args.start_at_mono
            log.emit("start_gate", target_mono_s=args.start_at_mono, late_s=max(0.0, lateness), ok=lateness <= 0)
            while time.monotonic() < args.start_at_mono:
                time.sleep(min(0.01, max(0.0, args.start_at_mono - time.monotonic())))
        # duration, eval cadence and phase timers all count from the start of training
        t_start = phase_t0 = last_eval = last_ckpt = time.monotonic()
        if args.mode == "cover_only":
            # control: the attack's cover work alone, no training at all
            log.emit("cover_start", kind=args.other)
            n_cover = 0
            while time.monotonic() - t_start < args.duration:
                filler.step()
                n_cover += 1
            torch.cuda.synchronize()
            log.emit("cover_end", steps=n_cover)
            args.updates = 0
        else:
            mark("train_start", note="first training work submitted next")
        while updates < args.updates and time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            if args.mode == "interleave":
                span = args.train_sec if phase_train else args.other_sec
                if now - phase_t0 >= span:
                    phase_train = not phase_train
                    phase_t0 = now
                    mark("phase_train" if phase_train else "phase_other", kind=args.other)
                if not phase_train:
                    filler.step()
                    continue
            if args.mode == "shape":
                pos = ((now - t_start) * 1000.0) % args.period_ms
                if pos >= args.duty * args.period_ms:
                    torch.cuda.synchronize()
                    time.sleep((args.period_ms - pos) / 1000.0)
                    continue
            b = task.batch()
            micro += 1
            if args.mode == "diluted" and micro % args.dilution != 0:
                with torch.no_grad():
                    task.loss(b)
                fwd_only += task.size(b)
                continue
            loss = task.loss(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            updates += 1
            useful += task.size(b)
            if updates == 1:
                mark("first_update_submitted")
            elif updates == args.warmup_updates:
                mark("warmup_end", updates=updates)
            elif gl is not None and args.gpu_marks:
                gl.emit("optimizer_update", gpu_mark=True, updates=updates)
            if updates % 50 == 0 and gl:
                gl.poll_gpu()
            if updates % args.eval_every == 0 or time.monotonic() - last_eval >= args.eval_every_sec:
                log.emit("eval_start", updates=updates)
                evals.append({"updates": updates, "t": time.monotonic() - t_start, "train_loss": float(loss.item()),
                              **task.evaluate()})
                log.emit("eval_end", **evals[-1])
                last_eval = time.monotonic()
            if args.ckpt_every_sec and time.monotonic() - last_ckpt >= args.ckpt_every_sec:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                log.emit("ckpt_start", updates=updates)
                st = task.trainable_state() if hasattr(task, "trainable_state") else task.model.state_dict()
                f = ckpt_dir / "latest.pt"
                torch.save({"updates": updates, "model": st, "opt": opt.state_dict()}, f)
                ckpt_bytes += f.stat().st_size
                log.emit("ckpt_end", updates=updates, bytes=f.stat().st_size)
                last_ckpt = time.monotonic()
        if args.mode != "cover_only":
            mark("train_end", updates=updates)
        torch.cuda.synchronize()
        log.emit("train_end_synced", updates=updates)
        evals.append({"updates": updates, "t": time.monotonic() - t_start, "final": True, **task.evaluate()})
        log.emit("eval", **evals[-1])
    finally:
        if prev_cap:
            subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(int(prev_cap))], capture_output=True)
            log.emit("power_cap_restored", watts=prev_cap)
    wall = time.monotonic() - t_start
    m = {"family": args.family, "mode": args.mode, "args": vars(args), "batch_size": bs, "unit": task.unit,
         "updates": updates, "useful_units": useful, "forward_only_units": fwd_only, "ckpt_bytes": ckpt_bytes,
         "lr_schedule": args.lr_schedule, "update_budget": args.updates if args.updates < 10**8 else None,
         "wall_s_train_loop": wall, "useful_units_per_s": useful / wall if wall else None,
         "evals": evals, "data_fingerprint": task.fingerprint, "model_revision": task.revision,
         "seed": args.seed, "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
         "objective": {"mlp": "CIFAR-10 cross-entropy", "resnet": "CIFAR-10 cross-entropy", "bert": "SST-2 cross-entropy",
                       "lora": f"{args.dataset} causal LM cross-entropy",
                       "effnet": "CIFAR-100 cross-entropy (ImageNet-pretrained EfficientNet-B0)"}[args.family]}
    Path(args.metrics).write_text(json.dumps(m, indent=1, default=str))
    if gl:
        gl.close()
    log.emit("process_exit", updates=updates)
    log.close()


if __name__ == "__main__":
    main()
