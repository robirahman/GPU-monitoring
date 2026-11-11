#!/usr/bin/env python3
"""Instrumented single-stream inference serving with a request-rate schedule.

  --model phi3     microsoft/Phi-3-mini-4k-instruct, prompts from tatsu-lab/alpaca
  --model whisper  openai/whisper-large-v3-turbo, audio from LibriSpeech clean/validation

--schedule "SECONDS:RATE,..." is a sequence of segments with Poisson arrivals at
RATE requests/s (0 = idle gap). Requests are served one at a time in arrival
order; queueing delay is recorded. Each completed request emits one event with
arrival/start/end monotonic ns. Whisper outputs are scored (WER vs reference);
Phi-3 outputs are kept as samples so success can be checked.
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h200_campaign.events import EventLog  # noqa: E402

STOP = False


def _sig(_n, _f):
    global STOP
    STOP = True


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.lower().split(), hyp.lower().replace(",", "").replace(".", "").split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


class Phi3:
    name = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(self, rng):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.bfloat16).cuda().eval()
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        self.items = [(r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")) for r in ds]
        self.fingerprint = ds._fingerprint
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.rng = rng

    @torch.no_grad()
    def request(self):
        prompt = self.rng.choice(self.items)
        max_new = self.rng.choice([32, 64, 128, 256])
        msgs = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].cuda()
        out = self.model.generate(ids, attention_mask=enc["attention_mask"].cuda(), max_new_tokens=max_new,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        return {"prompt_tokens": int(ids.shape[1]), "new_tokens": int(new.numel()), "max_new": max_new,
                "text": self.tok.decode(new, skip_special_tokens=True)[:200]}


class Whisper:
    name = "openai/whisper-large-v3-turbo"

    def __init__(self, rng):
        from datasets import Audio, load_dataset
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.name, torch_dtype=torch.float16).cuda().eval()
        ds = load_dataset("openslr/librispeech_asr", "clean", split="validation").cast_column("audio", Audio(decode=False))
        self.ds, self.fingerprint = ds, ds._fingerprint
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.rng = rng

    @torch.no_grad()
    def request(self):
        import io
        import soundfile as sf
        r = self.ds[self.rng.randrange(len(self.ds))]
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000, sr
        feats = self.proc(a, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        ids = self.model.generate(feats, language="en", task="transcribe")
        hyp = self.proc.batch_decode(ids, skip_special_tokens=True)[0]
        e, n = wer(r["text"], hyp)
        return {"audio_s": len(a) / 16000, "new_tokens": int(ids.shape[1]), "wer_errors": e, "wer_words": n,
                "text": hyp[:200]}


class CausalLM:
    """Base (non-chat) decoder LLM serving: Alpaca instructions as plain-text prompts."""

    def __init__(self, rng, name):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        self.items = [f"Instruction: {r['instruction']}\n{r['input']}\nResponse:" for r in ds]
        self.fingerprint, self.revision, self.rng = ds._fingerprint, getattr(self.model.config, "_commit_hash", None), rng

    @torch.no_grad()
    def request(self):
        max_new = self.rng.choice([32, 64, 128, 256])
        enc = self.tok(self.rng.choice(self.items), return_tensors="pt").to("cuda")
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        return {"prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": int(new.numel()), "max_new": max_new,
                "text": self.tok.decode(new, skip_special_tokens=True)[:200]}


class Wav2Vec2:
    name = "facebook/wav2vec2-large-960h-lv60-self"

    def __init__(self, rng):
        from datasets import Audio, load_dataset
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained(self.name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.name, torch_dtype=torch.float16).cuda().eval()
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test").cast_column("audio", Audio(decode=False))
        self.ds, self.fingerprint = ds, ds._fingerprint
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.rng = rng

    @torch.no_grad()
    def request(self):
        import io
        import soundfile as sf
        r = self.ds[self.rng.randrange(len(self.ds))]
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        x = self.proc(a, sampling_rate=sr, return_tensors="pt").input_values.cuda().half()
        hyp = self.proc.batch_decode(self.model(x).logits.argmax(-1))[0]
        e, n = wer(r["text"], hyp)
        return {"audio_s": len(a) / sr, "new_tokens": 0, "wer_errors": e, "wer_words": n, "text": hyp[:200]}


class Bark:
    name = "suno/bark-small"

    def __init__(self, rng):
        from datasets import load_dataset
        from transformers import AutoProcessor, BarkModel
        self.proc = AutoProcessor.from_pretrained(self.name)
        self.model = BarkModel.from_pretrained(self.name, torch_dtype=torch.float16).cuda().eval()
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        self.items = [" ".join(o.split()[:25]) for o in ds["output"] if len(o.split()) >= 8]
        self.fingerprint, self.revision, self.rng = ds._fingerprint, getattr(self.model.config, "_commit_hash", None), rng

    @torch.no_grad()
    def request(self):
        text = self.rng.choice(self.items)
        x = self.proc(text, voice_preset=f"v2/en_speaker_{self.rng.randrange(10)}").to("cuda")
        audio = self.model.generate(**x)
        sr = self.model.generation_config.sample_rate
        return {"text_words": len(text.split()), "audio_s": audio.shape[-1] / sr, "new_tokens": 0, "text": text[:200]}


class DiT:
    name = "facebook/DiT-XL-2-256"

    def __init__(self, rng):
        from diffusers import DiTPipeline
        self.pipe = DiTPipeline.from_pretrained(self.name, torch_dtype=torch.float16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.fingerprint, self.revision, self.rng = "imagenet-1k class labels", None, rng

    @torch.no_grad()
    def request(self):
        n = self.rng.choice([1, 2, 4])
        steps = self.rng.choice([25, 50])
        labels = [self.rng.randrange(1000) for _ in range(n)]
        imgs = self.pipe(class_labels=labels, num_inference_steps=steps,
                         generator=torch.manual_seed(self.rng.randrange(2**31))).images
        return {"images": len(imgs), "steps": steps, "new_tokens": 0, "text": f"classes={labels}"}


class EffNetClassify:
    name = "google/efficientnet-b0"

    def __init__(self, rng):
        from datasets import load_dataset
        from transformers import AutoModelForImageClassification
        self.model = AutoModelForImageClassification.from_pretrained(self.name, torch_dtype=torch.float16).cuda().eval()
        ds = load_dataset("uoft-cs/cifar100", split="test")
        self.imgs = torch.tensor(np.stack([np.asarray(i) for i in ds["img"]])).permute(0, 3, 1, 2).contiguous()
        self.fingerprint, self.revision, self.rng = ds._fingerprint, getattr(self.model.config, "_commit_hash", None), rng
        self.labels = self.model.config.id2label

    @torch.no_grad()
    def request(self):
        import torch.nn.functional as F
        n = self.rng.choice([1, 8, 32, 64])
        idx = torch.tensor([self.rng.randrange(self.imgs.shape[0]) for _ in range(n)])
        x = F.interpolate(self.imgs[idx].cuda().half() / 255, size=224, mode="bilinear")
        top = self.model(pixel_values=x).logits.argmax(-1).tolist()
        return {"images": n, "new_tokens": 0, "text": self.labels[top[0]][:80]}


MODELS = {"phi3": Phi3, "whisper": Whisper, "wav2vec2": Wav2Vec2, "bark": Bark, "dit": DiT, "effnet": EffNetClassify,
          "olmo": lambda rng: CausalLM(rng, "allenai/OLMo-2-0425-1B"),
          "pythia": lambda rng: CausalLM(rng, "EleutherAI/pythia-1b")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--schedule", default="120:0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--events", default=None)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sig)
    log = EventLog(args.events, stream="serve")
    log.emit("process_start", argv=sys.argv, model=args.model)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    log.emit("load_start")
    srv = MODELS[args.model](rng)
    torch.cuda.synchronize()
    log.emit("load_end", model=srv.name, revision=srv.revision, data=srv.fingerprint)
    log.emit("warmup_start")
    for _ in range(args.warmup):
        srv.request()
    torch.cuda.synchronize()
    log.emit("warmup_end")

    segs = [(float(a), float(b)) for a, b in (s.split(":") for s in args.schedule.split(","))]
    arrivals, t = [], 0.0
    for dur, rate in segs:
        end = t + dur
        if rate > 0:
            nxt = t + rng.expovariate(rate)
            while nxt < end:
                arrivals.append(nxt)
                nxt += rng.expovariate(rate)
        t = end
    total = t
    log.emit("serve_start", schedule=segs, planned_requests=len(arrivals))
    t0 = time.monotonic()
    t0_ns = time.monotonic_ns()
    reqs, seg_i, seg_end = [], 0, segs[0][0]
    log.emit("segment_start", idx=0, seconds=segs[0][0], rate=segs[0][1])
    i = 0
    while True:
        now = time.monotonic() - t0
        while seg_i < len(segs) and now >= seg_end:
            seg_i += 1
            if seg_i < len(segs):
                log.emit("segment_start", idx=seg_i, seconds=segs[seg_i][0], rate=segs[seg_i][1])
                seg_end += segs[seg_i][0]
        if now >= total or STOP:
            break
        if i < len(arrivals) and arrivals[i] <= now:
            start_ns = time.monotonic_ns()
            res = srv.request()
            torch.cuda.synchronize()
            end_ns = time.monotonic_ns()
            arr_ns = t0_ns + int(arrivals[i] * 1e9)
            rec = {"i": i, "arrival_mono_ns": arr_ns, "start_mono_ns": start_ns, "end_mono_ns": end_ns,
                   "queue_ms": (start_ns - arr_ns) / 1e6, "service_ms": (end_ns - start_ns) / 1e6, **res}
            reqs.append(rec)
            log.emit("request", **rec)
            i += 1
        else:
            nxt = arrivals[i] if i < len(arrivals) else total
            time.sleep(max(0.0, min(nxt - now, seg_end - now, 0.05)))
    log.emit("serve_end", completed=len(reqs), planned=len(arrivals), stopped_early=STOP)
    m = {"model": srv.name, "revision": srv.revision, "data_fingerprint": srv.fingerprint, "schedule": segs,
         "seed": args.seed, "planned_requests": len(arrivals), "completed": len(reqs),
         "dropped_at_end": len(arrivals) - len(reqs),
         "new_tokens": sum(r["new_tokens"] for r in reqs),
         "service_ms_p50": float(np.median([r["service_ms"] for r in reqs])) if reqs else None,
         "queue_ms_p95": float(np.quantile([r["queue_ms"] for r in reqs], 0.95)) if reqs else None,
         "samples": [r["text"] for r in reqs[:5]]}
    if args.model in ("whisper", "wav2vec2") and reqs:
        m["wer"] = sum(r["wer_errors"] for r in reqs) / max(1, sum(r["wer_words"] for r in reqs))
    Path(args.metrics).write_text(json.dumps(m, indent=1, default=str))
    log.emit("process_exit")
    log.close()


if __name__ == "__main__":
    main()
