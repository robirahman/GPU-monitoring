#!/usr/bin/env python3
"""LLM inference on a SECOND engine: HuggingFace generate, sharded by accelerate.

WHY THIS EXISTS. Every LLM inference stream in the corpus is vLLM. Held out by
family, `infershard_qwen` -- 14 workloads, a quarter of all honest multi-node
windows -- is falsely flagged as training on 71% of its windows, and it is the
single worst false-alarm source we have. Fourteen workloads did not help because
they are one program swept over context length and batch size; the detector has
effectively seen one inference engine and treats a novel point in that sweep as
something new.

HF `generate` is a genuinely different engine, not a different flag:

  * no paged KV cache -- a contiguous per-sequence cache that grows as it decodes,
    so the memory curve rises through a batch instead of being flat
  * no continuous batching -- the whole batch steps together and the slowest
    sequence sets the pace, so short sequences idle rather than being replaced
  * no CUDA-graph capture and no preallocation -- see below, this is the point
  * layer-sequential sharding across GPUs, so at any instant ONE device is
    computing and the rest wait, versus vLLM's tensor-parallel all-reduce per layer

That last pair matters more than the throughput difference. Section 33c of the
TPU findings recorded that every vLLM cell sits at 74-79 GB of resident memory
because `--gpu-memory-utilization` preallocates the KV cache to a fixed fraction
of the card, and 33d that the detector reads absolute memory level rather than
periodicity. So "inference" in this corpus is partly just "a process holding 77 GB".
accelerate's `device_map` allocates what the weights actually need and nothing
more, and `--max-memory-gb` moves that number on purpose -- which is how this cell
turns memory footprint from a confound into a swept axis.

TENSOR vs PIPELINE. `device_map="balanced"` splits the model by LAYER across the
visible GPUs, which is pipeline-style: activations hop device to device, one small
transfer per boundary. vLLM's TP8 splits every matmul and all-reduces each layer.
Both are "sharded 72B inference" to an operator; they are completely different on
the wire. That contrast is the cell's whole reason for being.

LABELLING: cells using this script are named s16_inferhf_* -- contains "infer",
contains none of the training keywords ("train", "gpt", "bert", "fsdp_", "ddp_",
"resnet_cifar", "mlp_cifar", "adversarial_"), so threeway_label() routes them to
ml_inference. Verified in classifier/rebuttal_analysis/test_cell_label_routing.py.
STEP lines carry kind="infer".

Runs under venv-main ($PY), NOT venv-vllm: it needs transformers + accelerate and
must not import vllm. Both are already installed (requirements_gcp_extra.txt).

    python workloads/llm_inference_hf_sharded.py --model qwen25_72b_fp16 \
        --duration 600 --batch-size 8 --max-new-tokens 128
    python workloads/llm_inference_hf_sharded.py --model qwen25_72b_fp16 \
        --duration 600 --max-memory-gb 40      # footprint sweep
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from workloads._step import emit_step
    from workloads.llm_inference_vllm import VLLM_MODELS, PROMPTS
except ImportError:
    from _step import emit_step
    from llm_inference_vllm import VLLM_MODELS, PROMPTS


def build_batch(tok, batch_size, prompt_tokens, device):
    """A batch of roughly `prompt_tokens` tokens each, padded to the longest.

    Deliberately not padded to an exact fixed width: uniform prompt length is
    half of what makes the existing vLLM cells periodic, and HF generate pads to
    the batch max anyway, so letting the lengths vary is both more realistic and
    closer to what a real serving batch looks like.
    """
    texts = []
    for i in range(batch_size):
        s = PROMPTS[i % len(PROMPTS)]
        # Repeat until roughly the target, varying per sequence.
        reps = max(1, int(prompt_tokens / max(len(s.split()), 1) * (0.7 + 0.3 * (i % 4))))
        texts.append(" ".join([s] * reps))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=prompt_tokens)
    return {k: v.to(device) for k, v in enc.items()}


def main():
    ap = argparse.ArgumentParser(description="HF generate, sharded by accelerate")
    ap.add_argument("--model", required=True, choices=list(VLLM_MODELS.keys()))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-memory-gb", type=float, default=0.0,
                    help="per-GPU cap handed to accelerate. 0 = let it use what "
                         "it needs. This is the footprint knob: the same model at "
                         "40 vs 60 vs unlimited spreads this family across memory "
                         "levels so the level stops predicting the class.")
    ap.add_argument("--device-map", default="balanced",
                    choices=["balanced", "sequential", "auto"])
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id, dtype, _min_gpus, _tp = VLLM_MODELS[args.model]
    dtype = args.dtype or dtype
    ngpu = torch.cuda.device_count()
    print(f"HF sharded inference: {args.model} ({hf_id}) over {ngpu} GPU(s), "
          f"device_map={args.device_map}, dtype={dtype}", flush=True)

    kw = {}
    if args.max_memory_gb > 0:
        kw["max_memory"] = {i: f"{args.max_memory_gb}GiB" for i in range(ngpu)}
        print(f"  per-GPU cap {args.max_memory_gb} GiB", flush=True)

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=getattr(torch, dtype), device_map=args.device_map, **kw)
    model.eval()
    print(f"  weights loaded in {time.time() - t_load:.0f}s; "
          f"resident {sum(torch.cuda.memory_allocated(i) for i in range(ngpu)) / 1e9:.1f} GB "
          f"across {ngpu} GPU(s)", flush=True)

    # The first device holds the embedding layer, so inputs go there.
    in_dev = next(model.parameters()).device
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0)))

    start = time.time()
    step = 0
    tokens_out = 0
    last = start
    while time.time() - start < args.duration:
        batch = build_batch(tok, args.batch_size, args.prompt_tokens, in_dev)
        with torch.inference_mode():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
        new = int(out.shape[0] * (out.shape[1] - batch["input_ids"].shape[1]))
        tokens_out += new
        step += 1
        emit_step("infer", step, tokens=new, passes=1, node=node_rank, rank=0,
                  engine="hf")
        now = time.time()
        if now - last >= 30:
            el = now - start
            print(f"  t={el:6.0f}s  batches={step}  out={tokens_out} tok "
                  f"({tokens_out / max(el, 1):.1f} tok/s)  "
                  f"peak {max(torch.cuda.max_memory_allocated(i) for i in range(ngpu)) / 1e9:.1f} GB/GPU",
                  flush=True)
            last = now

    el = time.time() - start
    print(f"{args.model} hf-sharded done: {step} batches, {tokens_out} output tokens, "
          f"{tokens_out / max(el, 1):.1f} tok/s, {el:.0f}s wall", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
