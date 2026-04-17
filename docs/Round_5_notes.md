# Round 5 Notes — White-Box Evasion on RTX 5080

Date: 2026-04-17
Hardware: NVIDIA GeForce RTX 5080 (16 GB, Blackwell), Windows 11, PyTorch nightly cu128
Model: TinyLlama-1.1B-Chat-v1.0 (largest 1B-param config that fits with optimizer states)

## Workloads run (all 300s, telemetry at 1 Hz)

- `whitebox_inference` — forward-only baseline
- `whitebox_diluted_N5 / N10 / N20 / N50` — full fine-tuning, 1/N batches train, grad ckpt
- `whitebox_diluted_N5_ckpt / N10_ckpt / N20_ckpt` — same without grad checkpointing
- `whitebox_lora_N5 / N10 / N20` — LoRA fine-tuning (rank 16), 1/N batches train
- `whitebox_oracle` — oracle-guided full training, classifier queried in real time, threshold P(train) < 0.3

## Key observation: oracle run produced **0 training steps**

With threshold P(train) < 0.3, the classifier returned P(train) = 0.995–0.999 for the entire 300s run on what was effectively TinyLlama-1.1B inference. Final counts: `train=0, infer=14165`. This is a false positive from the classifier's perspective.

### Why it happened

1. **Pre-allocated optimizer states.** WB-O calls `_pre_allocate_optimizer()` at startup, which runs one dummy backward+step to materialize AdamW buffers. Post-init memory: 6363 MB (vs ~2 GB for bare TinyLlama inference). This is a deliberate evasion trick to hide the "memory growth at first backward" signal, but it also means the memory footprint looks training-like from the very first telemetry sample.
2. **Out-of-distribution model size.** The classifier's training set is dominated by small-model workloads (ResNet-18/50, BERT-base, GPT-2 124M). Large LLM inference signatures (sustained near-TDP power, high util, large memory) are under-represented, so the classifier has no anchor for "this is what 1B-param inference looks like."
3. **Memory-dominant feature weighting (suspected).** If `mem_used_mb` / `mem_util_pct` dominate importance, the classifier is fooled by the pre-allocation trick. The distinguishing signal should be **power waveform variance**: training has periodic backward-pass spikes every N forwards; pure inference is steady.

## Recommendations for next round

### 1. Collect more real large-LLM inference data (user's primary recommendation)

The training set is thin on 1B+ parameter inference. Collect baselines across model sizes and architectures so the classifier has proper anchors:

- **TinyLlama-1.1B** — already have `whitebox_inference` from this run
- **Qwen 2.5 0.5B** — `qwen25_inference` exists in registry, confirm it's in training set
- **DeepSeek-R1-Distill-Qwen-1.5B** — `deepseek_r1_inference` exists, confirm
- **Whisper large-v3-turbo** — `whisper_inference` exists
- Consider adding: Llama 3.2 3B, Phi-3 Mini, Qwen 2.5 1.5B inference runs
- Vary batch size and sequence length to cover the memory/throughput envelope

Multiple repetitions per workload/config so the classifier sees variance within a class, not just across classes.

### 2. Retrain the classifier with the new data

Include this session's parquets:
- `whitebox_inference_*.parquet` (clean 1.1B inference)
- All `whitebox_diluted_*.parquet` and `whitebox_lora_*.parquet` (large-model training signatures)
- `whitebox_oracle_*.parquet` (telemetry even though 0 training steps occurred)

Along with any new inference data collected per (1).

### 3. Audit feature importance in `classifier/threeway_improved.py`

If the top-N features are memory-based (`mem_used_mb`, `mem_util_pct`, `mem_used_mb_mean`, etc.), the classifier is memory-footprint-driven and vulnerable to pre-allocation evasion. Ideal distinguishing features:

- `power_draw_w_std_30s` — training has periodic oscillation, inference is flatter
- `gpu_util_pct_fft_peak` — training has a dominant frequency matching backward+step cadence
- `sm_clock_mhz_std` / `power_draw_w_autocorr` — periodic signatures of backward passes
- Derivative/delta features over sliding windows

Consider feature engineering that explicitly targets the forward-only vs. forward+backward distinction rather than snapshot memory state.

### 4. Reframe WB-O as a partial defensive win

The oracle run produced **zero training steps** — the evader was rate-limited to pure inference. Even though the classifier's individual predictions were wrong (false positive on inference), the *defensive outcome* is what we care about: the adversary couldn't train.

The real failure mode to watch for in future runs: does WB-O find periods where P(train) drops below threshold and successfully squeezes training steps in? If so, those are the windows where classifier calibration matters most.

Metric to track per round: **oracle training throughput** (`train` counter at end of run). This round: 0. If future rounds show nonzero, those are successful evasion events.

### 5. Consider that pre-allocation itself is an anomaly worth flagging

The WB-O startup does one backward pass to allocate optimizer buffers, then tries to look like inference. This is detectable — no legitimate inference workload allocates AdamW state at startup. A feature like "memory footprint delta in first 30s after process start" could explicitly catch this. Or alternatively, don't treat it as a false positive — treat "training prep detected" as a valid signal in its own right.

## Also found / fixed this round

- Bug in `classifier/classifier_oracle.py:214`: was calling `.index()` on a list of class-name strings with an integer argument returned by `LabelEncoder.transform()`. Fixed to use `int(transform([...])[0])` directly. Oracle runs now work.
- `peft` was missing from the venv; installed `peft==0.19.1` for WB-L LoRA workloads.
- `adversarial_K_online_learning` no longer crashes at batch 128 (the mitigation from the earlier round held).
