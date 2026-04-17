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

## Cross-GPU replication: RTX 5080 (local) vs RTX 5090 (cloud)

All 14 workloads were re-run on an RTX 5090 (32 GB, dual-GPU cloud host, GPU 0; driver 580.142). Same registry, same 300 s duration, same TinyLlama-1.1B model. Gives us paired telemetry for each workload across two Blackwell cards.

### Aggregate medians across the 14 matched workloads

| Variable              | RTX 5080 | RTX 5090 |   Δ     |
|-----------------------|---------:|---------:|--------:|
| gpu_utilization_pct   |    68.0  |    54.5  |  −13.5  |
| mem_utilization_pct   |    36.0  |    21.5  |  −14.5  |
| mem_used_mb           | 13249.5  | 12230.0  | −1019.5 |
| power_draw_w          |   156.4  |   228.1  |  +71.7  |
| temperature_c         |    58.0  |    43.0  |  −15.0  |
| sm_clock_mhz          |  2733.5  |  2913.5  | +180.0  |
| mem_clock_mhz         | 14801.0  | 13801.0  | −1000.0 |

(Medians over samples with `gpu_utilization_pct > 5`, then aggregated across workloads.)

### Key findings

1. **5090 is faster on the same job.** Lower GPU-util (−14 pts) and lower mem-util (−15 pts) across the board, with SM clock boosting ~180 MHz higher. The 5090 is not saturated by these 1B-scale workloads.

2. **Power tracks capability, not workload.** 5090 draws ~72 W more on the same code — consistent with 575 W TDP vs. 360 W TDP. Gap widens on LLM-heavy runs (whitebox_lora, whitebox_oracle) where the 5090 has headroom to boost. Shapes are similar; magnitudes differ.

3. **GPU util traces diverge in shape, not just scale.** Same work completes faster on the 5090, so it idles more between iterations:
   - `adversarial_K_online_learning`: 5080 pegged at ~100 %, 5090 swinging 0–100 % — 5090 has only 133 "active" samples (util > 5 %) vs 5080's 298 in the same 300 s run.
   - `whitebox_inference`: 5080 ~66 %, 5090 ~50 %.
   - `whitebox_lora_*`: 5080 ~80–85 %, 5090 ~55–60 %.

4. **Memory footprint is lower on the 5090** (~1 GB less on average for the same workload). Different driver (580.142 vs 581.95) and a larger total pool (32 GB vs 16 GB) change CUDA caching/alloc behavior.

5. **Memory clock difference is a spec/reporting quirk, not a bug.** 5080 reports 14801 MHz, 5090 reports 13801 MHz — both GDDR7 at similar effective bandwidths, NVML reports the nominal base memory clock. Not a data-quality issue; don't use as a discriminating feature.

6. **All NVML fields readable on both cards.** Fan, encoder, decoder, PCIe — no NaN or `-1` sentinels on either device. The 9-variable check_telemetry.py script works identically on 5080 and 5090.

### Implications for classifier feature engineering

The same workload produces meaningfully different traces on different GPU classes. Raw magnitudes (`mem_used_mb`, `mem_clock_mhz`, `power_draw_w`, `gpu_utilization_pct`) are per-GPU-specific. Preferred normalized features:

- `mem_used_mb / mem_total_mb` — fraction of total memory, not absolute MB
- `power_draw_w / TDP` — fraction of rated power (requires per-GPU TDP table)
- `gpu_utilization_pct` trajectory features — variance, FFT peaks, autocorrelation — rather than raw level, since the level shifts with GPU class
- Power/util *shape* features (spike frequency, spike amplitude ratio) should generalize; *level* features will not

For Round 6, adding `mem_total_mb`-relative features and a TDP lookup would let the classifier transfer across GPU classes without needing per-GPU training data for every card.

## Also found / fixed this round

- Bug in `classifier/classifier_oracle.py:214`: was calling `.index()` on a list of class-name strings with an integer argument returned by `LabelEncoder.transform()`. Fixed to use `int(transform([...])[0])` directly. Oracle runs now work.
- `peft` was missing from the venv; installed `peft==0.19.1` for WB-L LoRA workloads.
- `adversarial_K_online_learning` no longer crashes at batch 128 (the mitigation from the earlier round held).
