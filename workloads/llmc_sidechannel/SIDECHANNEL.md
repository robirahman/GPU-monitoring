# Side-channel instrumentation for llm.c

Marks exactly one forward+backward on a hardware line so a power trace can be
aligned to computation without trusting cross-clock offsets, and optionally
reshapes the training cadence into inference-like bursts while performing
bit-identical arithmetic.

Added: `llmc/sidechannel.h` (new, self-contained). Modified: `train_gpt2.cu`
(include, six CLI flags, two CUDA events, instrumented micro-step loop,
init/teardown). Nothing else changed; default behaviour with no `-S*` flags is
byte-identical to upstream except for two `cudaEventRecord` calls per
micro-step.

Base commit: `f1e2ace`. Builds clean under nvcc 12.4, `-std=c++17 -O3
-DENABLE_BF16`, no new warnings.

---

## 1. Flags

| Flag | Meaning |
|---|---|
| `-Sd <path>` | USB-UART device whose **DTR** pin is the marker, e.g. `/dev/ttyUSB0` |
| `-Sl <path>` | CSV timeline log |
| `-Sw <int>` | workload shaping: `0` = training (default), `1` = modified-inference |
| `-Sg <int>` | idle gap between micro-steps, microseconds (mode 1 only) |
| `-Sj <int>` | uniform jitter added to that gap, microseconds (mode 1 only) |
| `-Sc <int>` | emit N marker calibration pulses at startup, then continue |
| `-Sm <int>` | hold a fixed extra GPU allocation, MiB (memory-footprint spoof) |

Wiring: DTR → a spare scope channel, GND → scope ground. DTR idles low and is
driven high for the marked region.

**`-Sm` (memory-footprint spoof).** A detector that keys on training's resident
memory footprint (weights + gradients + optimizer state + activations) is not
fooled by timing shaping alone. `-Sm N` `cudaMalloc`s and touches N MiB once at
startup (before any marked region) and frees it at teardown, so NVML
`memory.used` sits flat at a chosen level. Combined with a tiny micro-batch
(`-b 1`) plus gradient accumulation (`-d`) — which performs the *same* training
with ~1/16 the activation memory — the resident footprint can be driven down to
the optimizer-state floor and then padded to mimic a non-training workload
(e.g. `-b 1 -r 2 -Sm 1650` ≈ 4.8 GB, matching a ResNet-50 inference server on an
H200). Like the timing shaping, this changes only the memory *level*, never the
arithmetic.

---

## 2. Why there are no added `cudaDeviceSynchronize()` calls

This is the single most important correctness point, and it is a property of
llm.c rather than of this patch.

CUDA launches are asynchronous, so a marker placed around kernel launches would
normally bracket *launch* time, not *compute* time — deasserting while the GPU
is still busy. That failure mode would be invisible in the log and would
silently corrupt every measurement.

It does not arise here because **llm.c is already host-synchronous at every
phase boundary**:

| Function | Final statement |
|---|---|
| `gpt2_forward` | `cudaCheck(cudaDeviceSynchronize());` |
| `gpt2_backward_and_reduce` | `cudaCheck(cudaDeviceSynchronize());` |
| `gpt2_update` | `cudaCheck(cudaDeviceSynchronize());` |
| `gpt2_validate` | `cudaCheck(cudaDeviceSynchronize());` |

So when control returns to the host after `gpt2_backward_and_reduce`, the GPU is
genuinely idle. A marker toggle between these calls brackets computation.

⚠️ **If you port this to a codebase that does not sync (most real training
stacks do not), you must add the syncs yourself.** Section 3 shows the
measurement that proves it holds here, and that same measurement is how you
would detect the problem elsewhere: without a sync, the host-measured interval
collapses to a fraction of the CUDA-measured one.

---

## 3. Verification: do the host timestamps actually match the GPU?

The loop records, per micro-step, both a host-clock interval
(`marker_high` → `marker_low` from `CLOCK_MONOTONIC`) and a CUDA-event elapsed
time over the *same* region. If those disagree, host timestamps cannot be used
to align the scope.

Measured on testbed-node1, RTX 3090, GPT-2 124M, `-b 2 -t 1024 -d 4096 -a 1`,
mode 1 with `-Sg 50000 -Sj 20000`:

```
step.micro  host fwdbwd  cuda fwdbwd    delta  delta%  gap req  gap act
0.0            46.621ms     46.629ms  -0.008  -0.02%  62.41ms  62.47ms
0.1            49.067ms     49.072ms  -0.005  -0.01%  54.49ms  54.55ms
1.0            45.047ms     45.052ms  -0.005  -0.01%  68.71ms  68.77ms
1.1            46.524ms     46.529ms  -0.006  -0.01%  68.62ms  68.68ms
2.0            47.755ms     47.761ms  -0.006  -0.01%  67.48ms  67.54ms
2.1            44.944ms     44.950ms  -0.007  -0.01%  63.33ms  63.39ms

host fwdbwd  mean 46.660 ms   sd 1.585
cuda fwdbwd  mean 46.666 ms   sd 1.585
host-cuda    mean -0.006 ms   (-0.01%)
```

**The two clocks agree to 6 microseconds on a 47 ms region — 0.01%.** Host
timestamps in the timeline log are therefore sound for scope alignment, and the
sync assumption in section 2 is confirmed empirically, not just by reading the
source.

**Gap fidelity.** Requested gaps spanned 54.5–68.7 ms, all inside the
50–70 ms jitter band, and `nanosleep` overshot by a consistent **+0.061 ms**
(max +0.063 ms). Gaps are accurate and slightly long, never short.

**Budget reconciliation.** Marked regions plus gaps sum to 221.8 ms/step against
219.6 / 233.8 / 228.3 ms reported by llm.c's own `cudaEvent` timing (mean
227.3 ms). The **~5.5 ms/step residual is exactly the work deliberately left
outside the marked window**: `gpt2_update` (AdamW over all 124 M parameters),
`gpt2_calculate_grad_norm`, and `dataloader_next_batch`. The books balance, which
is the check that nothing measured is being double-counted or omitted.

---

## 4. Mapping code regions to signal features

What you should expect to see, and which source construct produces it.

| Signal feature | Source |
|---|---|
| marker rises | `sc_mark(..., 1)` immediately before `gpt2_forward` |
| first sustained compute plateau | `gpt2_forward` — layer loop `for (l = 0; l < L; l++)` |
| second, ~2× longer plateau | `gpt2_backward_and_reduce` — layer loop **`for (l = L-1; l >= 0; l--)`** |
| marker falls | `sc_mark(..., 0)` after backward returns (GPU already idle) |
| memory-bound spike *outside* the marker | `gpt2_update` — AdamW touches every parameter plus `m_memory`, `v_memory`, `master_weights` |
| flat idle *outside* the marker | `sidechannel_gap` (mode 1) and `dataloader_next_batch` |

The **layer-order reversal between the two plateaus** is the structural
signature worth targeting: forward walks layers 0→L-1, backward walks L-1→0.
That ordering is a property of the chain rule, not of scheduling, so it survives
amplitude scaling, gap insertion and power capping.

---

## 5. Recipes

**Calibrate the marker first** (measures the real DTR latency of *your*
adapter — do not assume it):

```bash
./train_gpt2cu -Sd /dev/ttyUSB0 -Sl /tmp/calib.csv -Sc 20 -x 1
```
Twenty 10 ms pulses at 50% duty. Measure their width and spacing on the scope;
the deviation from 10 ms is your marker latency and jitter.

**Baseline training, one marked pass per step:**
```bash
./train_gpt2cu -e gpt2_124M_bf16.bin -b 4 -t 1024 -x 20 -a 1 -v 100 -s 0 \
  -Sd /dev/ttyUSB0 -Sl /tmp/train.csv -Sw 0
```

**Half-length bursts — no code change, CLI only.** Halve the micro-batch and
keep total tokens, so `grad_accum_steps` doubles and each marked burst is
~half the work:
```bash
./train_gpt2cu -e gpt2_124M_bf16.bin -b 2 -t 1024 -d 4096 -x 20 -a 1 -v 100 -s 0 \
  -Sd /dev/ttyUSB0 -Sl /tmp/half.csv -Sw 0
```

**Modified-inference shaping** — same arithmetic, request-like cadence:
```bash
./train_gpt2cu -e gpt2_124M_bf16.bin -b 2 -t 1024 -d 4096 -x 20 -a 1 -v 100 -s 0 \
  -Sd /dev/ttyUSB0 -Sl /tmp/shaped.csv -Sw 1 -Sg 50000 -Sj 20000
```

Always pass **`-a 1`** for measurement runs: it reuses one batch, removing
dataloader I/O variance from the trace.

---

## 6. Limitations

1. **`-Sw 1` does not make this inference.** It changes only *when* work
   happens. Every kernel, every FLOP and every weight update is identical to
   mode 0. It tests whether a detector keys on cadence — nothing more.
2. **Marker latency is real.** DTR toggling is a USB control transfer, order
   0.1–1 ms with jitter. Under 1% of a 47 ms pass, but calibrate it (§5) rather
   than assume. In the run above the marker was disabled, so the numbers in §3
   contain *no* marker cost — expect host intervals to grow slightly with `-Sd`.
3. **`nanosleep` overshoots** by ~60 µs consistently. Gaps are never short.
4. **Single-GPU assumption.** With multi-GPU, `multi_gpu_async_reduce_gradient`
   overlaps communication with the backward pass, so the marked region then
   contains NCCL traffic too. The marker still brackets it correctly, but the
   signal is no longer pure compute.
5. **llm.c's own sampling is not a valid inference reference.** The generation
   block has no KV cache — the source says so — and re-runs the full forward over
   the whole growing prefix each token, so cost per token *rises*. Real
   KV-cache serving is flat and memory-bound. Use vLLM for the inference class,
   or you will train a classifier on the wrong signature.
6. **A current transformer cannot see DC.** On the 12 V rail an SCT-013 responds
   to dI/dt, so it is an edge detector: it shows phase *transitions* well and
   absolute power not at all. Note the consequence — adding gaps *increases*
   edge count and makes the trace more conspicuous, while an adversary who
   merely **ramps clocks slowly** becomes nearly invisible to dI/dt without
   changing the workload at all. A dI/dt-only channel is therefore not a sound
   basis for verification; pair it with a DC-capable Hall or shunt sensor.

---

## 7. Timeline CSV

`step,micro_step,phase,marker,t_unix_s,t_mono_ms,cuda_ms,note`

- `t_unix_s` — `CLOCK_REALTIME`, for correlating with the scope's capture timestamp
- `t_mono_ms` — `CLOCK_MONOTONIC`, for durations (immune to NTP steps)
- `cuda_ms` — GPU elapsed time on `fwdbwd_cuda_ms` rows; requested gap length on
  `gap_begin` rows; `-1` where not applicable
- `marker` — the asserted state *after* the event, so transitions are unambiguous

Phases: `gap_begin`, `gap_end`, `dataload_begin`, `marker_high`, `forward_end`,
`marker_low`, `fwdbwd_cuda_ms`, and `calib` for calibration pulses.
