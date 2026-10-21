# H200 follow-up collection campaign

Tooling for the four new H200 NVL recording types described in
[Tajdari-S/H200-scope-analysis `mlsys_followup/COLLECTION_PLAN.md`](https://github.com/Tajdari-S/H200-scope-analysis/blob/main/mlsys_followup/COLLECTION_PLAN.md),
run on amodo-gigabyte-6. Earlier H200 data and frozen results are untouched; every run here gets a
new run ID.

## Layout

| Module | Role |
|---|---|
| `scope_logger.py` | PicoScope 5464E+ streaming at 19.53 MS/s, FIR decimation to 976.6 kS/s, per-poll accounting, envelope + raw segments |
| `decimate.py` | Streaming two-stage linear-phase FIR (×4, ×5); output sample *k* is centred on raw sample 20*k* |
| `sensors.py` | NVML (the nine detector counters + extras), BMC (DCMI and SDR `SYS_POWER`), DCGM `dmon` |
| `events.py` | JSONL phase events with wall/monotonic ns, source location and chained CUDA-event GPU times |
| `markers.py` | Common physical markers (0.3 / 0.6 / 1.2 s bf16 matmul bursts ×2) at both ends of each acquisition |
| `train.py` | Real-data training: MLP / ResNet-18 (CIFAR-10), BERT (SST-2), LoRA (WikiText-2); modes `ordinary`, `diluted`, `interleave`, `shape` |
| `serve.py` | Phi-3-mini (Alpaca prompts) and Whisper large-v3-turbo (LibriSpeech) with Poisson request-rate schedules |
| `sequence.py` | Type 4: serving throughout, training bursts at planned offsets on the 10 s window grid, observation afterwards |
| `session.py` | Runs a JSON plan; one validated acquisition per run; manifest + ledger; nothing deleted |
| `validate.py`, `align.py` | Sensor health, rejected intervals with reasons, marker-based clock alignment |
| `make_pilot.py`, `pilot_report.py` | Validation-pilot plan generator and summary |

```
python -m h200_campaign.make_pilot --session-id pilot_s1 --seed 1 > plan.json
python -m h200_campaign.session plan.json            # -> ~/campaign/sessions/pilot_s1/...
python -m h200_campaign.pilot_report ~/campaign/sessions/pilot_s1
```

## Sensor settings (measured on this rig)

- **Scope:** PicoScope 5464E+MSO, s/n 12669/0001, cal 07Jul26, psospa driver 1.1.9.6000, FW 1.11.1.0 / 2.4.5.0,
  USB 3.0. Probe on channel A (idle noise 175 counts vs 65 on unconnected B–D). 16-bit, DC coupling,
  20 MHz hardware bandwidth limit (the narrowest available), requested 50 ns → actual 51.2 ns (19.53 MS/s).
- **Range ±10 V.** At ±5 V (the earlier campaign's setting) every full-load GPU step clipped: about 3,000
  samples (~150 µs) at the ADC rail per edge, driver over-range flag set, at idle-to-load and load-to-idle
  steps only. At ±10 V the same steps peak at 6.0–6.2 V, no clipping. Pilot runs peak at ≤0.63 of full scale.
  The range is to be frozen from pilot data before final tests.
- **Anti-aliasing:** the 20 MHz limiter cannot protect a ~1 MS/s stream, so the scope streams at 19.53 MS/s
  and `decimate.py` filters (Kaiser, 80 dB design) before ×20 decimation. Alias-protected band 0–400 kHz;
  400–488 kHz is not protected. Unit tests check exact streaming equivalence and ≥60 dB rejection of a tone
  that would fold onto 100 kHz.
- **Probe:** `probe_v_per_a_claimed = 0.100` is the configuration recorded by the earlier campaign, not a
  calibration. Amps are not computed from it in stored data; volts and counts are stored.
- **NVML** 1 s, **DCGM** 100 ms (profiling fields 1001–1010 + power/temp/util), **BMC DCMI** 0.5 s polling
  (the BMC reports a 5 s sampling period) and **SDR `SYS_POWER`** polled back to back in its own thread
  (each query takes ~1.6 s).

## Accounting and health

Every scope poll that delivers data is stored with host wall/monotonic ns, driver buffer index, start index,
sample count, status and over-range flag, and checked for contiguity. Over-range (driver flag, per poll) and
rail-clipped samples (per ~1 ms envelope bin) are separate from sample loss. `validate.py` writes
`health_intervals.csv`: every interval the monitor could not assess (rail, over-range, contiguity break,
sensor gap > 2× its interval, missing sensor, late start or early stop), with its reason. These stay in the
denominator.

## Pilot session 1 (2026-10-21) — results

10 of 10 runs completed (Phi-3 and Whisper serving, BERT / MLP / ResNet ordinary training, ResNet ordinary vs
diluted ×5, ResNet without instrumentation, cuFFT, path-trace render, idle).

| Check | Result |
|---|---|
| Scope delivery | 0 contiguity breaks, all driver polls status OK in every run |
| Saturation at ±10 V | 0 over-range polls, 0 rail samples; peak 0.61–0.63 of full scale |
| Unassessable time | none, for any sensor, in any run |
| Scope–GPU alignment | median offset −0.21 to +0.43 ms per run; IQR 1.6–2.1 ms (1 ms envelope bins); start vs end markers agree within ~1 ms |
| Scope clock vs host | −2.2 to +0.5 ppm per run |
| DCGM / NVML / BMC lag behind GPU edges | ~115–270 ms / ~0.5 s for power (~150 ms for utilization) / ~2 s |
| Instrumentation overhead | ResNet 44,730 vs 44,800 samples/s with and without all sensors + per-update GPU marks (0.16%, one pair) |
| Monitor cost | scope logger ~19% of one core and ~7.5 GB/h with full raw; BMC ~5%; NVML and DCGM <1% each |
| Useful learning | BERT 90.9% SST-2; ResNet-18 84–86% CIFAR-10 in 300 s; MLP 55%; Whisper WER 4.7%; Phi-3 20/20 requests |
| Training start vs first completed update (GPU) | ResNet 6.1 s (cuDNN autotuning), BERT 1.0 s, MLP 0.04 s |
| Diluted ×5 vs ordinary (ResNet) | 38.5% of useful throughput; 85% accuracy at 90 s vs 60 s (1.5×) |

## Not verified here (needs someone at the rig)

- Identity of the installed probe (RCP60XS variant), its sensitivity and transfer response. The plan requires a
  measured response; that needs a known injected current (for example the 5464E's AWG driving a loop through
  the coil).
- Exactly which conductors the probe encloses, any uncovered power path, and a wiring diagram.

## Known limitations

- Two ordinary ResNet runs with the same seed finished at 84.1% and 86.3% (cuDNN autotuning is
  non-deterministic and the learning rate is constant). Quality comparisons need repetitions, and a learning-rate
  schedule should be fixed before the attack campaign.
- Phi-3 is served one request at a time with HuggingFace `generate()` (median ~0.9–2.8 s per request), so request
  rates above ~0.3/s produce a sustained backlog.
- DCGM `dmon` prints no sample timestamps; its rows carry host receipt time only.
