# llm.c side-channel evader

The evader used in the workload-shaping evasion study
(`docs/shaping_evasion_results.md`). It is Karpathy's **llm.c** GPT-2 trainer
plus a side-channel/evasion patch: a hardware marker line, idle-gap workload
shaping, a memory-footprint spoof, and a phase-boundary timeline log — all of
which change only *timing* and *resident memory*, never the arithmetic.

Because `train_gpt2.cu` is upstream (MIT) code, this directory carries the
**delta** as a patch rather than a copy:

- `sidechannel.h` — the instrumentation/shaping/timeline header (self-contained; drops into `llmc/`)
- `train_gpt2.cu.sidechannel.patch` — the changes to `train_gpt2.cu` (six `-S*` CLI flags, two CUDA events, gap insertion, `-Sm` pad)
- `SIDECHANNEL.md` — design, verification, and the flag reference

## Build

```bash
git clone https://github.com/karpathy/llm.c.git && cd llm.c
git checkout f1e2ace                         # base commit the patch is against
cp /path/to/GPU-monitoring/workloads/llmc_sidechannel/sidechannel.h llmc/
git apply /path/to/GPU-monitoring/workloads/llmc_sidechannel/train_gpt2.cu.sidechannel.patch
bash dev/download_starter_pack.sh            # gpt2_124M_bf16.bin + tinyshakespeare
PATH=/usr/local/cuda/bin:$PATH make train_gpt2cu   # auto-detects sm_90 on H200
```

## Flags (see SIDECHANNEL.md for detail)

| flag | meaning |
|---|---|
| `-Sw <0/1>` | workload shaping: 0 = training, 1 = inference-cadence (idle gaps) |
| `-Sg <us>` / `-Sj <us>` | base idle gap and uniform jitter per micro-step (mode 1) |
| `-Sm <MiB>` | hold a fixed extra GPU allocation (memory-footprint spoof) |
| `-Sd <dev>` / `-Sl <path>` / `-Sc <n>` | DTR marker device / timeline CSV / calibration pulses |

## How the study drove it

`scripts/run_shaping_sweep.py` runs the binary through `run_workload_with_scope.py`
(NVML + Rogowski scope + DCGM), e.g. the memory-minimized evader is
`-b 1 -r 2` (micro-batch 1 + gradient accumulation via `-d`, same training at
~1/16 the activation memory) and the frequent-update variant is `-b 1 -r 2 -d 1024`.
Note: `-b 1` is ~1 ms/step on an H200, so a ~150 s run needs `-x` ≈ 16000.
