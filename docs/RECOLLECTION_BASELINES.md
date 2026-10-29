# Baseline gaps worth filling

69 non-adversarial tier-A jobs are aborted, truncated or starved, carrying 97 windows between them. A broken job only matters if it leaves a (device, cell) combination with nothing usable, and **32 of 44 combinations still have healthy runs** -- those need nothing.

Re-collection cannot repair tier A, which is frozen by collection date; anything collected now lands in `C_followup`. What it fixes is the **coverage claim**, so it is worth doing only where a combination is genuinely empty.

## The 12 empty combinations

| device | cell | lost jobs | GPUs needed | reproducible |
| :-- | :-- | --: | --: | :-- |
| GeForce RTX 4050 Laptop GPU | `deepseek_r1_inference` | 3 | 1 | yes |
| GeForce RTX 4050 Laptop GPU | `pytorch_mlp_cifar10` | 1 | 1 | yes |
| GeForce RTX 4050 Laptop GPU | `pytorch_resnet_cifar10` | 1 | 1 | yes |
| GeForce RTX 4050 Laptop GPU | `pytorch_resnet_cifar10_amp` | 1 | 1 | yes |
| GeForce RTX 4050 Laptop GPU | `resnet50_inference` | 1 | 1 | yes |
| GeForce RTX 4050 Laptop GPU | `whisper_inference` | 1 | 1 | yes |
| GeForce RTX 5080 | `llm_infer_phi3_mini_seq2048` | 1 | 1 | yes |
| H100 80GB HBM3 | `ffmpeg_nvenc` | 1 | 1 | yes |
| AMD Instinct MI300X VF | `ffmpeg_vaapi` | 1 | n/a | **no registry entry** |
| B200 | `nvlink_latency` | 1 | n/a | **no registry entry** |
| unknown | `baseline_infer` | 1 | n/a | **no registry entry** |
| unknown | `baseline_train` | 1 | n/a | **no registry entry** |

**8 are reproducible**, on 3 devices, all single-GPU. 10 jobs x 3 reps = 30 runs.

### GeForce RTX 4050 Laptop GPU

```bash
CELLS="deepseek_r1_inference pytorch_mlp_cifar10 pytorch_resnet_cifar10 pytorch_resnet_cifar10_amp resnet50_inference whisper_inference"
for cell in $CELLS; do for rep in 1 2 3; do
  log="logs/${STAMP}__$(hostname -s)__${cell}_rep${rep}.log"
  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
  python scripts/run_workload.py --workload "$cell" \
    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=$?" >> "$log"
done; done
```

### GeForce RTX 5080

```bash
CELLS="llm_infer_phi3_mini_seq2048"
for cell in $CELLS; do for rep in 1 2 3; do
  log="logs/${STAMP}__$(hostname -s)__${cell}_rep${rep}.log"
  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
  python scripts/run_workload.py --workload "$cell" \
    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=$?" >> "$log"
done; done
```

### H100 80GB HBM3

```bash
CELLS="ffmpeg_nvenc"
for cell in $CELLS; do for rep in 1 2 3; do
  log="logs/${STAMP}__$(hostname -s)__${cell}_rep${rep}.log"
  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
  python scripts/run_workload.py --workload "$cell" \
    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=$?" >> "$log"
done; done
```

## Not reproducible

`baseline_infer`, `baseline_train`, `ffmpeg_vaapi`, `nvlink_latency` have no registry entry, so there is no command to re-run them. They are gaps to state, not work to schedule -- inventing a command would fabricate the workload definition.

## Why most of the 69 need nothing

The broken jobs mostly sit alongside healthy ones: `llm_infer_mistral_7b_q4` on H200 NVL lost six jobs but keeps four (58 windows), and every AMD RX 9070 XT inference cell has three to fifteen healthy siblings. 61 of the 69 carry zero windows at all -- they are too short to produce a single 30 s window, so they never entered the evaluation.

