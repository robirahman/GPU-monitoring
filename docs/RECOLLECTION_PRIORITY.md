# Re-collection priority

Ranked by tier-A windows recovered, not by job count. A cell whose healthy runs are too short to fill a single 30 s window returns nothing however often it is re-collected, and several outstanding cells are exactly that -- `nbody_sim` and `resnet50_inference` on B200 both yield a median of **0 windows** per healthy run.

AMD is excluded: not available on Vast.ai and not part of the NeurIPS corpus.

| # | rent | cells | runs | est. windows | per run | empty combos |
| --: | :-- | --: | --: | --: | --: | --: |
| 1 | 1x A100-SXM4-80GB (+8x node) | 3 | 9 | **672** | 74.7 | False |
| 2 | 1x H200 NVL | 4 | 12 | **249** | 20.8 | False |
| 3 | 1x B200 | 4 | 12 | **150** | 12.5 | False |
| 4 | 1x H100 80GB HBM3 | 5 | 15 | **75** | 5.0 | False |
| 5 | 1x GeForce RTX 5080 | 2 | 6 | **74** | 12.3 | False |
| 6 | 1x H200 (+8x node) | 3 | 9 | **36** | 4.0 | False |
| 7 | 1x H100 NVL | 1 | 3 | **12** | 4.0 | False |

**66 runs, ~1268 tier-A windows** across 7 rentals.

## A100-SXM4-80GB

Running on GCP (`a2-ultragpu-8g`). `fsdp_70b_nvlink` has no healthy A100 run anywhere in the corpus and is expected to OOM -- AdamW fp32 state for 70b is ~560 GB against 8x80 GB -- so treat a failure as a hardware limit to state, not a collection error.

Quickstart: [`docs/quickstart_a100_80gb.md`](quickstart_a100_80gb.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `fsdp_30b_nvlink` | 8 | 204 | healthy runs on this device |
| `fsdp_70b_nvlink` | 8 | 0 | healthy runs on this device |
| `llm_infer_mistral_7b_q4` | 1 | 20 | healthy runs on this device |

## H200 NVL

Quickstart: [`docs/quickstart_h200_nvl.md`](quickstart_h200_nvl.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `llm_infer_mistral_7b_q4` | 1 | 19 | healthy runs on this device |
| `llm_infer_qwen25_14b_fp16` | 1 | 21 | healthy runs on this device |
| `llm_infer_qwen25_32b_fp16` | 1 | 22 | healthy runs on this device |
| `llm_infer_qwen25_7b_fp16` | 1 | 21 | healthy runs on this device |

## B200

`adversarial_G_clock_throttled` calls `nvidia-smi -pl` and `--lock-gpu-clocks`; it needs a **privileged** container or it will skip the throttle and collect an undisguised run.

Quickstart: [`docs/quickstart_b200.md`](quickstart_b200.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `adversarial_G_clock_throttled` | 1 | 19 | healthy runs on other devices |
| `cufft_benchmark` | 1 | 31 | healthy runs on this device |
| `nbody_sim` | 1 | 0 | healthy runs on this device |
| `resnet50_inference` | 1 | 0 | healthy runs on this device |

## H100 80GB HBM3

Quickstart: [`docs/quickstart_h100_80gb.md`](quickstart_h100_80gb.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `bert_sst2` | 1 | 1 | healthy runs on this device |
| `ffmpeg_nvenc` | 1 | 19 | healthy runs on other devices |
| `gpt2_wikitext2` | 1 | 3 | healthy runs on this device |
| `gpt2_wikitext2_amp` | 1 | 1 | healthy runs on this device |
| `pytorch_resnet_cifar10` | 1 | 1 | healthy runs on this device |

## GeForce RTX 5080

Quickstart: [`docs/quickstart_rtx5080.md`](quickstart_rtx5080.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `llm_infer_phi3_mini` | 1 | 4 | healthy runs on this device |
| `llm_infer_phi3_mini_seq2048` | 1 | 20 | healthy runs on other devices |

## H200

Two of the three cells are single-GPU and yield ~2 windows a run. The third, `fsdp_70b_nvlink`, needs an **8x H200** node for ~8 windows a run; renting eight GPUs for that is poor value unless you want the 70b point on H200 specifically.

Quickstart: [`docs/quickstart_h200.md`](quickstart_h200.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `fsdp_70b_nvlink` | 8 | 8 | healthy runs on this device |
| `pytorch_mlp_cifar10` | 1 | 2 | healthy runs on this device |
| `pytorch_resnet_cifar10` | 1 | 2 | healthy runs on this device |

## H100 NVL

Quickstart: [`docs/quickstart_h100_nvl.md`](quickstart_h100_nvl.md)

| cell | GPUs | est. windows per run | estimated from |
| :-- | --: | --: | :-- |
| `pytorch_resnet_cifar10` | 1 | 4 | healthy runs on this device |

