Your project has expanded substantially beyond v1: **multi-node GPU training and inference, multi-host TPU workloads, decentralized training, distributed RL, kernel-level disguises, and physical-sensor experiments** are now represented. The main weakness for the follow-up paper is experimental depth: several important additions have only one short execution.

I created a [searchable workload catalog](/home/robi/GPU-monitoring/docs/workload_review_2026-10-12/inventory.html) and a [complete CSV](/home/robi/GPU-monitoring/docs/workload_review_2026-10-12/workloads.csv), including descriptions, samples, durations, hardware, observed device/host counts, and known failures.

**The comparison baseline is the June 17, 2026 first edition.** It reported 162 workloads, 1,404 runs, approximately 117 GPU-hours, and 445,310 telemetry readings on nine NVIDIA GPU models. Its experiments were single-node, with up to eight GPUs. It already included DDP, FSDP training through 70B, LoRA, substantial LLM inference, diffusion/speech inference, and 20 evasion families. Those concepts should not be presented as new additions. [Paper v1](https://arxiv.org/html/2606.19262v1)

The current inventory contains:

| Measure | Recorded corpus |
|---|---:|
| Workload configurations, including parameter/scope variants | 529 |
| Reconstructed jobs | 2,888 |
| Accelerator streams | 9,913 |
| Telemetry samples | 5,590,675 |
| Accelerator-hours | **1,538.3** |
| Summed elapsed job durations | 349.1 hours |
| Collections observing multiple hosts | 510 jobs; **1,014.9 accelerator-hours** |
| After excluding 21 explicitly failed jobs | 1,490.3 accelerator-hours |
| Multi-host subset after those exclusions | 493 jobs; 972.3 accelerator-hours |

An accelerator-hour is one recorded device for one hour: a ten-minute, 16-GPU collection contributes approximately 2.67 accelerator-hours. These are timestamp spans, **not useful training time or a certified usable dataset**. Remaining collections include unverified runs and startup periods. Median job duration is approximately 5.3 minutes; most newer distributed collections are around 10–12 minutes.

The definitions differ from v1, so subtracting 162 from 529 would not give a defensible count of newly added algorithms. The [provenance record](/home/robi/GPU-monitoring/docs/workload_review_2026-10-12/provenance.json) documents the accounting.

The workload groups below cover the recorded corpus without overlap. Hours include known failed collections.

| Workload group | Description and relationship to v1 | Configurations / jobs | Accelerator-hours |
|---|---|---:|---:|
| Conventional legacy workloads | CNN/MLP, GPT-2/BERT, single-node DDP/FSDP/LoRA; image/text/speech/diffusion inference; FFT, N-body, mining, rendering, video and idle. Primarily existing families with expanded coverage. | 137 / 1,290 | 124.4 |
| Legacy evasion and white-box extensions | Utilization/memory manipulation, interleaving, dilution, accumulation, throttling, oracle attacks and matched inference controls; additional model sizes and longer runs. | 105 / 660 | 110.0 |
| **Symmetric-GEMM and PPO** | Matched forward-only, discarded-gradient, fused-SGD and Adam controls; ordinary and disguised PPO. Adds experiments targeting the computation of weight updates directly. | 17 / 50 | 8.9 |
| **llm.c disguises** | Memory minimization, footprint spoofing, cadence/frequency shaping and concurrent MPS cover. | 14 / 66 | 2.6 |
| **Physical-sensor evasion probes** | Training diluted or reshaped with ballast, gap filling and related controls, paired with physical measurements. | 22 / 25 | 0.6 |
| **Distributed FSDP/DDP/LoRA/pipeline** | GPU/TPU scale ladders; transformer size, batch, checkpoint and accumulation variants; pipeline microbatch/rank-placement disguises. Extends existing training families to new topologies. | 34 / 146 | 298.3 |
| **MoE, tensor-parallel, contrastive and diffusion training** | Expert routing/all-to-all, TP8/TP16, gathered contrastive embeddings and distributed denoising. | 7 / 13 | 36.0 |
| **Local SGD, DiLoCo and SWARM** | Infrequent synchronization, compressed gradients, streamed/decoupled updates, quorum/straggler behavior and pipeline rerouting; includes shaped-link variants. | 15 / 17 | 57.0 |
| **Distributed RL** | IMPALA actor/learner separation, Ape-X replay and synchronous PPO. Different ranks can perform inference and training simultaneously. | 5 / 5 | 14.3 |
| **Replicated/sharded LLM inference** | Replicas versus TP/PP sharding, Hugging Face device-map execution, context/batch/KV-memory sweeps, and TPU equivalents. | 40 / 148 | 237.0 |
| **Online LLM serving** | vLLM Poisson, bursty and saturated traffic; prefill-heavy, decode-heavy and chunked-prefill variants. Addresses a stated v1 gap. | 7 / 30 | 82.0 |
| **Other distributed inference** | Whisper, SDXL, BGE/CLIP, ResNet, ViT and sharded DLRM-style recommendation. Mostly topology extensions of familiar modalities. | 8 / 32 | 77.2 |
| **Distributed evasion and filler controls** | Training hidden within saturating non-ML work, memory-arena/bubble manipulation and TPU type-confusion variants. | 44 / 173 | 169.2 |
| **Distributed non-ML/stall controls** | Communication-only collectives, distributed FFT, stencil/halo exchange, high-memory/low-power/duty-cycle controls and initialized trainers that never update. | 74 / 233 | 320.9 |

The catalog supplies the hardware-specific breakdown for every row. Broadly, the hardware coverage is:

| Hardware | Configuration evidence | Accelerator-hours |
|---|---|---:|
| A100 40GB | Single GPU, eight-GPU node, two nodes × eight GPUs | 466.5 |
| A100 80GB | Single-node configurations and two nodes × eight GPUs | 227.5 |
| H100 80GB | Single GPU, eight-GPU node, two nodes × eight GPUs | 443.0 |
| H100 NVL | Legacy single-node experiments; v1 documents up to two GPUs | 6.4 |
| H200 / H200 NVL | Legacy one/two/eight-GPU configurations; extensive later single-GPU instrumented experiments | 65.9 |
| B200 | Single-node experiments, including up to eight recorded devices | 21.8 |
| RTX 4050 / 5080 / 5090 | Consumer/laptop and single-node experiments | 3.5 / 18.3 / 7.8 |
| **RTX 3090** | Three-GPU physical-sensor testbed; individual traces generally monitor one device | 2.2 |
| **AMD RX 9070 XT** | Single GPU | 21.2 |
| **AMD MI300X VF** | Single virtual-function GPU, approximately 192GB HBM | 9.1 |
| **TPU v5e** | Eight chips on one host; two hosts × one chip; four hosts × four chips | 171.5 |
| **TPU v6e**, including `v6e-1` | One/four chips on one host; two/four hosts × one chip | 73.2 |
| Unidentified legacy hardware | Hardware identity absent from retained telemetry | 0.3 |

AMD, TPU and RTX 3090 coverage was absent from v1. Some AMD captures actually predate publication; “absent from v1” does not necessarily mean “collected afterward.” Trainium currently has a planning document, but no collected Trainium telemetry appears in this inventory.

Also, **multiple observed hosts do not establish cross-host computational dependence**. Replicated inference fleets and genuinely sharded jobs should remain separate experimental strata.

Your storage recollection is partly correct. The previous audit verified **13,675 LFS objects totaling 21.12GB**. Approximately 20.28GB is scope/raw-waveform data; accelerator telemetry itself accounts for roughly 451MB, including quarantine. The project supports reading LFS objects directly from external storage through `SPAR_LFS_OBJECTS`. [Loader implementation](/home/robi/GPU-monitoring/classifier/data_sources.py:1)

In this session:

- All **16,638 inventoried paths** match their saved LFS identities or locally available contents.
- The documented cache at `/tmp/gpu-workload-audit/lfs/objects` is absent, as is `.git/lfs/objects`.
- No GCS filesystem mount is visible.
- GCS listing failed because no active account is selected.

Consequently, these totals use the earlier hash-verified inventory, checked against the current checkout. I could not establish whether additional unimported collections exist in cloud storage.

There are several findings to address before publishing new accuracy claims:

1. **The newer families need independent repetitions.** Across local/decentralized training, MoE, TP and distributed RL, 24 of 25 configurations have only one reconstructed execution. Collectively these comprise only about **5.3 elapsed job-hours**, despite their much larger accelerator-hour total.

2. **Several workloads are mechanism probes rather than production training.** The distributed RL implementation explicitly uses synthetic token rewards and repeated-prefill generation without a KV cache; its purpose is traffic/computation shape rather than task learning. This is useful coverage, but cannot establish robustness to realistic RLHF. [Implementation and limitations](/home/robi/GPU-monitoring/workloads/train_rl_dist.py:1)

3. **The October 5 results cannot establish an improvement from adding data.** The repository correction says they were random family holdouts, not exhaustive LOFO; the ablation also changed test populations and selected models. Corrected evaluation code exists, but replacement results require rebuilt caches and rerunning. [Audit correction](/home/robi/GPU-monitoring/docs/workload_audit_2026-10-11.md:1)

4. **The saved inventory had become partly stale.** I refreshed seven stalled jobs—11.36 accelerator-hours—to the current non-training objective in the new tables. I also separately cataloged eight omitted legacy JSON captures: 474 samples and 0.026 accelerator-hours, all too short for a 30-second window.

5. **Ground-truth coverage has improved, but is not uniform.** October 12 recovery supplied temporally matching STEP evidence for all 104 targeted TPU disguise jobs. Historical vLLM collections still cannot supply activity events that were never logged. One committed raw waveform is truncated despite having the correct committed hash. [Recovery and quality details](/home/robi/GPU-monitoring/docs/workload_audit_2026-10-11.md)

For additional workloads, I would prioritize these:

| Priority | Add or strengthen | Why it matters |
|---|---|---|
| **1** | Production implementations of dense and MoE training, including combined DP/TP/PP and **context parallelism**, communication overlap, checkpointing and offload | Tests whether the detector recognizes training beyond shared custom workload code. Megatron documents these distinct parallelism mechanisms. [NVIDIA guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html) |
| **1** | Real PPO/GRPO-style post-training with cached rollouts, reward scoring and sparse learners | Strengthens the existing synthetic RL coverage and directly tests training hidden within an inference-heavy fleet. |
| **1** | Matched inference negatives: **disaggregated prefill/decode**, speculative decoding, multi-adapter serving and realistic mixed request lengths | These produce substantial communication and changing compute patterns without training. vLLM provides a concrete disaggregated-prefill implementation. [vLLM documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/) |
| **2** | Recommendation and GNN **training**, plus production diffusion/video or multimodal training | Adds learning workloads with substantially different memory access and communication. MLPerf provides datasets and quality targets for selecting reproducible cases. [MLPerf Training](https://mlcommons.org/benchmarks/training/) |
| **2** | Real distributed HPC, GPU ETL, vector search and mixed-tenant jobs | Your synthetic non-ML controls are broad; realistic applications would better measure false alarms and background interference. |
| **2** | Existing difficult workloads on **four/eight GPU hosts**, different fabrics and partial observation | Current measured GPU multi-host coverage tops out at two hosts. TPU four-host coverage does not substitute for larger GPU clusters. |

For adversarial robustness, I would strengthen the experimental design before adding dozens more named variants:

- **Freeze a detector, then attack that exact detector.** Search combinations of dilution, scheduling, memory shaping, communication suppression/padding and learner placement. Hold back final attacks from defender development. Adaptive evaluation is essential; success against a fixed catalog is a narrower claim. [Tramèr et al.](https://arxiv.org/abs/2002.08347)
- **Measure evasion at preserved learning utility.** Report useful training tokens, parameter updates, validation improvement, throughput, energy and cost against matched honest training. Training-time fraction alone is insufficient: a long synchronization stall can occupy a “training” interval while accomplishing little useful work.
- **Separate generalization tests.** Use job-disjoint splits, exhaustive mechanism-family holdouts, implementation/backbone holdouts, hardware/site holdouts and a final chronological campaign holdout. Keep attack/control siblings together.
- **Optimize and report operational metrics.** Training recall at fixed false-alert rates, false alerts per job-hour, time to detection and worst-family performance are more informative than pooled accuracy. Evaluate the actual alert aggregation rule on long benign runs.
- **Test realistic sensor degradation.** Missing ranks, clock offsets, dropped samples, absent NIC channels and changing topology should be explicit tests. Compare GPU-only, GPU+NIC and richer instrumentation on the same eligible windows.
- **Preserve a durable experiment manifest.** Record immutable job/rank identities, topology, software/container versions, completion status, activity events and object hashes. Keep raw objects on durable external storage and bind every result to its exact inputs.

My first collection campaign would repeat the hardest existing cells across independent launches and days, add production training/inference matched pairs, and rerun the corrected paired evaluation. That would provide stronger evidence for the follow-up paper than increasing the workload-name count alone.
