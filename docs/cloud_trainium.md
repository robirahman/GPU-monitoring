# Next-Steps Doc: AWS Trainium2 Corpus

## Why Trainium

Trainium is the third major training accelerator architecture after
NVIDIA GPUs and Google TPUs. AWS has invested heavily in Trainium2 for
2025–2026 (used in Project Rainier for Anthropic). If our methodology
generalizes from GPU → TPU but **not** to Trainium, that's a meaningful
finding about whether the detection signal is fundamentally physical or
specific to particular accelerator families.

Architecturally Trainium2:
- 96 GiB HBM3 per chip, 16 NeuronCore-v3 per chip
- Like TPU, uses systolic-array-style matmul engines, not warps
- Compute model is dataflow (Neuron compiler statically schedules
  operations)
- Telemetry via the Neuron SDK's `neuron-monitor` CLI + JSON output

## What we don't know yet

I have not tested any of this on actual Trainium hardware. The procedure
below is what the docs and forum posts suggest should work. Phase 2 is
specifically about probing this before committing to the full corpus
collection. Be ready to iterate.

## Provisioning

Two paths:

### A. EC2 trn1.2xlarge (2 Trainium-1 chips, cheapest entry point)

- $1.34/h on-demand (~$1.34 × 2h Phase-2 probe = **$3**)
- Trainium-1 (older), not Trainium2 — but tests the Neuron SDK end-to-end
- Available without quota request
- Use Deep Learning AMI (Neuron) - Ubuntu 22.04

### B. EC2 trn2.48xlarge (16 Trainium2 chips, full Phase-3 target)

- $30+/h on-demand (Project Rainier pricing)
- Trainium2 (current generation, matches MI300X / H200 capacity tier)
- **Requires quota request** from AWS — file a service-quota increase
  for "Running On-Demand Trn2 instances" weeks in advance. Without
  prior request, request will likely be denied immediately.
- Use Deep Learning AMI Neuron (Ubuntu 22.04)

### Recommendation

**Phase 2 on trn1.2xlarge first** to exercise the entire stack cheaply.
If that works, file the trn2 quota request and use it for Phase 3.

### Launch (CLI)

```bash
# Phase 2 (trn1.2xlarge, cheap)
aws ec2 run-instances \
    --image-id ami-XXX  \  # Deep Learning AMI Neuron for ubuntu22.04
    --instance-type trn1.2xlarge \
    --key-name my-keypair \
    --security-group-ids sg-XXX \
    --subnet-id subnet-XXX \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=spar-trainium-probe}]"

# Get the AMI for your region:
# aws ec2 describe-images --owners amazon \
#     --filters "Name=name,Values=Deep Learning AMI Neuron PyTorch*" \
#               "Name=architecture,Values=x86_64" \
#     --query 'Images[*].[ImageId,Name,CreationDate]' --output table | head
```

## Phase 2 — telemetry probe (~2 h, $3 on trn1)

```bash
# After SSH (default user is `ubuntu`):
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring
git checkout amd

# Neuron SDK is preinstalled on the AMI. Verify:
neuron-ls
# Should print device count + Neuron driver version

# Telemetry surface — there are 3 sources:

# 1. neuron-monitor (JSON-streaming CLI) — primary
neuron-monitor /opt/aws/neuron/share/monitor.conf | head -200
# Streams JSON every N seconds. Default config polls 1 Hz.

# 2. neuron-top (interactive, like nvidia-smi)
neuron-top
# Useful for ad-hoc inspection, not for programmatic collection.

# 3. /sys/class/neuron/* sysfs entries — kernel-level
ls /sys/class/neuron/neuron0/
# May expose: vcpu_util, mem_used, fan_speed, temp, sram, dma
# Verify what's actually there

# Python:
python - <<'PY'
import subprocess, json, time
# neuron-monitor streams; capture 5 seconds of output
proc = subprocess.Popen(
    ["neuron-monitor", "-c", "/opt/aws/neuron/share/monitor.conf"],
    stdout=subprocess.PIPE, text=True,
)
samples = []
start = time.time()
while time.time() - start < 5:
    line = proc.stdout.readline()
    if line.strip().startswith("{"):
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            pass
proc.terminate()
print(f"Captured {len(samples)} samples in 5s")
if samples:
    s = samples[0]
    # Top-level keys in neuron-monitor JSON, per AWS docs:
    print("Top-level keys:", list(s.keys()))
    # 'neuron_runtime_data', 'instance_info', 'hardware_info', ...
    if 'neuron_runtime_data' in s:
        print("Runtime data keys:", list(s['neuron_runtime_data'][0].keys()))
PY
```

Document everything found in `docs/telemetry_signal_matrix.md`'s
Trainium column. Specifically determine:

- **NeuronCore utilization (%)** — analog to gfx_activity / SM
  utilization. Likely yes, per-core.
- **Device memory used (MB)** — HBM used per chip. Likely yes.
- **Power draw (W)** — uncertain. AWS exposes some power metrics in
  CloudWatch but not always per-second.
- **Temperature** — likely yes (Neuron has thermal sensors).
- **Clock frequency** — TBD. Trainium clocks are largely fixed by the
  compiler-scheduled dataflow; may not have a meaningful "clock" signal
  to read.
- **PCIe** — bandwidth in/out of the chip is reported by neuron-monitor
  per-host-link. Format probably differs from NVML's TX/RX per-chip.

## Phase 3 — full Trainium corpus (~6 h, $180 on trn2.48xlarge)

Caveats:

1. **Workload porting takes longer than for AMD/TPU.** Trainium requires
   compiling models with `neuronx-cc`, not just running PyTorch. This is
   an upfront cost — figure on 1 day to get the first training workload
   running, less for subsequent ones.

2. **Model selection.** Neuron SDK supports a curated list:
   - Llama, Mistral, Phi families for inference
   - ResNet, BERT, GPT-2 for training
   - Stable Diffusion through optimum-neuron
   - Custom models require custom compilation

3. **Workload script structure.** Replace:
   ```python
   device = torch.device("cuda")
   model = model.to(device)
   ```
   with:
   ```python
   import torch_neuronx
   model = torch_neuronx.trace(model, example_inputs)  # one-time compile
   # Then model() runs on Trainium
   ```

### Plan: minimal Trainium workload set

```python
# workloads/registry_trainium.py (new file)
WORKLOAD_REGISTRY_TRAINIUM = {
    "idle": ...,
    "trainium_resnet_cifar10": ...,        # torch.neuronx trained model
    "trainium_bert_sst2": ...,
    "trainium_llm_infer_llama3_8b": ...,    # via optimum-neuron
    "trainium_llm_infer_qwen25_3b": ...,    # untested
    "trainium_hpc_matmul": ...,             # raw matrix mult benchmark
}
```

Skip adversarial workloads entirely on Trainium for the first cut — the
GPU-specific evasion mechanics don't have direct Trainium analogs.

### Telemetry collector

`scripts/collect_trainium_telemetry.py` (new). Spawns `neuron-monitor`
as a subprocess, reads JSON lines from its stdout, extracts the fields
identified in Phase 2 into the 26-column parquet schema. Missing fields
get `-1` sentinel.

### Launch

```bash
nohup ./scripts/run_trainium_corpus.sh --reps 3 \
    > data/collection_logs/trainium_corpus.out 2>&1 &
```

## Risks / unknowns

- **AWS quota for trn2.** Likely the biggest risk. File the request
  early. If denied, fall back to trn1.2xlarge for both phases — paper
  story degrades but doesn't die (Trainium-1 is still a non-GPU
  architecture).
- **Neuron SDK fragility.** Workload compilation can fail on
  unsupported ops. Stick to AWS's curated model list for the first
  corpus; expand only after the headline result is in.
- **`neuron-monitor` polling rate.** Default config is 1 s; some
  versions cap at 5 s. Verify in Phase 2.
- **No power/clock telemetry.** If Trainium exposes neither power nor
  clock at 1 Hz, half the paper's 9 signals are missing. The Stage-2
  shape-only head is the only thing that can possibly work — but that's
  also the interesting experimental question.
- **Cost discipline.** trn2.48xlarge is $30+/h. Don't leave it idle.
  Use `aws ec2 stop-instances --instance-ids i-XXX` when stepping away.

## Cost summary

| Phase | Instance | $/hr | Duration | Total |
|---|---|---|---|---|
| Phase 2 (probe) | trn1.2xlarge | $1.34 | 2h | **$3** |
| Phase 3 (corpus) | trn2.48xlarge | $30 | 6h | **$180** |
| ...if trn2 quota denied | trn1.32xlarge | $21 | 6h | **$130** |

**Total recommended Trainium spend: ~$185 happy path, ~$135 if trn2
unavailable.**

## After collection

Same pattern as TPU — extend `classifier/eval_cross_vendor.py` into
`eval_cross_architecture.py` to include Trainium as a held-out
vendor for zero-shot eval, and extend the heatmap to include a
Trainium row/column.
