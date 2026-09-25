# SPAR Hour-1 on RunPod Instant Cluster — A/B/C/D interconnect

Cross-node interconnect smoke for SPAR workloads **A–D** on RunPod Instant Cluster (EUR-IS-3, 2×1×H100). Successful runs only: **A/B** session 1, **C/D** (real collectives) session 2.

Live runbook: [`personal_note/runpod_hour1_smoke_e2e.md`](../personal_note/runpod_hour1_smoke_e2e.md)  
Launcher (C/D): [`scripts/run_hour1_crossnode_serve.sh`](../scripts/run_hour1_crossnode_serve.sh)

---

## Interconnect truth (HARD GATE)

| Check | Result |
|---|---|
| `/sys/class/infiniband/mlx5_*` | Present (8 HCAs/node) |
| `ports/*/counters/` + `hw_counters/` | **MISSING** (ENOENT — devices/attrs present, counter dirs absent) |
| NCCL RoCE (`NET/IB`) | **WC vendor err 129** → datapath broken |
| Working multi-node path | **`NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=ens1`** |
| Byte throughput proxy | `/sys/class/net/ens*/statistics/{rx,tx}_bytes` |
| Fabric | Instant Cluster ens1–ens8 + mlx5 RoCE — **not** native IB |

**Can:** True multi-node dense DP AllReduce (A), DiLoCo outer AllReduce (B), **TP AllReduce** (D), **EP AllToAll** (C) over ens1 Socket, with GB-scale ens deltas under load + SPAR sidecar.

**Cannot:** Native IB / healthy RoCE verbs; sysfs IB `ports/*/counters`.

---

## Clusters

| | Session 1 (A/B) | Session 2 (C/D) |
|---|---|---|
| Cluster | `hq4r9nu5fmjtcg` / `spar-ib-2x1h100` | `5qz3vze54u7zb2` / `spar-cd-real-2x1h100` |
| Shape | 2×1×H100 80GB HBM3 · EUR-IS-3 (wanted 2×2; stock forced 2×1) | same |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | same |
| ens1 | n0 `10.65.0.2` / n1 `10.65.0.3` | same |
| SSH n0 | `root@213.181.105.225:16203` | `root@213.181.105.230:12760` |
| SSH n1 | `root@213.181.105.210:17652` | `root@213.181.122.200:15378` |
| Stack | torch 2.8.0+cu128 | + vLLM **0.10.2**, transformers **4.55.4**, ray **2.48.0** |
| Billable | ~19:00–20:32Z (~92 min) ≈ **~$12.8** | ~22:57–23:50Z (~50 min) ≈ **~$7** |
| Torn down | `delete-cluster` → 204 | `delete-cluster 5qz3vze54u7zb2` |
| Artifacts | `data/hour1_runpod/{node0,node1}/` (+ `hour1_n{0,1}.tgz`) | `data/hour1_runpod/cd_real/` |

```bash
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519_vcf)
```

---

## A — W01 dense DP (2-node) ✅

Session 1 · ~19:16–19:18Z · label `hour1_w01_ddp`  
NCCL AllReduce · Socket/`ens1` · eth ≈ **91.7 GB tx / 91.5 GB rx** (n0)

### Pre: FakeData patch

CIFAR download raced rank1; patched both nodes to `FakeData` before the successful run.

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("workloads/pytorch_training_ddp.py")
t = p.read_text()
old = """    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=(rank == 0), transform=transform
    )
    # Wait for rank 0 to download
    dist.barrier()
"""
new = """    # Hour-1 / interconnect: FakeData — no CIFAR download
    train_dataset = torchvision.datasets.FakeData(
        size=10000, image_size=(3, 32, 32), num_classes=10, transform=transform
    )
    dist.barrier()
"""
p.write_text(t.replace(old, new, 1))
print("patched_ok")
PY
```

### Commands (MASTER_PORT=29501)

```bash
# NODE0
cd /workspace/GPU-monitoring
export NNODES=2 NPROC_PER_NODE=1 MASTER_ADDR=10.65.0.2 MASTER_PORT=29501 NODE_RANK=0
export PYTHONUNBUFFERED=1 NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=ens1 NCCL_IB_HCA=mlx5
export PATH=/usr/local/bin:$PATH
/usr/local/bin/python scripts/run_with_telemetry.py \
  --label hour1_w01_ddp --output-dir data/hour1 \
  --nnodes 2 --node-rank 0 -- \
  /usr/local/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    workloads/pytorch_training_ddp.py --model resnet18 --duration 90 \
  > runs/hour1_w01_ddp_node0.log 2>&1
echo A0_EXIT:$? | tee -a runs/hour1_w01_ddp_node0.log

# NODE1 (parallel): NODE_RANK=1 / --node-rank=1 → hour1_w01_ddp_node1.log / A1_EXIT
```

### Key output (n0)

```
Bootstrap: Using ens1:10.65.0.2<0>
NCCL_IB_DISABLE set by environment to 0.
Failed to open libibverbs.so[.1]          # → Socket fallback
NET/Socket : Using [0]ens1:10.65.0.2<0>
Using network Socket
nRanks 2 nNodes 2 localRanks 1
Channel … via NET/Socket/0
DDP training complete. 52 epochs, 90.1s wall time
Fabric totals ib_bytes=0 eth_tx+rx=183149325874 samples=100
A0_EXIT:0
```

n1: `A1_EXIT:0` · `eth_tx+rx=183142534791`

Fabric summary n0 (`hour1_w01_ddp_*_fabric_summary.json`):

| field | value |
|---|---|
| `eth_total_tx_bytes` | 91 691 912 461 |
| `eth_total_rx_bytes` | 91 457 413 413 |
| `ib_total_bytes` | 0 |

---

## B — DiLoCo smoke (2-node) ✅

Session 1 · ~19:25Z · label `hour1_diloco`  
Outer AllReduce (`outer_sync`) · eth ≈ **1.13 GB** tx/rx (n0)

### Commands

```bash
# NODE0
cd /workspace/GPU-monitoring
export MASTER_ADDR=10.65.0.2 MASTER_PORT=29510 NODE_RANK=0
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=ens1 NCCL_IB_DISABLE=1
export PATH=/usr/local/bin:$PATH PYTHONUNBUFFERED=1
/usr/local/bin/python scripts/run_with_telemetry.py \
  --label hour1_diloco --output-dir data/hour1 --nnodes 2 --node-rank 0 -- \
  /usr/local/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  workloads/diloco_production.py --model-size 150m --local-steps 20 --max-steps 60 \
  --batch-size 1 --seq-len 512 --dataset synthetic --reduce-dtype fp16 --precision bf16 \
  --step-log runs/hour1_diloco_steps.jsonl --comm-log runs/hour1_diloco_comm.jsonl \
  > runs/hour1_diloco_n0.log 2>&1; echo B0:$? | tee -a runs/hour1_diloco_n0.log

# NODE1: NODE_RANK=1; step/comm logs *_n1.jsonl
```

### Key output (n0)

```
DiLoCo production | world=2 H=20 … model=150m
  params=140.5M  device=cuda:0
step=20 phase=outer_sync loss=10.2544 tok/s=3250 H_rem=20
step=40 phase=outer_sync loss=10.3136 tok/s=5179 H_rem=20
step=60 phase=outer_sync loss=10.3170 tok/s=6196 H_rem=20
done in 5.2s
Fabric totals ib_bytes=0 eth_tx+rx=2250826053 samples=15
B0:0
```

`hour1_diloco_comm.jsonl`: 12 lines, all `phase=outer_sync` / `AllReduceMean` / `world_size=2` (3 syncs × 4 fragments).

Fabric summary n0: `eth_total_tx_bytes=1125456630`, `eth_total_rx_bytes=1125369423`.

---

## D — Qwen TP=2 Ray cross-node ✅

Session 2 · ~23:20–23:25Z · label `hour1_qwen_tp2` · port **8010**  
Real TP AllReduce · ens1 Δ 60s ≈ **11.57 GB** rx/tx · GPU ~**76 GB** both nodes

### Commands

```bash
# Ray head (n0) / worker (n1)
export NCCL_SOCKET_IFNAME=ens1 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=ens1
# n0: VLLM_HOST_IP=10.65.0.2
ray start --head --port=6379 --num-gpus=1 --node-ip-address=10.65.0.2
# n1: VLLM_HOST_IP=10.65.0.3
ray start --address=10.65.0.2:6379 --num-gpus=1 --node-ip-address=10.65.0.3

# Engine (n0)
cd /workspace/GPU-monitoring
export HF_HOME=/workspace/hf RAY_ADDRESS=10.65.0.2:6379
export NCCL_SOCKET_IFNAME=ens1 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=ens1
export VLLM_HOST_IP=10.65.0.2 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
nohup /usr/local/bin/python scripts/run_with_telemetry.py \
  --label hour1_qwen_tp2 --output-dir data/hour1_cd --nnodes 2 --node-rank 0 -- \
  /usr/local/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
    --host 0.0.0.0 --port 8010 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --max-model-len 4096 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --trust-remote-code \
  > /workspace/runs/d_engine_tp2.log 2>&1 &

# n1 fabric sidecar while Ray worker holds GPU
nohup /usr/local/bin/python scripts/run_with_telemetry.py \
  --label hour1_qwen_tp2 --output-dir data/hour1_cd --nnodes 2 --node-rank 1 -- \
  bash -c 'while true; do sleep 30; done' > /workspace/runs/d_telem_n1.log 2>&1 &

# Traffic + ens1 delta (n0)
RX0=$(cat /sys/class/net/ens1/statistics/rx_bytes)
TX0=$(cat /sys/class/net/ens1/statistics/tx_bytes)
/usr/local/bin/python workloads/llm_moe_serving.py \
  --model qwen25_7b --tp 2 --dp 1 \
  --no-server --base-url http://127.0.0.1:8010 \
  --traffic mixed --concurrency 8 --duration 60 \
  --enable-prefix-caching \
  --traffic-log data/hour1_cd/hour1_qwen_tp2_mixed.jsonl \
  2>&1 | tee /workspace/runs/d_traffic.log
echo "ens1_delta_rx=$(($(cat /sys/class/net/ens1/statistics/rx_bytes)-RX0)) \
      ens1_delta_tx=$(($(cat /sys/class/net/ens1/statistics/tx_bytes)-TX0))"
```

### Key output

```
(RayWorkerWrapper … ip=10.65.0.3)
Custom allreduce is disabled because this process group spans across nodes.
rank 0 … TP rank 0 · rank 1 … TP rank 1
Application startup complete.
Traffic done: {'prefill': 0, 'decode': 17, 'midfill': 0, 'error': 31}
ens1 Δ ≈ 11.57 GB rx / 11.57 GB tx   # SSH stdout, not in d_traffic.log
```

Logs: `cd_real/runs/{d_engine_tp2,d_traffic}.log` · traffic jsonl under `cd_real/node0/hour1_cd/`

---

## C — OLMoE DP=2 + EP cross-node ✅

Session 2 · ~23:47–23:50Z · label `hour1_olmoe_dp2ep` · port **8000**  
Real EP AllToAll (32/64 experts) · ens1 Δ 60s ≈ **11.89 GB** rx/tx · GPU ~**75 GB** both · **272 OK / 0 err**

### Commands

```bash
# n0 — Ray head + vllm serve DP2+EP
cd /workspace/GPU-monitoring
export MASTER_ADDR=10.65.0.2 NODE_RANK=0 API_PORT=8000 HF_HOME=/workspace/hf
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET VLLM_HOST_IP=10.65.0.2
nohup /usr/local/bin/python scripts/run_with_telemetry.py \
  --label hour1_olmoe_dp2ep --output-dir data/hour1_cd --nnodes 2 --node-rank 0 -- \
  bash scripts/run_hour1_crossnode_serve.sh c-engine \
  > /workspace/runs/c_engine_n0.log 2>&1 &

# n1 — Ray worker (script --block)
export MASTER_ADDR=10.65.0.2 NODE_RANK=1 API_PORT=8000 HF_HOME=/workspace/hf
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET VLLM_HOST_IP=10.65.0.3
nohup /usr/local/bin/python scripts/run_with_telemetry.py \
  --label hour1_olmoe_dp2ep --output-dir data/hour1_cd --nnodes 2 --node-rank 1 -- \
  bash scripts/run_hour1_crossnode_serve.sh c-engine \
  > /workspace/runs/c_engine_n1.log 2>&1 &
```

`c-engine` on n0 expands to (after 2 Ray nodes alive):

```bash
# NCCL_SOCKET_IFNAME=ens1 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=ens1
ray start --head --port=6379 --node-ip-address=10.65.0.2 --num-gpus=1
vllm serve allenai/OLMoE-1B-7B-0924 \
  --dtype bfloat16 --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-backend ray \
  --data-parallel-address 10.65.0.2 \
  --enable-expert-parallel \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

n1: `ray start --address=10.65.0.2:6379 --node-ip-address=10.65.0.3 --num-gpus=1 --block`

Traffic:

```bash
/usr/local/bin/python scripts/hour1_completions_traffic.py \
  --base-url http://127.0.0.1:8000 \
  --model allenai/OLMoE-1B-7B-0924 \
  --duration 60 --concurrency 8 \
  --traffic-log data/hour1_cd/hour1_olmoe_dp2ep_mixed.jsonl \
  2>&1 | tee /workspace/runs/c_traffic.log
# + ens1 before/after byte delta as in D
```

### Key output

```
(DPEngineCoreActor … ip=10.65.0.3) NET/Socket : Using [0]ens1:10.65.0.3
Using network Socket
[EP Rank 0/2] … Local/global experts: 32/64
[EP Rank 1/2] … Local/global experts: 32/64   # map 0->32 … 31->63
Application startup complete.
Traffic done: {'prefill': 57, 'decode': 135, 'midfill': 80, 'error': 0}
ens1 Δ ≈ 11.89 GB rx / 11.89 GB tx
```

### Do not regress (C)

1. `python -m vllm.entrypoints.openai.api_server --headless` **ignores** headless → use **`vllm serve`**.
2. mp/ZMQ DP HELLO deadlock → **`--data-parallel-backend ray`**.
3. Ray must advertise ens1: `--node-ip-address=10.65.0.2|3` else `DP master node missing`.

---

## Issues (append-only)

| ID | Symptom | Cause | Fix / lesson |
|---|---|---|---|
| I1 | NCCL `NET/IB` WC 129 | RoCE datapath broken in container | `NCCL_IB_DISABLE=1` + ens1 Socket |
| I2 | `ib_*` all False | no `counters/` sysfs | `eth_*` + `rdma_stat_*` |
| I3 | CIFAR DDP crash rank1 | dataset download race | FakeData patch |
| I4 | vLLM 0.30 CUDA init fail | torch cu130 vs driver 12.8 | pin torch cu128 + vllm 0.10.2 |
| I5 | multi-node DP+EP won't start | mp ZMQ / wrong entrypoint | Ray DP backend + `vllm serve` |
| I6 | OLMoE chat 400 | no chat template | `/v1/completions` |
| I7 | Qwen on :8001 all errors | nginx owns 8001 | use **8010** |
| I8 | 2×2 H100 Instant Cluster no stock | capacity | ran 2×1 |
| I9 | `api_server --headless` ignored | module entry ≠ `vllm serve` | always `vllm serve` |
| I10 | mp DP ZMQ handshake hang | EngineCore HELLO unanswered | `--data-parallel-backend ray` |
| I11 | Ray DP `master node missing` | Ray node IP ≠ ens1 | `ray start --node-ip-address=$ens1` |

Also: after `libibverbs`, NCCL selected `NET/IB` → WC 129 (`hour1_w01_ibok_*`); kept Socket path.

---

## Follow-ups

1. Pin `vllm==0.10.2` + `torch==2.8.0+cu128` + `transformers~=4.55`; never bare `pip install vllm` on cu128 hosts.
2. Default C/D launcher = Ray + ens1 `--node-ip-address` (already in `run_hour1_crossnode_serve.sh`).
3. Re-try RoCE when RunPod exposes IB `ports/*/counters` (ticket filed).
4. Default serve port away from 8001 on RunPod (nginx).

---

## Artifact map

| Exp | Logs | Telemetry / traffic |
|---|---|---|
| A | `node{0,1}/runs/hour1_w01_ddp_node*.log` | `node*/data/hour1/hour1_w01_ddp_*` |
| B | `node*/runs/hour1_diloco_n*.log` | `hour1_diloco_*` + `runs/hour1_diloco_comm.jsonl` |
| D | `cd_real/runs/d_engine_tp2.log`, `d_traffic.log` | `cd_real/node*/hour1_cd/*qwen*` |
| C | `cd_real/runs/c_engine_n{0,1}.log`, `c_traffic.log` | `cd_real/node*/hour1_cd/*olmoe*` |

Root: `data/hour1_runpod/`.
