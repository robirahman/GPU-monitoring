#!/usr/bin/env python3
"""Pre-draw feasibility: does each candidate lineage do real work on this H200 in the campaign envs?

A lineage is feasible if its check completes on the GPU with real data and produces
a sane output (finite decreasing-or-finite loss for training, valid output for
inference, completed GPU run for non-ML). No sensor data is recorded and nothing
here is used for calibration. Output: one JSON line per lineage.

  python -m h200_campaign.partition.feasibility ml      # in ~/venvs/main
  python -m h200_campaign.partition.feasibility other   # in ~/venvs/nonml
"""
from __future__ import annotations

import json
import sys
import time
import traceback


def _report(name, fn):
    t0 = time.time()
    try:
        info = fn() or {}
        ok = info.pop("_ok", True)
        res = {"lineage": name, "feasible": bool(ok), **info}
    except Exception as e:
        res = {"lineage": name, "feasible": False, "error": f"{type(e).__name__}: {e}"[:400],
               "trace_tail": traceback.format_exc()[-600:]}
    res["check_s"] = round(time.time() - t0, 1)
    try:
        import torch
        if torch.cuda.is_available():
            res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(json.dumps(res), flush=True)


# ------------------------------------------------------------------ ML
def _train_steps(model, batch_fn, n=3, lr=1e-4, params=None):
    import torch
    opt = torch.optim.AdamW(params or [p for p in model.parameters() if p.requires_grad], lr=lr)
    losses, times = [], []
    model.train()
    for _ in range(n):
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = batch_fn()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(float(loss))
        times.append(time.time() - t)
    import math
    return {"losses": [round(x, 4) for x in losses], "step_s": [round(x, 3) for x in times],
            "_ok": all(math.isfinite(x) for x in losses)}


def check_t5():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    name = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSeq2SeqLM.from_pretrained(name).cuda()
    ds = load_dataset("knkarthick/samsum", split="train").select(range(64))
    def b():
        x = tok(["summarize: " + d for d in ds["dialogue"][:16]], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        y = tok(ds["summary"][:16], return_tensors="pt", padding=True, truncation=True, max_length=96).input_ids.cuda()
        y[y == tok.pad_token_id] = -100
        return m(**x, labels=y).loss
    r = _train_steps(m, b)
    m.eval()
    x = tok("summarize: " + ds["dialogue"][0], return_tensors="pt").to("cuda")
    r["sample_output"] = tok.decode(m.generate(**x, max_new_tokens=40)[0], skip_special_tokens=True)
    r["_ok"] = r["_ok"] and len(r["sample_output"].split()) > 2
    return r


def _vision(name):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForImageClassification
    m = AutoModelForImageClassification.from_pretrained(name, num_labels=100, ignore_mismatched_sizes=True).cuda()
    ds = load_dataset("uoft-cs/cifar100", split="train").select(range(64))
    x = torch.tensor(np.stack([np.asarray(i) for i in ds["img"]])).permute(0, 3, 1, 2).float().cuda() / 255
    x = F.interpolate(x, size=224, mode="bilinear")
    y = torch.tensor(ds["fine_label"]).cuda()
    r = _train_steps(m, lambda: m(pixel_values=x, labels=y).loss)
    m.eval()
    with torch.no_grad():
        r["infer_logits_shape"] = list(m(pixel_values=x[:8]).logits.shape)
    return r


def check_swin():
    return _vision("microsoft/swin-tiny-patch4-window7-224")


def check_efficientnet():
    return _vision("google/efficientnet-b0")


def _causal(name, lora=True):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
    if lora:
        from peft import LoraConfig, get_peft_model
        lin = sorted({n.split(".")[-1] for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)} - {"lm_head", "embed_out"})
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=lin, task_type="CAUSAL_LM"))
    ds = load_dataset("roneneldan/TinyStories", split="validation").select(range(32))
    x = tok(ds["text"][:8], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
    lab = x.input_ids.clone()
    lab[x.attention_mask == 0] = -100
    r = _train_steps(m, lambda: m(**x, labels=lab).loss)
    m.eval()
    p = tok("Once upon a time", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**p, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    r["sample_output"] = tok.decode(out[0], skip_special_tokens=True)
    r["lora"] = lora
    return r


def check_pythia():
    return _causal("EleutherAI/pythia-410m")


def check_olmo():
    return _causal("allenai/OLMo-2-0425-1B")


def check_falcon():
    return _causal("tiiuae/Falcon3-1B-Base")


def check_granite():
    return _causal("ibm-granite/granite-3.3-2b-base")


def check_mamba():
    return _causal("state-spaces/mamba-370m-hf", lora=False)


def check_gnn_gin():
    import torch
    from datasets import load_dataset
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GINConv, global_add_pool
    ds = load_dataset("OGB/ogbg-molhiv", split="train").select(range(256))
    graphs = [Data(x=torch.tensor(g["node_feat"], dtype=torch.float), edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                   y=torch.tensor([g["y"][0]], dtype=torch.float)) for g in ds]
    batch = Batch.from_data_list(graphs).cuda()

    class GIN(torch.nn.Module):
        def __init__(self, d_in, h=300, L=5):
            super().__init__()
            self.convs = torch.nn.ModuleList(GINConv(torch.nn.Sequential(torch.nn.Linear(d_in if i == 0 else h, h),
                                                                           torch.nn.ReLU(), torch.nn.Linear(h, h))) for i in range(L))
            self.out = torch.nn.Linear(h, 1)

        def forward(self, b):
            x = b.x
            for c in self.convs:
                x = torch.relu(c(x, b.edge_index))
            return self.out(global_add_pool(x, b.batch)).squeeze(-1)
    m = GIN(batch.x.shape[1]).cuda()
    lossf = torch.nn.BCEWithLogitsLoss()
    r = _train_steps(m, lambda: lossf(m(batch).float(), batch.y), lr=1e-3)
    r["graphs"] = len(graphs)
    return r


def check_wav2vec2():
    import io
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    name = "facebook/wav2vec2-large-960h-lv60-self"
    proc = AutoProcessor.from_pretrained(name)
    m = Wav2Vec2ForCTC.from_pretrained(name, dtype=torch.float16).cuda().eval()
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test").cast_column("audio", Audio(decode=False)).select(range(4))
    outs = []
    for r in ds:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        x = proc(a, sampling_rate=sr, return_tensors="pt").input_values.cuda().half()
        with torch.no_grad():
            ids = m(x).logits.argmax(-1)
        outs.append((r["text"].lower(), proc.batch_decode(ids)[0].lower()))
    match = sum(w in h.split() for ref, h in outs for w in ref.split()) / sum(len(ref.split()) for ref, _ in outs)
    return {"word_overlap": round(match, 3), "sample_output": outs[0][1][:120], "_ok": match > 0.8}


def check_dit():
    import torch
    from diffusers import DiTPipeline
    p = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16).to("cuda")
    t = time.time()
    img = p(class_labels=[207, 360], num_inference_steps=20, generator=torch.manual_seed(0)).images
    return {"images": len(img), "size": list(img[0].size), "gen_s": round(time.time() - t, 2), "_ok": len(img) == 2}


def check_bark():
    import torch
    from transformers import AutoProcessor, BarkModel
    proc = AutoProcessor.from_pretrained("suno/bark-small")
    m = BarkModel.from_pretrained("suno/bark-small", dtype=torch.float16).cuda()
    x = proc("The quick brown fox jumps over the lazy dog.", voice_preset="v2/en_speaker_6").to("cuda")
    t = time.time()
    with torch.no_grad():
        audio = m.generate(**x)
    n = audio.shape[-1]
    return {"audio_samples": int(n), "audio_s": round(n / m.generation_config.sample_rate, 2), "gen_s": round(time.time() - t, 2),
            "_ok": n > m.generation_config.sample_rate}


# ------------------------------------------------------------------ non-ML (run in ~/venvs/nonml)
def check_openmm():
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(6, 6, 6) * u.nanometer)
    sysm = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=1 * u.nanometer, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(300 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
    plat = mm.Platform.getPlatformByName("CUDA")
    sim = app.Simulation(mod.topology, sysm, integ, plat)
    sim.context.setPositions(mod.positions)
    sim.minimizeEnergy(maxIterations=100)
    t = time.time()
    sim.step(2000)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    import math
    return {"atoms": mod.topology.getNumAtoms(), "platform": plat.getName(), "steps_per_s": round(2000 / (time.time() - t), 1),
            "potential_kj_mol": round(e, 1), "_ok": math.isfinite(e)}


def check_rapids():
    import cudf
    import cugraph
    import cupy as cp
    n = 20_000_000
    df = cudf.DataFrame({"k": cp.random.randint(0, 100_000, n), "v": cp.random.random(n)})
    t = time.time()
    g = df.groupby("k").v.agg(["mean", "sum", "count"])
    j = df.merge(g.reset_index(), on="k").shape[0]
    t1 = time.time() - t
    src = cp.random.randint(0, 1_000_000, 10_000_000)
    dst = cp.random.randint(0, 1_000_000, 10_000_000)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(cudf.DataFrame({"s": src, "d": dst}), source="s", destination="d")
    t = time.time()
    pr = cugraph.pagerank(G)
    return {"groupby_join_s": round(t1, 2), "join_rows": int(j), "pagerank_s": round(time.time() - t, 2),
            "pagerank_sum": round(float(pr.pagerank.sum()), 4), "_ok": abs(float(pr.pagerank.sum()) - 1) < 1e-3}


def check_qiskit_aer():
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFT
    from qiskit_aer import AerSimulator
    nq = 28
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    qc.append(QFT(nq), range(nq))
    qc.measure_all()
    sim = AerSimulator(method="statevector", device="GPU")
    t = time.time()
    res = sim.run(transpile(qc, sim), shots=100).result()
    return {"qubits": nq, "device": res.results[0].metadata.get("device"), "sim_s": round(time.time() - t, 2),
            "_ok": res.success and res.results[0].metadata.get("device") == "GPU"}


def check_cupy_montecarlo():
    import cupy as cp
    S0, K, r, sig, T, n = 100.0, 105.0, 0.03, 0.2, 1.0, 50_000_000
    t = time.time()
    z = cp.random.standard_normal(n, dtype=cp.float32)
    ST = S0 * cp.exp((r - 0.5 * sig ** 2) * T + sig * cp.sqrt(T) * z)
    price = float(cp.exp(-r * T) * cp.maximum(ST - K, 0).mean())
    from math import erf, exp, log, sqrt
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    d1 = (log(S0 / K) + (r + sig ** 2 / 2) * T) / (sig * sqrt(T))
    bs = S0 * N(d1) - K * exp(-r * T) * N(d1 - sig * sqrt(T))
    return {"mc_price": round(price, 4), "black_scholes": round(bs, 4), "mc_s": round(time.time() - t, 3),
            "_ok": abs(price - bs) < 0.05}


ML = ["t5", "swin", "efficientnet", "pythia", "olmo", "falcon", "granite", "mamba", "gnn_gin", "wav2vec2", "dit", "bark"]
OTHER = ["openmm", "rapids", "qiskit_aer", "cupy_montecarlo"]

if __name__ == "__main__":
    which = sys.argv[1]
    names = sys.argv[2:] or (ML if which == "ml" else OTHER)
    for n in names:
        _report(n, globals()[f"check_{n}"])
