"""Cross-rank ("multinode") features — the vocabulary the scope-ladder test needs.

The deployed pipeline treats every GPU as an independent stream: no feature in
`classifier/threeway_improved.py` ever looks at two GPUs jointly, and there is
no interconnect signal at all. The hypothesis under test -- that training is
easier to tell from inference as communication scope grows -- therefore has no
feature vocabulary yet. This module adds it, in two layers:

  1. per-stream comm channels (DCGM NVLink, host NIC) featurised the same way
     the existing extractor treats the 9 NVML signals, and
  2. genuinely cross-rank features that only exist once you join a job's ranks:
     correlation (training is phase-locked by the sync barrier; replicated
     inference is independent; sharded inference is partially coupled), an
     inter-minus-intra-node correlation gap (the S16-specific signature), rank
     lag structure (data-parallel is zero-lag, pipeline-parallel is ordered),
     collective-cadence coherence, and coincident collective stalls.

OBSERVER MODEL: these features assume the operator can join telemetry across
GPUs *and servers* -- an extension of the paper's per-GPU-NVML observer. The
harness that consumes this module (scope_ladder_eval.py) must therefore always
report a paper-faithful per-GPU arm alongside the cross-rank arm.

Identity handling copes with two corpora:
  * NEW multi-node data -- every rank stream carries the job's shared run_id
    (scripts/run_workload_multi_gpu.py --run-id) plus a distinct gpu_uuid and a
    hostname column, so a job = one run_id spanning several streams.
  * LEGACY data -- each per-GPU file has its own collector-minted run_id, so a
    job is reconstructed exactly as classifier/check_job_grouping.py does it:
    (base label with rank tokens stripped, gpu_name, start-time bucket).
Both resolve through assign_stream_and_job_ids() below, so the whole harness --
and this module's tests -- run on the existing 8-GPU FSDP/DDP corpus today.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from classifier.threeway_improved import (
    RAW_SIGNALS,
    acf_lag,  # noqa: F401  (re-exported for harness convenience)
    extract_enhanced_features,
    fft_dominant_period,
    fft_peak_power,
)

# Job-identity bucket, matching check_job_grouping.BUCKET: sibling ranks start
# within a couple of seconds, repeated runs of a cell are minutes apart.
BUCKET = 20.0

# Extra per-stream comm channels, if present after normalize_comm_columns /
# attach_nic_sidecar. Featurised per stream like a RAW_SIGNAL.
EXTRA_SIGNALS = [
    "nvlink_total_mbps", "dcgm_nvlink_tx_mbps", "dcgm_nvlink_rx_mbps",
    "nic_total_mbps", "nic_tx_mbps", "nic_rx_mbps",
    # ── TPU-native channels (present only on TPU parquets; absent = skipped) ──
    # Measured live on a 2x v6e-1 multislice, 2026-09-01, over a 600 s cell with
    # 708 samples at ~1.2 Hz: eight accelerator channels carry real values while
    # power, temperature, clocks, PCIe and hbm_bw_util_pct are dead by contract.
    # Only two of those eight used to be featurised, which left most of the
    # TPU-native information unused -- and the whole premise of the TPU rung is
    # whether these channels can stand in for the power channel that does not
    # exist here. RAW_SIGNALS is deliberately untouched: it is shared with the
    # GPU corpus, where none of these columns exist.
    # CAVEAT, verified from collect_tpu_telemetry.py: the _p50/_p90/_p99 suffixes
    # are NOT percentiles. libtpu's distribution payload carries only
    # count/mean/min/max, and the collector passes [min, mean, max] into
    # np.percentile, so _p50 IS the interval mean, _p99 is ~the interval max, and
    # _p90 is an interpolation 80% of the way between them with no statistical
    # meaning. Treat _p90 as redundant with _p99, and check any LEVEL feature on
    # hlo_exec_us_* against window index before trusting it: the series climbs two
    # orders of magnitude across a run, so its level partly encodes how far into
    # the run a window sits. dcn_collective_lat_us_* is additionally nonzero on
    # cells that run no cross-slice collectives (nonml_duty30 medians ~86,000),
    # because jax.distributed.initialize()'s own traffic lands in the first
    # sampling interval. See results/tpu_multislice_findings.md section 2b.
    "hlo_exec_us_p50", "hlo_exec_us_p90", "hlo_exec_us_p99", "hlo_queue_size",
    "dcn_collective_lat_us_p50", "dcn_collective_lat_us_p90",
    "dcn_collective_lat_us_p99", "buffer_xfer_lat_us_p90",
    "tensorcore_util_pct",
    # Host<->device transfer latencies, added 2026-09-02 once the metrics server
    # was enumerated properly (it serves 11 names; only 7 were being read).
    # CAVEAT measured 2026-09-03: these are megascale.* metrics and are live ONLY
    # ON MULTISLICE (666/683 samples on a 2-slice group, 0/661 on a single slice),
    # exactly like dcn_collective_lat_us_*. They cannot contribute anything on a
    # single-slice corpus. Where they DO exist they are plausibly the two channels
    # best placed to separate TRAINING from INFERENCE --
    # training streams batches host->device, inference streams generated tokens
    # device->host -- which is Stage 2's job and the part of the pipeline the TPU
    # corpus could not train at all until inference cells existed.
    "h2d_transfer_lat_us_p50", "h2d_transfer_lat_us_p90",
    "d2h_transfer_lat_us_p50", "d2h_transfer_lat_us_p90",
    # NOT tpu_uptime_s: it rises monotonically through a run, so as a feature it
    # encodes how far into the run a window sits rather than anything about the
    # workload. NOT slice_error_detected either: that is a data-quality flag for
    # filtering windows, not a signal about the workload.
    # NOT tpu_duty_cycle_pct: collect_tpu_telemetry writes the same `duty` value
    # to both it and gpu_utilization_pct, differing only in the missing-value
    # sentinel (0.0 vs -1.0), so including it would duplicate a RAW_SIGNAL and
    # split its importance in two. Note the consequence for interpretation:
    # gpu_utilization_pct on TPU IS TensorCore duty cycle, the same kind of
    # silent redefinition as mem_utilization_pct meaning HBM bandwidth here.
]

# Signals used for the cross-rank (stream x stream) features. Only those
# actually present in the frame are used; the rest are skipped so the feature
# set adapts to GPU vs TPU vs legacy data without erroring.
XR_SIGNALS = [
    "gpu_utilization_pct", "power_draw_w", "mem_utilization_pct",
    "pcie_total_mbps", "nvlink_total_mbps", "nic_total_mbps",
    # Cross-rank structure of the TPU comm channels: a straggler slice shows up
    # as dispersion across ranks, which is exactly what the DCN channel is for.
    "hlo_exec_us_p90", "dcn_collective_lat_us_p90", "tensorcore_util_pct",
]

_RANK_TOKEN_RE = re.compile(r"_node\d+_gpu\d+|_gpu\d+|_w\d+c\d+")


def _base_label(label: str) -> str:
    return _RANK_TOKEN_RE.sub("", str(label))


def normalize_comm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive mbps + total comm columns from raw counters, in place-ish.

    DCGM reports NVLink traffic in bytes/sample; convert to MiB/s to match the
    pcie_*_mbps convention (bytes are per 1 s sample at 1 Hz). Also synthesise
    ``pcie_total_mbps`` and ``nvlink_total_mbps`` / ``nic_total_mbps`` so the
    cross-rank layer has single per-direction-summed channels.
    """
    df = df.copy()
    mib = 1024.0 * 1024.0

    for d in ("tx", "rx"):
        raw = f"dcgm_nvlink_{d}_bytes"
        col = f"dcgm_nvlink_{d}_mbps"
        if raw in df.columns and col not in df.columns:
            df[col] = pd.to_numeric(df[raw], errors="coerce").fillna(0.0) / mib

    if {"dcgm_nvlink_tx_mbps", "dcgm_nvlink_rx_mbps"} <= set(df.columns):
        df["nvlink_total_mbps"] = (df["dcgm_nvlink_tx_mbps"].clip(lower=0)
                                   + df["dcgm_nvlink_rx_mbps"].clip(lower=0))
    if {"pcie_tx_mbps", "pcie_rx_mbps"} <= set(df.columns):
        # -1 sentinels (fanless/absent) must not pollute the sum.
        tx = df["pcie_tx_mbps"].where(df["pcie_tx_mbps"] >= 0, 0.0)
        rx = df["pcie_rx_mbps"].where(df["pcie_rx_mbps"] >= 0, 0.0)
        df["pcie_total_mbps"] = tx + rx
    if {"nic_tx_mbps", "nic_rx_mbps"} <= set(df.columns) and "nic_total_mbps" not in df.columns:
        df["nic_total_mbps"] = (df["nic_tx_mbps"].clip(lower=0)
                                + df["nic_rx_mbps"].clip(lower=0))
    return df


def attach_nic_sidecar(df: pd.DataFrame, nic_glob: str = "data/**/nic_*.parquet") -> pd.DataFrame:
    """Left-join host-NIC sidecar series onto every GPU row of the same host.

    The join key is (node_id, integer epoch second): the sidecar and the GPU
    collector both stamp time.time() and NTP keeps hosts within milliseconds,
    so rounding to the second is a safe, tolerant join. Missing side => the
    nic_* columns stay absent and the cross-rank layer simply skips them.
    """
    from pathlib import Path
    from glob import glob

    paths = sorted(glob(str(Path(nic_glob)), recursive=True))
    if not paths:
        return df
    sides = []
    for p in paths:
        try:
            s = pd.read_parquet(p)
        except Exception:
            continue
        if "timestamp_epoch" in s.columns:
            s = s.rename(columns={"timestamp_epoch": "ts"})
        sides.append(s)
    if not sides:
        return df
    nic = pd.concat(sides, ignore_index=True)
    keep = [c for c in ("ts", "run_id", "hostname", "node_rank", "nic_tx_mbps",
                        "nic_rx_mbps", "nic_total_rx_mbps", "nic_total_tx_mbps",
                        "cpu_util_pct") if c in nic.columns]
    nic = nic[keep].copy()
    nic["_sec"] = nic["ts"].round().astype("int64")
    join_host = "hostname" in df.columns and "hostname" in nic.columns
    df = df.copy()
    df["_sec"] = df["ts"].round().astype("int64")
    on = ["run_id", "_sec"] + (["hostname"] if join_host else [])
    nic_j = nic.drop(columns=["ts"]).drop_duplicates(on)
    merged = df.merge(nic_j, on=on, how="left", suffixes=("", "_nic"))
    return merged.drop(columns=["_sec"], errors="ignore")


def assign_stream_and_job_ids(df: pd.DataFrame, bucket_sec: float = BUCKET) -> pd.DataFrame:
    """Add `stream_id`, `job_id`, `node_id`, `rank` columns.

    stream_id = one physical GPU's telemetry stream. Under a shared run_id this
    is (run_id, gpu_uuid); for legacy per-collector run_ids the run_id alone is
    already unique, and adding gpu_uuid is harmless.

    job_id = one physical execution (all its ranks). If a run_id already spans
    several streams (the shared-run_id case), that run_id *is* the job. If not
    (legacy), reconstruct the job the check_job_grouping way so sibling files of
    one 8-GPU run collapse together.
    """
    df = df.copy()

    def _idcol(name, default=""):
        """NA-free string column for identity concatenation.

        Every identity column here is string-concatenated into stream_id/job_id,
        and a NULL anywhere poisons the whole corpus: the concatenation goes
        NULL, groupby(stream_id) DROPS that key, transform("min") returns NaN for
        those rows, and `(run_start / bucket_sec).astype("int64")` then raises
        IntCastingNaNError -- for every run, not just the affected one.
        `.astype(str)` does not save you: under the pandas `str` dtype NA
        propagates through `+` rather than becoming "nan". This is not
        hypothetical: on the full 3.6 M-row corpus gpu_uuid is null on 16,812
        rows and gpu_name on 14,185, because load_all_data concatenates sections
        (week4/5/7, ddp, dscale) whose frames never had those columns.
        """
        if name not in df.columns:
            return pd.Series(default, index=df.index, dtype=object)
        s = df[name].astype(object)
        return s.where(df[name].notna(), default).astype(str)

    uuid_col = _idcol("gpu_uuid")
    df["stream_id"] = _idcol("run_id") + "|" + uuid_col
    df["node_id"] = _idcol("hostname", "node0") if "hostname" in df.columns else "node0"

    # Which run_ids span more than one stream? Those are shared-run_id jobs.
    streams_per_run = df.groupby("run_id")["stream_id"].transform("nunique")
    run_start = df.groupby("stream_id")["ts"].transform("min")
    # A stream whose ts are all NULL still yields NaN here; it cannot be windowed
    # anyway, so bucket it at 0 rather than failing the cast for the whole frame.
    run_start = run_start.astype("float64").fillna(0.0)
    base = df["workload_label"].map(_base_label)
    gpu = _idcol("gpu_name")
    legacy_job = (base + "|" + gpu + "|"
                  + (run_start / bucket_sec).round().astype("int64").astype(str))
    df["job_id"] = np.where(streams_per_run > 1, _idcol("run_id"), legacy_job)

    # A shared run_id does NOT always mean a shared JOB. multinode_queue.sh derives
    # the run_id from (stamp, cell, rep) with no node component, so on a
    # SINGLE-NODE cell (s1_/s8_) each node runs its own independent execution while
    # both carry the same run_id. Grouping those as one job splices two unrelated
    # executions together: an s8 "job" becomes 16 streams spanning two hosts, an s1
    # "job" becomes 2 streams (which is why check_job_grouping reported zero
    # single-GPU training jobs). The cross-rank features would then correlate
    # genuinely-coupled ranks against unrelated ones, diluting the s8 signal and
    # biasing DeltaAUC(s8-s1) down / DeltaAUC(s16-s8) up -- flattering the
    # hypothesis. Scope tells us the truth: s1/s8 are per-node by construction,
    # s16 (and the TPU tpuT* cells) genuinely span hosts.
    _per_node_scope = df["workload_label"].map(lambda l: _scope_of(l) in ("s1", "s8"))
    df["job_id"] = np.where(_per_node_scope,
                            df["job_id"].astype(str) + "|" + df["node_id"].astype(str),
                            df["job_id"].astype(str))

    # Ordinal rank within a job: a stable distinct integer per stream. Ranks of
    # a distributed job start simultaneously, so ordering by start time would
    # tie them all together -- order by stream_id instead.
    uniq = (df[["job_id", "stream_id"]].drop_duplicates()
            .sort_values(["job_id", "stream_id"]))
    uniq["rank"] = uniq.groupby("job_id").cumcount()
    df = df.merge(uniq, on=["job_id", "stream_id"], how="left")
    return df


def _stream_matrix(job_win: pd.DataFrame, signal: str):
    """[T x R] matrix of `signal` over the common integer-second grid of a window.

    Each stream's samples are binned to integer epoch seconds and aligned on the
    seconds every stream shares, so drifting 1 Hz collectors still line up.
    Returns (matrix, node_of_rank) or (None, None) if <2 streams overlap.
    """
    if signal not in job_win.columns:
        return None, None
    g = job_win[["rank", "node_id", "ts", signal]].copy()
    g = g[g[signal].notna()]
    # -1 sentinels mean "channel absent" -> treat as no data for correlations.
    g = g[g[signal] != -1]
    if g.empty:
        return None, None
    g["_sec"] = g["ts"].round().astype("int64")
    piv = g.pivot_table(index="_sec", columns="rank", values=signal, aggfunc="mean")
    piv = piv.dropna(axis=0, how="any")
    if piv.shape[0] < 5 or piv.shape[1] < 2:
        return None, None
    node_of_rank = (g.drop_duplicates("rank").set_index("rank")["node_id"].to_dict())
    return piv, node_of_rank


def _pairwise_corr(mat: pd.DataFrame):
    """List of (rank_i, rank_j, pearson_r) over columns of an aligned matrix."""
    cols = list(mat.columns)
    out = []
    for a in range(len(cols)):
        xa = mat[cols[a]].values
        for b in range(a + 1, len(cols)):
            xb = mat[cols[b]].values
            if xa.std() < 1e-9 or xb.std() < 1e-9:
                r = 0.0
            else:
                r = float(np.corrcoef(xa, xb)[0, 1])
                if not np.isfinite(r):
                    r = 0.0
            out.append((cols[a], cols[b], r))
    return out


def _best_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> int:
    """Lag (in samples) of y vs x maximising cross-correlation, within +-max_lag."""
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = np.dot(x[-lag:], y[:len(y) + lag]) / (len(x) + lag)
        elif lag > 0:
            c = np.dot(x[:len(x) - lag], y[lag:]) / (len(x) - lag)
        else:
            c = np.dot(x, y) / len(x)
        if c > best:
            best, best_lag = c, lag
    return best_lag


def cross_rank_features(job_win: pd.DataFrame) -> dict:
    """The xr_* vocabulary for one (job, window): correlation, inter-vs-intra
    node coupling, rank lag structure, collective-cadence coherence, coincident
    stalls, and level spread. Computed only for signals actually present."""
    feats = {}
    ranks = sorted(job_win["rank"].dropna().unique())
    feats["xr_n_ranks"] = float(len(ranks))
    # active fraction across the window (util > 10 anywhere in a stream)
    if "gpu_utilization_pct" in job_win.columns and ranks:
        act = (job_win.groupby("rank")["gpu_utilization_pct"].max() > 10).mean()
        feats["xr_active_ranks_frac"] = float(act)

    for sig in XR_SIGNALS:
        mat, node_of_rank = _stream_matrix(job_win, sig)
        if mat is None:
            continue
        pairs = _pairwise_corr(mat)
        rs = np.array([r for _, _, r in pairs], dtype=float)
        feats[f"xr_corr_mean_{sig}"] = float(rs.mean())
        feats[f"xr_corr_min_{sig}"] = float(rs.min())
        feats[f"xr_corr_frachi_{sig}"] = float((rs > 0.5).mean())

        # inter- vs intra-node correlation: the S16-specific signature.
        inter = [r for i, j, r in pairs if node_of_rank.get(i) != node_of_rank.get(j)]
        intra = [r for i, j, r in pairs if node_of_rank.get(i) == node_of_rank.get(j)]
        if inter and intra:
            im, am = float(np.mean(inter)), float(np.mean(intra))
            feats[f"xr_corr_internode_mean_{sig}"] = im
            feats[f"xr_corr_intranode_mean_{sig}"] = am
            feats[f"xr_corr_inter_minus_intra_{sig}"] = im - am

        # rank lag structure vs a reference rank (min rank present).
        ref = mat.columns[0]
        xref = mat[ref].values
        lags = [abs(_best_lag(xref, mat[c].values)) for c in mat.columns[1:]]
        if lags:
            feats[f"xr_lag_median_{sig}"] = float(np.median(lags))
            feats[f"xr_lag_iqr_{sig}"] = float(np.subtract(*np.percentile(lags, [75, 25])))

        # collective-cadence coherence: do all ranks share one step period?
        periods = [fft_dominant_period(mat[c].values) for c in mat.columns]
        peaks = [fft_peak_power(mat[c].values) for c in mat.columns]
        feats[f"xr_fft_period_disp_{sig}"] = float(np.std(periods))
        feats[f"xr_fft_peakpow_min_{sig}"] = float(np.min(peaks))

        # coincident stalls: fraction of a rank's low-activity seconds shared by
        # >=75% of ranks (synchronised collective waits vs independent idles).
        lo = mat < mat.quantile(0.25)
        shared = (lo.sum(axis=1) >= 0.75 * mat.shape[1])
        denom = lo.any(axis=1).sum()
        feats[f"xr_dip_coalign_{sig}"] = float(shared.sum() / denom) if denom else 0.0

        # scale-free imbalance (shape-safe for stage 2): cross-rank CV of means.
        means = mat.mean(axis=0).values
        feats[f"xr_level_spread_{sig}"] = float(means.std() / (abs(means.mean()) + 1e-9))

    return feats


def sliding_windows_multinode(
    df: pd.DataFrame,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    min_rank_coverage: float = 0.8,
) -> pd.DataFrame:
    """Job-level, epoch-aligned windows carrying rank-0 features + cross-rank
    features. One row per (job_id, window). ``run_id`` is set to ``job_id`` so
    downstream run-level detection groups by physical job automatically.

    Rank-0 features reuse extract_enhanced_features VERBATIM (names unchanged),
    so twostage's level-feature drop keeps working on this block.
    """
    df = assign_stream_and_job_ids(df)
    rows = []
    for job_id, job_df in df.groupby("job_id"):
        job_df = job_df.sort_values("ts")
        n_ranks = job_df["rank"].nunique()
        t0 = job_df["ts"].min()
        t_end = job_df["ts"].max()
        # Epoch-aligned grid so every rank shares exact window boundaries.
        wstart = np.ceil(t0 / stride_sec) * stride_sec
        label = job_df["threeway_label"].iloc[0]
        wl = _base_label(job_df["workload_label"].iloc[0])
        gpu = job_df["gpu_name"].iloc[0] if "gpu_name" in job_df.columns else ""
        scope = _scope_of(job_df["workload_label"].iloc[0])
        # canonical stream for the single-stream feature block = lowest rank
        ref_rank = job_df["rank"].min()

        while wstart + window_sec <= t_end + 1:
            wend = wstart + window_sec
            win = job_df[(job_df["ts"] >= wstart) & (job_df["ts"] < wend)]
            ranks_here = win["rank"].nunique()
            if ranks_here == 0:
                wstart += stride_sec
                continue
            ref = win[win["rank"] == ref_rank]
            if len(ref) < 5:
                wstart += stride_sec
                continue
            feats = extract_enhanced_features(ref)
            feats.update(cross_rank_features(win))
            # per-rank comm-channel aggregates (mean over ranks of window means)
            for sig in EXTRA_SIGNALS:
                if sig in win.columns:
                    v = win[sig].where(win[sig] != -1, np.nan)
                    feats[f"xragg_{sig}_mean"] = float(v.mean()) if v.notna().any() else 0.0
                    feats[f"xragg_{sig}_max"] = float(v.max()) if v.notna().any() else 0.0
            feats["run_id"] = job_id           # group-by key for detection
            feats["threeway_label"] = label
            feats["workload_label"] = wl
            feats["gpu_name"] = gpu
            feats["window_start"] = wstart
            feats["window_end"] = wend
            # meta columns as STRINGS so get_feature_cols never treats them as
            # features (they would otherwise leak scope/rank identity).
            feats["meta_scope"] = scope
            feats["meta_n_ranks"] = str(n_ranks)
            feats["meta_rank_coverage"] = f"{ranks_here / max(n_ranks, 1):.3f}"
            feats["meta_degraded"] = str(ranks_here < min_rank_coverage * n_ranks)
            rows.append(feats)
            wstart += stride_sec

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# tpuTMS\d* covers the multislice ladder rungs (tpuTMS = 2 slices, tpuTMS4,
# tpuTMS8). Without the \d* the 4- and 8-slice cells match nothing here and are
# dropped from the multinode features silently, which is the same failure mode
# that once discarded "Google TPU v6e" rows without a count.
_SCOPE_RE = re.compile(r"^(s1|s8|s16|tpuT\d+|tpuTMS\d*)_", re.IGNORECASE)


def _scope_of(label: str) -> str:
    m = _SCOPE_RE.match(str(label))
    return m.group(1).lower() if m else "na"


# Level statistics that stage 2 must not see (an evader modulates absolute
# levels freely). Mirrors twostage.LEVEL_FEATURE_NAMES semantics, extended to
# the new comm/cross-rank names. Applied by stage2_mask_multinode.
_LEVEL_SUFFIXES = ("_mean", "_min", "_max", "_p25", "_p50", "_p75", "_p95", "_range")


def stage2_mask_multinode(feature_cols) -> np.ndarray:
    """Boolean keep-mask dropping absolute-level features (incl. new comm /
    xragg level stats), keeping shape/structure (cv, fft, acf, correlation,
    lag, nonzero_frac). Lives here so twostage.py stays untouched."""
    keep = []
    for c in feature_cols:
        is_level = c.endswith(_LEVEL_SUFFIXES) or c == "pcie_total_mean"
        # cross-rank correlation/lag/cadence are structure, never levels.
        if c.startswith(("xr_corr", "xr_lag", "xr_fft", "xr_dip", "xr_active", "xr_n_ranks")):
            is_level = False
        keep.append(not is_level)
    return np.array(keep, dtype=bool)
