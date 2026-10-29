# Which rentals to take, in order

Ranked by the tier-A windows each one restores, estimated from the healthy siblings of the same cell on the same device. Re-collections can be promoted back into tier A (`data/tier_a_recollections.json`), so these windows return to the frozen corpus rather than only to coverage claims.

| # | rent | cells | runs | windows restored | quickstart |
| --: | :-- | --: | --: | --: | :-- |
| 1 | 1x H200 141GB, plus 8x H200 for the FSDP cells | 3 | 9 | ~303 | `quickstart_h200.md` |
| 2 | 1x A100 80GB, plus 8x A100 80GB for the FSDP cells | 3 | 9 | ~224 | `quickstart_a100_80gb.md` |
| 3 | 1x B200 | 4 | 12 | ~101 | `quickstart_b200.md` |
| 4 | 1x H100 SXM 80GB | 5 | 15 | ~30 | `quickstart_h100_80gb.md` |
| 5 | 1x RTX 5080 (Vast.ai) | 2 | 6 | ~26 | `quickstart_rtx5080.md` |
| 6 | 1x H100 **NVL** (94GB PCIe) | 1 | 3 | ~4 | `quickstart_h100_nvl.md` |

**54 runs, ~688 windows, 6 rentals.**

Where a device has both single- and multi-GPU cells, one multi-GPU rental covers both: a 1-GPU cell runs on device 0 and leaves the rest idle.

The bottom of this list earns its place poorly. H100 NVL is one cell worth about four windows -- a whole rental for a rounding error. Worth stating as a gap instead unless the rental is nearly free.

Not reproducible at all: `baseline_infer`, `baseline_train`, `ffmpeg_vaapi`, `nvlink_latency`, `nvlink_p2p` have no registry entry, so there is no command to re-run them.

