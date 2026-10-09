# Hung streams — excluded from the corpus

`_quarantine_idle` screens on median utilisation, which cannot see these.

`s16_train_swarm_s4` (aec76ae45865) and `s16_train_swarm_blocked`
(23fe2bf61b21) were killed by their cell timeouts on 2026-10-02, deadlocked
in NCCL point-to-point: every rank blocked, and a blocked NCCL collective
spins its kernel, so NVML reports **100% utilisation**. swarm_s4 wrote 1566
rows at util_med 100 while doing no work at all. The utilisation screen
passed 11 of its streams and 5 of swarm_blocked's for exactly that reason,
and I read those numbers as a healthy cell before checking the log.

Root cause and fix: NCCL P2P does not work under partial participation when
interleaved with collectives the non-participating ranks race ahead into
(commit b626eb84, "P2P moves to gloo"). Both cells were re-collected in
session 20261003 and ran properly:

    s16_train_swarm_s4       4936 rows, util_med 94
    s16_train_swarm_blocked  37927 ticks, 2 migrations, 509 GB inter-node

Use those. The streams here are kept rather than deleted because they are the
only recorded example of a deadlock that looks identical to a saturated GPU,
which is the failure mode the idle screen is structurally blind to.

The loader skips any path containing `_quarantine`
(classifier/threeway_improved.py:356).
