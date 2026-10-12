"""Recorded base seed for repeatable distributed mechanism probes.

Rank/stage/shard offsets remain the workload's responsibility. Setting a seed
does not make asynchronous CUDA collectives or wall-clock scheduling deterministic.
"""
import argparse
import json
import os
import random


def seed_value(value):
    value = int(value)
    if not 0 <= value < 2**31:
        raise argparse.ArgumentTypeError('seed must be in [0, 2**31)')
    return value


def add_seed_argument(parser):
    parser.add_argument('--seed', type=seed_value,
                        default=os.environ.get('SPAR_WORKLOAD_SEED', '1234'),
                        help='Base model/data seed; default SPAR_WORKLOAD_SEED or 1234')


def record_seed(seed):
    """Seed auxiliary generators and emit provenance before distributed setup."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    print('RUN_SEED ' + json.dumps(dict(base_seed=seed,
        rank=int(os.environ.get('RANK', '0')),
        local_rank=int(os.environ.get('LOCAL_RANK', '0')),
        run_stamp=os.environ.get('RUN_STAMP'),
        deterministic_algorithms=False)), flush=True)


def record_progress(seed, updates, tokens, elapsed_s, *, infer_steps=0):
    """Per-rank completed work, not a claim of task learning or unique tokens."""
    print('RUN_PROGRESS ' + json.dumps(dict(base_seed=seed,
        rank=int(os.environ.get('RANK', '0')), run_stamp=os.environ.get('RUN_STAMP'),
        optimizer_updates=updates, local_tokens=tokens, elapsed_s=elapsed_s,
        infer_steps=infer_steps, token_accounting='local; TP replicas may overlap',
        useful_task_learning='not_measured')), flush=True)
