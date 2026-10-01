#!/usr/bin/env python3
"""Guard the HPL input deck's positional layout.

HPL.dat is read by line INDEX, so one missing or extra line shifts everything
after it and HPL rejects the file naming a field you never touched. Measured on
2026-10-01: a missing count line pushed NDIVs onto a "1", and the container
reported "Value of NDIV less than 2" -- a complaint about line 19 caused by
line 15. The cell failed in 12 s having done no work.

    python3 scripts/test_hpl_deck.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.hpc_container import make_dat  # noqa: E402

FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


for ngpu in (1, 2, 4, 8, 16):
    L = make_dat(ngpu, 45000).splitlines()
    check(len(L) == 31, f"ngpu={ngpu}: deck has {len(L)} lines, HPL expects 31")
    if len(L) != 31:
        continue
    p, q = int(L[10]), int(L[11])
    check(p * q == ngpu,
          f"ngpu={ngpu}: P*Q={p*q} != rank count; HPL refuses to start")
    check(int(L[18]) >= 2, f"ngpu={ngpu}: NDIV={L[18]} but HPL requires >= 2")
    check(int(L[16]) >= 1, f"ngpu={ngpu}: NBMIN={L[16]} but HPL requires >= 1")
    check(int(L[5]) > 0, f"ngpu={ngpu}: N={L[5]} is not a positive order")
    check(int(L[7]) > 0, f"ngpu={ngpu}: NB={L[7]} is not positive")
    check(L[3] == "6", f"ngpu={ngpu}: device-out is {L[3]}, expected 6 (stdout)")
    for i, v in enumerate(L[4:12], start=4):
        check(v.strip() != "", f"ngpu={ngpu}: line {i} is blank; every line is "
                               f"positional and a blank shifts the rest")

if FAILS:
    print("FAIL")
    for f in FAILS:
        print("  " + f)
    sys.exit(1)
print("OK: deck is 31 lines, P*Q matches the rank count, NDIV>=2, NBMIN>=1 "
      "for every GPU count")
