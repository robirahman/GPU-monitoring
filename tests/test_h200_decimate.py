import numpy as np

from h200_campaign.decimate import Decimator, StageDecimator, design_stage


def _reference_stage(h, m, x):
    c = (len(h) - 1) // 2
    full = np.convolve(np.concatenate([x, np.zeros(c)]), h)  # full[n] = sum h[j] x[n-j]
    k = np.arange(0, (len(x) - 1 - c) // m + 1)
    return full[k * m + c]


def test_stage_matches_reference_for_random_chunking():
    rng = np.random.default_rng(0)
    h = design_stage(19.53e6, 4, 400e3, 19.53e6 / 4 - 400e3)
    x = rng.standard_normal(200_003)
    st = StageDecimator(h, 4)
    out, i = [], 0
    while i < x.size:
        n = int(rng.integers(1, 9000))
        out.append(st.feed(x[i:i + n]))
        i += n
    got = np.concatenate(out)
    ref = _reference_stage(h, 4, x)
    assert got.size == ref.size
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)


def test_two_stage_index_alignment_and_rejection():
    fs = 1e9 / 51.2
    d = Decimator(fs)
    n = int(fs * 0.5)
    t = np.arange(n) / fs
    inband = np.sin(2 * np.pi * 50e3 * t)
    alias = np.sin(2 * np.pi * (d.fs_out + 100e3) * t)   # would fold onto 100 kHz
    y_in = np.concatenate([d.feed(c) for c in np.array_split(inband, 37)])
    d2 = Decimator(fs)
    y_al = np.concatenate([d2.feed(c) for c in np.array_split(alias, 41)])
    # centred: decimated k equals the raw value at 20k (away from start-up edge)
    k = np.arange(2000, y_in.size - 10)
    assert np.max(np.abs(y_in[k] - inband[20 * k])) < 1e-3
    assert np.sqrt(np.mean(y_al[2000:] ** 2)) < 1e-3   # >= ~60 dB below a unit sine (0.707 rms)
