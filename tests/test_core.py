import json, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foveakv.quant import NestedCodebook, lloyd_max, gaussian_mse, GainShape, LogGain, kivi_keys, kivi_values
from foveakv.rotate import rht_matrix, hadamard
from foveakv.attention import mixed, delta_corrected, exact, scores
from foveakv.select import select, always_hot, k_for
from foveakv.load import bf16_to_f32, load_config, snapshot_dir
from foveakv import kvdata

CB = NestedCodebook(4, 2)


def test_lloyd_max_known_values():
    b2, l2 = lloyd_max(4)
    assert abs(gaussian_mse(b2, l2) - 0.11748) < 1e-4          # Max (1960) 2-bit
    assert abs(CB.mse_fine - 0.009497) < 2e-5                  # Max (1960) 4-bit
    assert abs(CB.mse_coarse - 0.1203) < 1e-4                  # nested coarse (fine-first)
    np.testing.assert_allclose(l2[2:], [0.4528, 1.5104], atol=1e-3)


def test_lloyd_max_mse_empirical():
    z = np.random.default_rng(0).standard_normal(2_000_000)
    idx = CB.encode(z)
    assert abs(np.mean((CB.fine(idx) - z) ** 2) - CB.mse_fine) < 3e-4
    assert abs(np.mean((CB.coarse(idx) - z) ** 2) - CB.mse_coarse) < 1e-3


def test_nesting_exact():
    assert set(np.round(CB.bnd4, 12)) <= set(np.round(CB.bnd16, 12))
    z = np.random.default_rng(1).standard_normal(100_000) * 1.5
    idx = CB.encode(z)
    direct = CB.lut4[np.searchsorted(CB.bnd4, z)]
    np.testing.assert_array_equal(CB.coarse(idx), direct)
    hi, lo = idx >> 2, idx & 3
    np.testing.assert_array_equal((hi << 2) | lo, idx)
    np.testing.assert_array_equal(CB.sign(idx), np.where(z > 0, 1, -1))


def test_rht_orthogonal():
    for d in (64, 128, 256):
        R = rht_matrix(d, seed=3)
        np.testing.assert_allclose(R @ R.T, np.eye(d), atol=1e-12)
        np.testing.assert_allclose(hadamard(d) @ hadamard(d), np.eye(d), atol=1e-12)
    x, y = np.random.default_rng(2).standard_normal((2, 256))
    R = rht_matrix(256, 5)
    assert abs((x @ R.T) @ (y @ R.T) - x @ y) < 1e-10


def test_bf16_loader():
    raw = np.array([0x3F80, 0xC000, 0x3E20, 0x0000, 0x7F80], np.uint16)
    np.testing.assert_array_equal(bf16_to_f32(raw), [1.0, -2.0, 0.15625, 0.0, np.inf])
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    top = (x.view(np.uint32) >> 16).astype(np.uint16)
    assert np.all(np.abs(bf16_to_f32(top) - x) <= np.abs(x) * 2 ** -7)


def _rand_case(rng, E=7, T=50, Hkv=2, G=4, d=16):
    q = rng.standard_normal((E, Hkv * G, d)) * 2
    Kf, Vf = rng.standard_normal((2, T, Hkv, d))
    Kc, Vc = Kf + 0.3 * rng.standard_normal(Kf.shape), Vf + 0.3 * rng.standard_normal(Vf.shape)
    qpos = np.sort(rng.choice(np.arange(10, T), E, replace=False))
    return q, Kc, Kf, Vc, Vf, qpos


@pytest.mark.parametrize("shared", [True, False])
def test_delta_correction_equals_recompute(shared):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, Kc, Kf, Vc, Vf, qpos = _rand_case(rng)
        shape = (2, 7, 50) if shared else (2, 4, 7, 50)
        hot = rng.random(shape) < rng.uniform(0, 0.6)
        a = mixed(q, Kc, Kf, Vc, Vf, hot, qpos)
        b = delta_corrected(q, Kc, Kf, Vc, Vf, hot, qpos)
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
    # all hot == fine exact, none hot == coarse exact
    allh = np.ones((2, 7, 50), bool)
    np.testing.assert_allclose(delta_corrected(q, Kc, Kf, Vc, Vf, allh, qpos), exact(q, Kf, Vf, qpos)[0], atol=1e-10)
    np.testing.assert_allclose(delta_corrected(q, Kc, Kf, Vc, Vf, ~allh, qpos), exact(q, Kc, Vc, qpos)[0], atol=1e-10)


def test_rotated_domain_kernel_equivalence():
    """Kernel form: rotate q, score against mean-free rotated keys g*LUT/sqrt(d), accumulate values in the
    rotated domain, un-rotate once, add mu_V. Must equal attention on reconstructed keys/values."""
    rng = np.random.default_rng(4)
    d, T = 64, 200
    R = rht_matrix(d, 1)
    K = rng.standard_normal((T, d)) * rng.uniform(0.2, 5, d) + 3
    V = rng.standard_normal((T, d)) + 1
    qk = GainShape(R, CB, 8).fit(K[:150]); qv = GainShape(R, CB, 8).fit(V[:150])
    ik, gk, _, _ = qk.encode(K); iv, gv, _, _ = qv.encode(V)
    q = rng.standard_normal(d)
    Khat, Vhat = qk.decode(ik, gk), qv.decode(iv, gv)
    s = Khat @ q / np.sqrt(d); p = np.exp(s - s.max()); p /= p.sum(); ref = p @ Vhat
    qt = R @ q
    s2 = qk.decode_rot(ik, gk) @ qt / np.sqrt(d)          # mu_K never reconstructed
    p2 = np.exp(s2 - s2.max()); p2 /= p2.sum()
    out = (p2 @ qv.decode_rot(iv, gv)) @ R + qv.mu
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_select_budget_and_always_hot():
    rng = np.random.default_rng(0)
    T, E = 400, 5
    qpos = np.arange(T - E, T)
    s = scores(rng.standard_normal((E, 8, 16)), rng.standard_normal((T, 2, 16)), qpos)
    hot = select(s, qpos, 0.25, S=4, W=16)
    ah = always_hot(qpos, T, 4, 16)
    assert np.all(hot[:, ah])
    np.testing.assert_array_equal(hot.sum(-1), np.broadcast_to(k_for(qpos, 0.25), (2, E)))
    assert not np.any(hot & (np.arange(T)[None, None, :] > qpos[None, :, None]))
    u = select(s, qpos, 0.05, S=0, W=0, mode="union")
    assert np.all(u.sum(-1) >= k_for(qpos, 0.05)) and np.all(u.sum(-1) <= 4 * k_for(qpos, 0.05))


def test_kivi_shapes_and_error():
    rng = np.random.default_rng(0)
    K = rng.standard_normal((300, 64)); V = rng.standard_normal((300, 64))
    Kq, n = kivi_keys(K, 4, 32)
    assert n == 288 and np.array_equal(Kq[288:], K[288:])
    assert np.mean((Kq - K) ** 2) < 0.05 and np.mean((kivi_values(V, 2, 32) - V) ** 2) < 0.3


def test_config_matches_kvdata_constants():
    c = load_config(snapshot_dir())
    rp = c["rope_parameters"]
    assert int(c["head_dim"] * rp["partial_rotary_factor"]) == kvdata.ROT_DIM
    assert rp["rope_theta"] == kvdata.THETA and c["head_dim"] == kvdata.HD
